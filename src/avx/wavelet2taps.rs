/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::avx::util::{_mm_hsum_pd, _mm_hsum_ps, _mm256_hpadd2_ps, shuffle};
use crate::err::OscletError;
use crate::filter_padding::make_arena_1d;
use crate::util::{dwt_length, idwt_length, low_pass_to_high_from_arr};
use crate::{BorderMode, DwtForwardExecutor, DwtInverseExecutor, IncompleteDwtExecutor};
use std::arch::x86_64::*;

pub(crate) struct AvxWavelet2TapsF64 {
    border_mode: BorderMode,
    low_pass: [f64; 4],
    high_pass: [f64; 4],
}

impl AvxWavelet2TapsF64 {
    pub fn new(border_mode: BorderMode, wavelet: &[f64; 2]) -> Self {
        let g = low_pass_to_high_from_arr(wavelet);
        Self {
            border_mode,
            low_pass: [wavelet[0], wavelet[1], wavelet[0], wavelet[1]],
            high_pass: [g[0], g[1], g[0], g[1]],
        }
    }
}

impl DwtForwardExecutor<f64> for AvxWavelet2TapsF64 {
    fn execute_forward(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_forward_impl(input, approx, details) }
    }
}

impl AvxWavelet2TapsF64 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_forward_impl(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
    ) -> Result<(), OscletError> {
        let half = dwt_length(input.len(), 2);

        if input.len() < 2 {
            return Err(OscletError::MinFilterSize(input.len(), 2));
        }

        if approx.len() != half {
            return Err(OscletError::ApproxDetailsSize(approx.len()));
        }
        if details.len() != half {
            return Err(OscletError::ApproxDetailsSize(details.len()));
        }

        let padded_input = make_arena_1d(
            input,
            0,
            if !input.len().is_multiple_of(2) { 1 } else { 0 },
            self.border_mode,
        )?;

        unsafe {
            let l0 = _mm256_loadu_pd(self.low_pass.as_ptr());
            let h0 = _mm256_loadu_pd(self.high_pass.as_ptr());

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(2)
                .zip(details.chunks_exact_mut(2))
                .enumerate()
            {
                let base0 = 2 * 2 * i;

                let input0 = padded_input.get_unchecked(base0..);

                let xw0 = _mm256_loadu_pd(input0.as_ptr());

                let a0 = _mm256_mul_pd(xw0, l0);
                let d0 = _mm256_mul_pd(xw0, h0);

                let wa = _mm256_hadd_pd(a0, a0);
                let wd = _mm256_hadd_pd(d0, d0);

                let wa0 = _mm256_permute4x64_pd::<{ shuffle(3, 1, 2, 0) }>(wa);
                let wd0 = _mm256_permute4x64_pd::<{ shuffle(3, 1, 2, 0) }>(wd);

                _mm_storeu_pd(approx.as_mut_ptr(), _mm256_castpd256_pd128(wa0));
                _mm_storeu_pd(detail.as_mut_ptr(), _mm256_castpd256_pd128(wd0));
            }

            let processed = 2 * approx.chunks_exact_mut(2).len() * 2;

            let approx = approx.chunks_exact_mut(2).into_remainder();
            let details = details.chunks_exact_mut(2).into_remainder();

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = processed + 2 * i;

                let input = padded_input.get_unchecked(base..);

                let xw = _mm_loadu_pd(input.as_ptr());

                let wa = _mm_mul_pd(xw, _mm256_castpd256_pd128(l0));
                let wd = _mm_mul_pd(xw, _mm256_castpd256_pd128(h0));

                let a = _mm_hsum_pd(wa);
                let d = _mm_hsum_pd(wd);

                _mm_storel_pd(approx as *mut f64, a);
                _mm_storel_pd(detail as *mut f64, d);
            }
        }
        Ok(())
    }
}
impl DwtInverseExecutor<f64> for AvxWavelet2TapsF64 {
    fn execute_inverse(
        &self,
        approx: &[f64],
        details: &[f64],
        output: &mut [f64],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_inverse_impl(approx, details, output) }
    }
}

impl AvxWavelet2TapsF64 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_inverse_impl(
        &self,
        approx: &[f64],
        details: &[f64],
        output: &mut [f64],
    ) -> Result<(), OscletError> {
        if approx.len() != details.len() {
            return Err(OscletError::ApproxDetailsNotMatches(
                approx.len(),
                details.len(),
            ));
        }

        let rec_len = idwt_length(approx.len(), 2);

        if output.len() != rec_len {
            return Err(OscletError::OutputSizeIsTooSmall(output.len(), rec_len));
        }

        const FILTER_OFFSET: usize = 0;
        const FILTER_LENGTH: usize = 2;

        unsafe {
            let safe_start = FILTER_OFFSET;
            // 2*x - off + len >= output.len()
            // x >= (output.len() + off - len)/2
            let safe_end = ((output.len() + FILTER_OFFSET).saturating_sub(FILTER_LENGTH)) / 2;

            for i in 0..safe_start.min(safe_end) {
                let (h, g) = (
                    _mm_set1_pd(*approx.get_unchecked(i)),
                    _mm_set1_pd(*details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..2 {
                    let k = k + j as isize;
                    if k >= 0 && k < rec_len as isize {
                        let mut w = _mm_fmadd_sd(
                            _mm_set1_pd(self.high_pass[j]),
                            g,
                            _mm_load_sd(output.get_unchecked(k as usize) as *const f64),
                        );
                        w = _mm_fmadd_sd(_mm_set1_pd(self.low_pass[j]), h, w);
                        _mm_store_sd(output.get_unchecked_mut(k as usize) as *mut f64, w);
                    }
                }
            }

            let l0 = _mm256_loadu_pd(self.low_pass.as_ptr());
            let h0 = _mm256_loadu_pd(self.high_pass.as_ptr());

            let mut uq = safe_start;

            while uq + 2 < safe_end {
                let (h, g) = (
                    _mm_loadu_pd(approx.get_unchecked(uq..).as_ptr()),
                    _mm_loadu_pd(details.get_unchecked(uq..).as_ptr()),
                );
                let k0 = 2 * uq as isize - FILTER_OFFSET as isize;
                let part0_src = output.get_unchecked(k0 as usize..);
                let xw01 = _mm256_loadu_pd(part0_src.as_ptr());

                let wh0 = _mm256_permute4x64_pd::<{ shuffle(1, 1, 0, 0) }>(_mm256_set_m128d(h, h));
                let wg0 = _mm256_permute4x64_pd::<{ shuffle(1, 1, 0, 0) }>(_mm256_set_m128d(g, g));

                let q0 = _mm256_fmadd_pd(l0, wh0, _mm256_fmadd_pd(wg0, h0, xw01));

                _mm256_storeu_pd(output.get_unchecked_mut(k0 as usize..).as_mut_ptr(), q0);
                uq += 2;
            }

            for i in uq..safe_end {
                let (h, g) = (
                    _mm_set1_pd(*approx.get_unchecked(i)),
                    _mm_set1_pd(*details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                let part = output.get_unchecked_mut(k as usize..);
                let xw = _mm_loadu_pd(part.as_ptr());
                let q = _mm_fmadd_pd(
                    _mm256_castpd256_pd128(l0),
                    h,
                    _mm_fmadd_pd(_mm256_castpd256_pd128(h0), g, xw),
                );
                _mm_storeu_pd(part.as_mut_ptr(), q);
            }

            for i in safe_end..approx.len() {
                let (h, g) = (
                    _mm_set1_pd(*approx.get_unchecked(i)),
                    _mm_set1_pd(*details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..2 {
                    let k = k + j as isize;
                    if k >= 0 && k < rec_len as isize {
                        let mut w = _mm_fmadd_sd(
                            _mm_set1_pd(self.high_pass[j]),
                            g,
                            _mm_load_sd(output.get_unchecked(k as usize) as *const f64),
                        );
                        w = _mm_fmadd_sd(_mm_set1_pd(self.low_pass[j]), h, w);
                        _mm_store_sd(output.get_unchecked_mut(k as usize) as *mut f64, w);
                    }
                }
            }
        }
        Ok(())
    }
}

impl IncompleteDwtExecutor<f64> for AvxWavelet2TapsF64 {
    fn filter_length(&self) -> usize {
        2
    }
}

pub(crate) struct AvxWavelet2TapsF32 {
    border_mode: BorderMode,
    low_pass: [f32; 8],
    high_pass: [f32; 8],
}

impl AvxWavelet2TapsF32 {
    pub fn new(border_mode: BorderMode, wavelet: &[f32; 2]) -> Self {
        let k = low_pass_to_high_from_arr(wavelet);
        Self {
            border_mode,
            low_pass: [
                wavelet[0], wavelet[1], wavelet[0], wavelet[1], wavelet[0], wavelet[1], wavelet[0],
                wavelet[1],
            ],
            high_pass: [k[0], k[1], k[0], k[1], k[0], k[1], k[0], k[1]],
        }
    }
}

impl DwtForwardExecutor<f32> for AvxWavelet2TapsF32 {
    fn execute_forward(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
    ) -> Result<(), OscletError> {
        let half = dwt_length(input.len(), 2);

        if input.len() < 2 {
            return Err(OscletError::MinFilterSize(input.len(), 2));
        }

        if approx.len() != half {
            return Err(OscletError::ApproxDetailsSize(approx.len()));
        }
        if details.len() != half {
            return Err(OscletError::ApproxDetailsSize(details.len()));
        }

        let padded_input = make_arena_1d(
            input,
            0,
            if !input.len().is_multiple_of(2) { 1 } else { 0 },
            self.border_mode,
        )?;

        unsafe {
            let l0 = _mm256_loadu_ps(self.low_pass.as_ptr());
            let h0 = _mm256_loadu_ps(self.high_pass.as_ptr());

            let mut processed = 0usize;

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(8)
                .zip(details.chunks_exact_mut(8))
                .enumerate()
            {
                let base0 = 2 * 8 * i;

                let input0 = padded_input.get_unchecked(base0..);

                let xw0 = _mm256_loadu_ps(input0.as_ptr());
                let xw2 = _mm256_loadu_ps(input0.get_unchecked(8..).as_ptr());

                let a0 = _mm256_mul_ps(xw0, l0);
                let d0 = _mm256_mul_ps(xw0, h0);

                let a2 = _mm256_mul_ps(xw2, l0);
                let d2 = _mm256_mul_ps(xw2, h0);

                let mut wa = _mm256_hadd_ps(a0, a2);
                let mut wd = _mm256_hadd_ps(d0, d2);

                wa = _mm256_castpd_ps(_mm256_permute4x64_pd::<{ shuffle(3, 1, 2, 0) }>(
                    _mm256_castps_pd(wa),
                ));
                wd = _mm256_castpd_ps(_mm256_permute4x64_pd::<{ shuffle(3, 1, 2, 0) }>(
                    _mm256_castps_pd(wd),
                ));

                _mm256_storeu_ps(approx.as_mut_ptr(), wa);
                _mm256_storeu_ps(detail.as_mut_ptr(), wd);

                processed += 8;
            }

            let approx = approx.chunks_exact_mut(8).into_remainder();
            let details = details.chunks_exact_mut(8).into_remainder();

            let padded_input = padded_input.get_unchecked(processed * 2..);
            processed = 0;

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(4)
                .zip(details.chunks_exact_mut(4))
                .enumerate()
            {
                let base0 = 2 * 4 * i;

                let input0 = padded_input.get_unchecked(base0..);

                let xw0 = _mm256_loadu_ps(input0.as_ptr());

                let a0 = _mm256_mul_ps(xw0, l0);
                let d0 = _mm256_mul_ps(xw0, h0);

                let wa = _mm256_hpadd2_ps(a0, a0);
                let wd = _mm256_hpadd2_ps(d0, d0);

                _mm_storeu_ps(approx.as_mut_ptr(), _mm256_castps256_ps128(wa));
                _mm_storeu_ps(detail.as_mut_ptr(), _mm256_castps256_ps128(wd));

                processed += 4;
            }

            let approx = approx.chunks_exact_mut(4).into_remainder();
            let details = details.chunks_exact_mut(4).into_remainder();
            let padded_input = padded_input.get_unchecked(processed * 2..);

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * i;

                let input = padded_input.get_unchecked(base..);

                let xw = _mm_castsi128_ps(_mm_loadu_si64(input.as_ptr().cast()));

                let wa = _mm_mul_ps(xw, _mm256_castps256_ps128(l0));
                let wd = _mm_mul_ps(xw, _mm256_castps256_ps128(h0));

                let a = _mm_hsum_ps(wa);
                let d = _mm_hsum_ps(wd);

                _mm_store_ss(approx as *mut f32, a);
                _mm_store_ss(detail as *mut f32, d);
            }
        }
        Ok(())
    }
}

impl DwtInverseExecutor<f32> for AvxWavelet2TapsF32 {
    fn execute_inverse(
        &self,
        approx: &[f32],
        details: &[f32],
        output: &mut [f32],
    ) -> Result<(), OscletError> {
        if approx.len() != details.len() {
            return Err(OscletError::ApproxDetailsNotMatches(
                approx.len(),
                details.len(),
            ));
        }

        let rec_len = idwt_length(approx.len(), 2);

        if output.len() != rec_len {
            return Err(OscletError::OutputSizeIsTooSmall(output.len(), rec_len));
        }

        const FILTER_OFFSET: usize = 0;
        const FILTER_LENGTH: usize = 2;

        unsafe {
            let safe_start = FILTER_OFFSET;
            // 2*x - off + len >= output.len()
            // x >= (output.len() + off - len)/2
            let safe_end = ((output.len() + FILTER_OFFSET).saturating_sub(FILTER_LENGTH)) / 2;
            for i in 0..safe_start.min(safe_end) {
                let (h, g) = (
                    _mm_set_ss(*approx.get_unchecked(i)),
                    _mm_set_ss(*details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..2 {
                    let k = k + j as isize;
                    if k >= 0 && k < rec_len as isize {
                        let mut w = _mm_fmadd_ss(
                            _mm_set1_ps(self.high_pass[j]),
                            g,
                            _mm_load_ss(output.get_unchecked(k as usize) as *const f32),
                        );
                        w = _mm_fmadd_ss(_mm_set1_ps(self.low_pass[j]), h, w);
                        _mm_store_ss(output.get_unchecked_mut(k as usize) as *mut f32, w);
                    }
                }
            }

            let l0 = _mm256_loadu_ps(self.low_pass.as_ptr());
            let h0 = _mm256_loadu_ps(self.high_pass.as_ptr());

            let mut uq = safe_start;

            while uq + 2 < safe_end {
                let (h, g) = (
                    _mm_castsi128_ps(_mm_loadu_si64(approx.get_unchecked(uq..).as_ptr().cast())),
                    _mm_castsi128_ps(_mm_loadu_si64(details.get_unchecked(uq..).as_ptr().cast())),
                );
                let fh = _mm_permute_ps::<{ shuffle(1, 1, 0, 0) }>(h);
                let fg = _mm_permute_ps::<{ shuffle(1, 1, 0, 0) }>(g);
                let k0 = 2 * uq as isize - FILTER_OFFSET as isize;
                let part0_src = output.get_unchecked(k0 as usize..);
                let xw0 = _mm_loadu_ps(part0_src.as_ptr());
                let q0 = _mm_fmadd_ps(
                    _mm256_castps256_ps128(l0),
                    fh,
                    _mm_fmadd_ps(_mm256_castps256_ps128(h0), fg, xw0),
                );
                _mm_storeu_ps(output.get_unchecked_mut(k0 as usize..).as_mut_ptr(), q0);
                uq += 2;
            }

            for i in uq..safe_end {
                let (h, g) = (
                    _mm_set1_ps(*approx.get_unchecked(i)),
                    _mm_set1_ps(*details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                let part = output.get_unchecked_mut(k as usize..);
                let xw = _mm_castsi128_ps(_mm_loadu_si64(part.as_ptr().cast()));
                let q = _mm_fmadd_ps(
                    _mm256_castps256_ps128(l0),
                    h,
                    _mm_fmadd_ps(_mm256_castps256_ps128(h0), g, xw),
                );
                _mm_storel_pd(part.as_mut_ptr().cast(), _mm_castps_pd(q));
            }

            for i in safe_end..approx.len() {
                let (h, g) = (
                    _mm_set_ss(*approx.get_unchecked(i)),
                    _mm_set_ss(*details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..2 {
                    let k = k + j as isize;
                    if k >= 0 && k < rec_len as isize {
                        let mut w = _mm_fmadd_ss(
                            _mm_set1_ps(self.high_pass[j]),
                            g,
                            _mm_load_ss(output.get_unchecked(k as usize) as *const f32),
                        );
                        w = _mm_fmadd_ss(_mm_set1_ps(self.low_pass[j]), h, w);
                        _mm_store_ss(output.get_unchecked_mut(k as usize) as *mut f32, w);
                    }
                }
            }
        }
        Ok(())
    }
}

impl IncompleteDwtExecutor<f32> for AvxWavelet2TapsF32 {
    fn filter_length(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DaubechiesFamily, WaveletFilterProvider};

    fn has_avx_with_fma() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    #[test]
    fn test_db1_odd() {
        if !has_avx_with_fma() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db1 = AvxWavelet2TapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db1
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 2);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];

        db1.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 9] = [
            2.121320343559643,
            4.949747468305834,
            2.121320343559643,
            0.7071067811865476,
            6.293250352560273,
            6.222539674441618,
            4.101219330881976,
            1.272792206135786,
            2.474873734152916,
        ];
        const REFERENCE_DETAILS: [f64; 9] = [
            -0.7071067811865476,
            -0.7071067811865475,
            0.7071067811865476,
            -0.7071067811865476,
            -2.899137802864845,
            -2.828427124746191,
            3.252691193458119,
            -0.5656854249492381,
            1.060660171779821,
        ];

        approx.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_APPROX[i] - x).abs() < 1e-7,
                "approx difference expected to be < 1e-7, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-7,
                "details difference expected to be < 1e-7, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![
            0.0;
            if input.len() % 2 != 0 {
                input.len() + 1
            } else {
                input.len()
            }
        ];
        db1.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-7,
                "reconstructed difference expected to be < 1e-7, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_db1_even() {
        if !has_avx_with_fma() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db1 = AvxWavelet2TapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db1
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 2);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db1.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 8] = [
            2.12132034, 4.94974747, 2.12132034, 0.70710678, 6.29325035, 6.22253967, 4.10121933,
            1.27279221,
        ];
        const REFERENCE_DETAILS: [f64; 8] = [
            -0.70710678,
            -0.70710678,
            0.70710678,
            -0.70710678,
            -2.8991378,
            -2.82842712,
            3.25269119,
            -0.56568542,
        ];

        approx.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_APPROX[i] - x).abs() < 1e-7,
                "approx difference expected to be < 1e-7, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-7,
                "details difference expected to be < 1e-7, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 2)];
        db1.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-7,
                "reconstructed difference expected to be < 1e-7, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_db1_odd_f32() {
        if !has_avx_with_fma() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db1 = AvxWavelet2TapsF32::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db1
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 2);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db1.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f32; 9] = [
            2.121320343559643,
            4.949747468305834,
            2.121320343559643,
            0.7071067811865476,
            6.293250352560273,
            6.222539674441618,
            4.101219330881976,
            1.272792206135786,
            2.474873734152916,
        ];
        const REFERENCE_DETAILS: [f32; 9] = [
            -0.7071067811865476,
            -0.7071067811865475,
            0.7071067811865476,
            -0.7071067811865476,
            -2.899137802864845,
            -2.828427124746191,
            3.252691193458119,
            -0.5656854249492381,
            1.060660171779821,
        ];

        approx.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_APPROX[i] - x).abs() < 1e-4,
                "approx difference expected to be < 1e-4, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-4,
                "details difference expected to be < 1e-4, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![
            0.0;
            if input.len() % 2 != 0 {
                input.len() + 1
            } else {
                input.len()
            }
        ];
        db1.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-4,
                "reconstructed difference expected to be < 1e-4, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_db1_even_f32() {
        if !has_avx_with_fma() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db1 = AvxWavelet2TapsF32::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db1
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 2);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db1.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f32; 8] = [
            2.12132034, 4.94974747, 2.12132034, 0.70710678, 6.29325035, 6.22253967, 4.10121933,
            1.27279221,
        ];
        const REFERENCE_DETAILS: [f32; 8] = [
            -0.70710678,
            -0.70710678,
            0.70710678,
            -0.70710678,
            -2.8991378,
            -2.82842712,
            3.25269119,
            -0.56568542,
        ];

        approx.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_APPROX[i] - x).abs() < 1e-4,
                "approx difference expected to be < 1e-4, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-4,
                "details difference expected to be < 1e-4, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 2)];
        db1.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-4,
                "reconstructed difference expected to be < 1e-4, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_db1_even_f32_2() {
        if !has_avx_with_fma() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.0,
            1.0,
        ];
        let db1 = AvxWavelet2TapsF32::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db1
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 2);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db1.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f32; 9] = [
            2.12132034, 4.94974747, 2.12132034, 0.70710678, 6.29325035, 6.22253967, 4.10121933,
            1.27279221, 2.12132034,
        ];
        const REFERENCE_DETAILS: [f32; 9] = [
            -0.70710678,
            -0.70710678,
            0.70710678,
            -0.70710678,
            -2.8991378,
            -2.82842712,
            3.25269119,
            -0.56568542,
            0.70710678,
        ];

        approx.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_APPROX[i] - x).abs() < 1e-4,
                "approx difference expected to be < 1e-4, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-4,
                "details difference expected to be < 1e-4, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 2)];
        db1.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-4,
                "reconstructed difference expected to be < 1e-4, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }
}
