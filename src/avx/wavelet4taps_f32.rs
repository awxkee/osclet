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
use crate::avx::util::{_mm_hsum_ps, _mm256_hpadd2_ps, shuffle};
use crate::border_mode::BorderMode;
use crate::err::OscletError;
use crate::filter_padding::make_arena_1d;
use crate::util::{dwt_length, idwt_length, low_pass_to_high_from_arr};
use crate::{DwtForwardExecutor, DwtInverseExecutor, IncompleteDwtExecutor};
use std::arch::x86_64::*;

pub(crate) struct AvxWavelet4TapsF32 {
    border_mode: BorderMode,
    low_pass: [f32; 8],
    high_pass: [f32; 8],
}

impl AvxWavelet4TapsF32 {
    #[allow(unused)]
    pub(crate) fn new(border_mode: BorderMode, w: &[f32; 4]) -> Self {
        let g = low_pass_to_high_from_arr(w);
        Self {
            border_mode,
            low_pass: [w[0], w[1], w[2], w[3], w[0], w[1], w[2], w[3]],
            high_pass: [g[0], g[1], g[2], g[3], g[0], g[1], g[2], g[3]],
        }
    }
}

impl DwtForwardExecutor<f32> for AvxWavelet4TapsF32 {
    fn execute_forward(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_forward_impl(input, approx, details) }
    }
}

impl AvxWavelet4TapsF32 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_forward_impl(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
    ) -> Result<(), OscletError> {
        let half = dwt_length(input.len(), 4);

        if input.len() < 4 {
            return Err(OscletError::MinFilterSize(input.len(), 4));
        }

        if approx.len() != half {
            return Err(OscletError::ApproxDetailsSize(approx.len()));
        }
        if details.len() != half {
            return Err(OscletError::ApproxDetailsSize(details.len()));
        }

        let padded_input = make_arena_1d(
            input,
            2,
            if !input.len().is_multiple_of(2) { 3 } else { 2 },
            self.border_mode,
        )?;

        unsafe {
            let h = _mm256_loadu_ps(self.low_pass.as_ptr());
            let g = _mm256_loadu_ps(self.high_pass.as_ptr());

            let mut processed = 0usize;

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(4)
                .zip(details.chunks_exact_mut(4))
                .enumerate()
            {
                let base0 = 2 * 4 * i;

                let input0 = padded_input.get_unchecked(base0..);

                let xw00 = _mm_loadu_ps(input0.as_ptr());
                let xw01 = _mm_loadu_ps(input0.get_unchecked(2..).as_ptr());
                let xw02 = _mm_loadu_ps(input0.get_unchecked(4..).as_ptr());
                let xw03 = _mm_loadu_ps(input0.get_unchecked(6..).as_ptr());

                let xw0 = _mm256_setr_m128(xw00, xw01);
                let xw2 = _mm256_setr_m128(xw02, xw03);

                let a0 = _mm256_mul_ps(xw0, h);
                let d0 = _mm256_mul_ps(xw0, g);

                let a2 = _mm256_mul_ps(xw2, h);
                let d2 = _mm256_mul_ps(xw2, g);

                let wa0 = _mm256_hpadd2_ps(a0, a2);
                let wd0 = _mm256_hpadd2_ps(d0, d2);

                let wa2 = _mm_hadd_ps(_mm256_castps256_ps128(wa0), _mm256_extractf128_ps::<1>(wa0));
                let wd2 = _mm_hadd_ps(_mm256_castps256_ps128(wd0), _mm256_extractf128_ps::<1>(wd0));

                _mm_storeu_ps(approx.as_mut_ptr(), wa2);
                _mm_storeu_ps(detail.as_mut_ptr(), wd2);

                processed += 4;
            }

            let approx = approx.chunks_exact_mut(4).into_remainder();
            let details = details.chunks_exact_mut(4).into_remainder();
            let padded_input = padded_input.get_unchecked(processed * 2..);

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * i;

                let input = padded_input.get_unchecked(base..);

                let xw = _mm_loadu_ps(input.as_ptr().cast());

                let mut a = _mm_mul_ps(xw, _mm256_castps256_ps128(h));
                let mut d = _mm_mul_ps(xw, _mm256_castps256_ps128(g));

                a = _mm_hsum_ps(a);
                d = _mm_hsum_ps(d);

                _mm_store_ss(approx as *mut f32, a);
                _mm_store_ss(detail as *mut f32, d);
            }
        }
        Ok(())
    }
}

impl DwtInverseExecutor<f32> for AvxWavelet4TapsF32 {
    fn execute_inverse(
        &self,
        approx: &[f32],
        details: &[f32],
        output: &mut [f32],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_inverse_impl(approx, details, output) }
    }
}

impl AvxWavelet4TapsF32 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_inverse_impl(
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

        let rec_len = idwt_length(approx.len(), 4);

        if output.len() != rec_len {
            return Err(OscletError::OutputSizeIsTooSmall(output.len(), rec_len));
        }

        const FILTER_OFFSET: usize = 2;
        const FILTER_LENGTH: usize = 4;

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
                for j in 0..4 {
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

            let wh = _mm256_loadu_ps(self.low_pass.as_ptr());
            let wg = _mm256_loadu_ps(self.high_pass.as_ptr());

            let mut ui = safe_start;

            while ui + 2 < safe_end {
                let (h, g) = (
                    _mm_loadu_ps(approx.get_unchecked(ui)),
                    _mm_loadu_ps(details.get_unchecked(ui)),
                );
                let k = 2 * ui as isize - FILTER_OFFSET as isize;
                let part0 = output.get_unchecked_mut(k as usize..);
                let q0 = _mm_loadu_ps(part0.as_ptr());
                let q1 = _mm_loadu_ps(part0.get_unchecked(2..).as_ptr());

                let h0 = _mm_permute_ps::<{ shuffle(0, 0, 0, 0) }>(h);
                let g0 = _mm_permute_ps::<{ shuffle(0, 0, 0, 0) }>(g);

                let h1 = _mm_permute_ps::<{ shuffle(1, 1, 1, 1) }>(h);
                let g1 = _mm_permute_ps::<{ shuffle(1, 1, 1, 1) }>(g);

                let w0 = _mm_fmadd_ps(
                    _mm256_castps256_ps128(wg),
                    g0,
                    _mm_fmadd_ps(_mm256_castps256_ps128(wh), h0, q0),
                );
                let mut w1 = _mm_fmadd_ps(
                    _mm256_castps256_ps128(wg),
                    g1,
                    _mm_fmadd_ps(_mm256_castps256_ps128(wh), h1, q1),
                );
                let w0_hi = _mm_shuffle_ps::<{ shuffle(0, 0, 3, 2) }>(w0, _mm_setzero_ps());
                w1 = _mm_add_ps(w1, w0_hi);
                _mm_storeu_ps(part0.as_mut_ptr(), w0);
                _mm_storeu_ps(part0.get_unchecked_mut(2..).as_mut_ptr(), w1);
                ui += 2;
            }

            for i in ui..safe_end {
                let (h, g) = (
                    _mm_broadcast_ss(approx.get_unchecked(i)),
                    _mm_broadcast_ss(details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                let part = output.get_unchecked_mut(k as usize..);
                let q0 = _mm_loadu_ps(part.as_ptr());
                let w0 = _mm_fmadd_ps(
                    _mm256_castps256_ps128(wg),
                    g,
                    _mm_fmadd_ps(_mm256_castps256_ps128(wh), h, q0),
                );
                _mm_storeu_ps(part.as_mut_ptr(), w0);
            }

            for i in safe_end..approx.len() {
                let (h, g) = (
                    _mm_set_ss(*approx.get_unchecked(i)),
                    _mm_set_ss(*details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..4 {
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

impl IncompleteDwtExecutor<f32> for AvxWavelet4TapsF32 {
    fn filter_length(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod tests {

    fn has_avx_with_fma() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    use super::*;
    use crate::symlets::SymletFamily;
    use crate::{DaubechiesFamily, WaveletFilterProvider};

    #[test]
    fn test_db2_odd() {
        if !has_avx_with_fma() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db2 = AvxWavelet4TapsF32::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db2
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 4);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db2.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f32; 10] = [
            2.68446737, 2.31078903, 5.11383217, 1.67303261, 0.53329969, 6.3061913, 7.60071774,
            2.95715649, 1.7599028, 2.10398276,
        ];
        const REFERENCE_DETAILS: [f32; 10] = [
            -8.58001572e-01,
            1.00000000e-16,
            -9.47343455e-02,
            -9.65925826e-01,
            -1.35576367e+00,
            -2.85084151e+00,
            2.31500342e+00,
            -1.01700947e+00,
            1.25223606e+00,
            -3.23523806e-01,
        ];

        println!("approx {:?}", approx);

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
        db2.execute_inverse(&approx, &details, &mut reconstructed)
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
    fn test_db2_even() {
        if !has_avx_with_fma() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db4 = AvxWavelet4TapsF32::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db2
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 4);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f32; 9] = [
            1.29427747, 2.31078903, 5.11383217, 1.67303261, 0.53329969, 6.3061913, 7.60071774,
            2.95715649, 1.29427747,
        ];
        const REFERENCE_DETAILS: [f32; 9] = [
            -4.85501312e-01,
            1.00000000e-16,
            -9.47343455e-02,
            -9.65925826e-01,
            -1.35576367e+00,
            -2.85084151e+00,
            2.31500342e+00,
            -1.01700947e+00,
            -4.85501312e-01,
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
        db4.execute_inverse(&approx, &details, &mut reconstructed)
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
    fn test_sym2_even() {
        if !has_avx_with_fma() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db4 = AvxWavelet4TapsF32::new(
            BorderMode::Wrap,
            SymletFamily::Sym2
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 4);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f32; 9] = [
            1.29427747, 2.31078903, 5.11383217, 1.67303261, 0.53329969, 6.3061913, 7.60071774,
            2.95715649, 1.29427747,
        ];
        const REFERENCE_DETAILS: [f32; 9] = [
            -4.85501312e-01,
            1.00000000e-16,
            -9.47343455e-02,
            -9.65925826e-01,
            -1.35576367e+00,
            -2.85084151e+00,
            2.31500342e+00,
            -1.01700947e+00,
            -4.85501312e-01,
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
        db4.execute_inverse(&approx, &details, &mut reconstructed)
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
