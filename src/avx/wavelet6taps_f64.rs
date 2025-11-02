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
use crate::avx::util::{_mm256_hpadd2_pd, _mm256_hsum_pd};
use crate::border_mode::BorderMode;
use crate::err::OscletError;
use crate::filter_padding::make_arena_1d;
use crate::util::{dwt_length, idwt_length, low_pass_to_high_from_arr};
use crate::{DwtForwardExecutor, DwtInverseExecutor, IncompleteDwtExecutor};
use std::arch::x86_64::*;

pub(crate) struct AvxWavelet6TapsF64 {
    border_mode: BorderMode,
    low_pass: [f64; 8],
    high_pass: [f64; 8],
}

impl AvxWavelet6TapsF64 {
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[f64; 6]) -> Self {
        let g = low_pass_to_high_from_arr(wavelet);
        Self {
            border_mode,
            low_pass: [
                wavelet[0], wavelet[1], wavelet[2], wavelet[3], wavelet[4], wavelet[5], 0., 0.,
            ],
            high_pass: [g[0], g[1], g[2], g[3], g[4], g[5], 0., 0.],
        }
    }
}

impl DwtForwardExecutor<f64> for AvxWavelet6TapsF64 {
    fn execute_forward(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_forward_impl(input, approx, details) }
    }
}

impl AvxWavelet6TapsF64 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_forward_impl(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
    ) -> Result<(), OscletError> {
        let half = dwt_length(input.len(), 6);

        if input.len() < 6 {
            return Err(OscletError::MinFilterSize(input.len(), 6));
        }

        if approx.len() != half {
            return Err(OscletError::ApproxDetailsSize(approx.len()));
        }
        if details.len() != half {
            return Err(OscletError::ApproxDetailsSize(details.len()));
        }

        const FILTER_SIZE: usize = 6;

        let whole_size = (2 * half + FILTER_SIZE - 2) - input.len();
        let left_pad = whole_size / 2;
        let right_pad = whole_size - left_pad;

        let padded_input = make_arena_1d(input, left_pad, right_pad, self.border_mode)?;

        unsafe {
            let h0 = _mm256_loadu_pd(self.low_pass.as_ptr());
            let g0 = _mm256_loadu_pd(self.high_pass.as_ptr());

            let h2 = _mm256_loadu_pd(self.low_pass.get_unchecked(4..).as_ptr());
            let g2 = _mm256_loadu_pd(self.high_pass.get_unchecked(4..).as_ptr());

            let mut processed = 0usize;

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(4)
                .zip(details.chunks_exact_mut(4))
                .enumerate()
            {
                let base0 = 2 * 4 * i;

                let input0 = padded_input.get_unchecked(base0..);

                let xw00xw01 = _mm256_loadu_pd(input0.as_ptr());
                let xw02xw03 = _mm256_loadu_pd(input0.get_unchecked(4..).as_ptr());
                let xw04xw05 = _mm256_loadu_pd(input0.get_unchecked(8..).as_ptr());

                let a0 = _mm256_fmadd_pd(xw02xw03, h2, _mm256_mul_pd(xw00xw01, h0));
                let d0 = _mm256_fmadd_pd(xw02xw03, g2, _mm256_mul_pd(xw00xw01, g0));

                const HI_LO: i32 = 0b0010_0001;

                let xw01xw02 = _mm256_permute2f128_pd::<HI_LO>(xw00xw01, xw02xw03);
                let xw0300 = _mm256_permute2f128_pd::<HI_LO>(xw02xw03, _mm256_setzero_pd());

                let a1 = _mm256_fmadd_pd(xw0300, h2, _mm256_mul_pd(xw01xw02, h0));
                let d1 = _mm256_fmadd_pd(xw0300, g2, _mm256_mul_pd(xw01xw02, g0));

                let a2 = _mm256_fmadd_pd(xw04xw05, h2, _mm256_mul_pd(xw02xw03, h0));
                let d2 = _mm256_fmadd_pd(xw04xw05, g2, _mm256_mul_pd(xw02xw03, g0));

                let xw03xw04 = _mm256_permute2f128_pd::<HI_LO>(xw02xw03, xw04xw05);
                let xw0500 = _mm256_permute2f128_pd::<HI_LO>(xw04xw05, _mm256_setzero_pd());

                let a3 = _mm256_fmadd_pd(xw0500, h2, _mm256_mul_pd(xw03xw04, h0));
                let d3 = _mm256_fmadd_pd(xw0500, g2, _mm256_mul_pd(xw03xw04, g0));

                let q0 = _mm256_hpadd2_pd(a0, a1);
                let q1 = _mm256_hpadd2_pd(a2, a3);

                let xq = _mm256_hpadd2_pd(q0, q1);

                let fq0 = _mm256_hpadd2_pd(d0, d1);
                let fq1 = _mm256_hpadd2_pd(d2, d3);

                let xf = _mm256_hpadd2_pd(fq0, fq1);

                _mm256_storeu_pd(approx.as_mut_ptr(), xq);
                _mm256_storeu_pd(detail.as_mut_ptr(), xf);

                processed += 4;
            }

            let approx = approx.chunks_exact_mut(4).into_remainder();
            let details = details.chunks_exact_mut(4).into_remainder();
            let padded_input = padded_input.get_unchecked(processed * 2..);

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * i;
                let input = padded_input.get_unchecked(base..);

                let x01 = _mm256_loadu_pd(input.as_ptr());
                let x45 = _mm256_castpd128_pd256(_mm_loadu_pd(input.get_unchecked(4..).as_ptr()));

                let wa = _mm256_fmadd_pd(x45, h2, _mm256_mul_pd(x01, h0));
                let wd = _mm256_fmadd_pd(x45, g2, _mm256_mul_pd(x01, g0));

                let wa = _mm256_hsum_pd(wa);
                let wd = _mm256_hsum_pd(wd);

                _mm_storel_pd(approx as *mut f64, wa);
                _mm_storel_pd(detail as *mut f64, wd);
            }
        }
        Ok(())
    }
}

impl DwtInverseExecutor<f64> for AvxWavelet6TapsF64 {
    fn execute_inverse(
        &self,
        approx: &[f64],
        details: &[f64],
        output: &mut [f64],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_inverse_impl(approx, details, output) }
    }
}

impl AvxWavelet6TapsF64 {
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

        let rec_len = idwt_length(approx.len(), 6);

        if output.len() != rec_len {
            return Err(OscletError::OutputSizeIsTooSmall(output.len(), rec_len));
        }

        const FILTER_OFFSET: usize = 4;
        const FILTER_LENGTH: usize = 6;

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
                for j in 0..6 {
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

            let h0 = _mm256_loadu_pd(self.low_pass.as_ptr());
            let g0 = _mm256_loadu_pd(self.high_pass.as_ptr());

            let h2 = _mm256_loadu_pd(self.low_pass.get_unchecked(4..).as_ptr());
            let g2 = _mm256_loadu_pd(self.high_pass.get_unchecked(4..).as_ptr());

            let mut ui = safe_start;

            while ui + 2 < safe_end {
                let (h, g) = (
                    _mm_loadu_pd(approx.get_unchecked(ui)),
                    _mm_loadu_pd(details.get_unchecked(ui)),
                );
                let k = 2 * ui as isize - FILTER_OFFSET as isize;
                let part0 = output.get_unchecked_mut(k as usize..);
                let q0 = _mm256_loadu_pd(part0.as_ptr());
                let q2 = _mm256_loadu_pd(part0.get_unchecked(4..).as_ptr());

                let hh00 = _mm_permute_pd::<0>(h);
                let wh0 = _mm256_set_m128d(hh00, hh00);
                let gg00 = _mm_permute_pd::<0>(g);
                let wg0 = _mm256_set_m128d(gg00, gg00);

                let hh11 = _mm_permute_pd::<0b11>(h);
                let wh1 = _mm256_set_m128d(hh11, hh11);
                let gg11 = _mm_permute_pd::<0b11>(g);
                let wg1 = _mm256_set_m128d(gg11, gg11);

                const LO_LO: i32 = 0b0010_0000;

                let w0 = _mm256_fmadd_pd(g0, wg0, _mm256_fmadd_pd(h0, wh0, q0));
                let w2 = _mm256_fmadd_pd(g2, wg0, _mm256_fmadd_pd(h2, wh0, q2));

                const HI_LO: i32 = 0b0010_0001;

                // pass high parts from first results to next stage
                let interim = _mm256_permute2f128_pd::<HI_LO>(w0, w2);

                let w3 = _mm256_fmadd_pd(g0, wg1, _mm256_fmadd_pd(h0, wh1, interim));
                let w5 = _mm256_fmadd_pd(
                    g2,
                    wg1,
                    _mm256_fmadd_pd(
                        h2,
                        wh1,
                        _mm256_permute2f128_pd::<HI_LO>(q2, _mm256_setzero_pd()),
                    ),
                );

                // permute to use just 2 stores
                let xw01 = _mm256_permute2f128_pd::<LO_LO>(w0, w3);
                let xw02 = _mm256_permute2f128_pd::<HI_LO>(w3, w5);

                _mm256_storeu_pd(part0.as_mut_ptr(), xw01);
                _mm256_storeu_pd(part0.get_unchecked_mut(4..).as_mut_ptr(), xw02);
                ui += 2;
            }

            for i in ui..safe_end {
                let (h, g) = (
                    _mm256_set1_pd(*approx.get_unchecked(i)),
                    _mm256_set1_pd(*details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                let part = output.get_unchecked_mut(k as usize..);

                let w0 = _mm256_loadu_pd(part.as_ptr());
                let w2 = _mm_loadu_pd(part.get_unchecked(4..).as_ptr());

                let q0 = _mm256_fmadd_pd(g0, g, _mm256_fmadd_pd(h0, h, w0));
                let q4 = _mm_fmadd_pd(
                    _mm256_castpd256_pd128(g2),
                    _mm256_castpd256_pd128(g),
                    _mm_fmadd_pd(_mm256_castpd256_pd128(h2), _mm256_castpd256_pd128(h), w2),
                );

                _mm256_storeu_pd(part.as_mut_ptr(), q0);
                _mm_storeu_pd(part.get_unchecked_mut(4..).as_mut_ptr(), q4);
            }

            for i in safe_end..approx.len() {
                let (h, g) = (
                    _mm_set1_pd(*approx.get_unchecked(i)),
                    _mm_set1_pd(*details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..6 {
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

impl IncompleteDwtExecutor<f64> for AvxWavelet6TapsF64 {
    fn filter_length(&self) -> usize {
        6
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coiflet::CoifletFamily;
    use crate::{DaubechiesFamily, WaveletFilterProvider};

    fn has_avx_with_fma() -> bool {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }

    #[test]
    fn test_db3_odd() {
        if !has_avx_with_fma() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db3 = AvxWavelet6TapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db3
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 6);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db3.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 11] = [
            0.8483726, 2.5241373, 2.65038574, 5.04554797, 1.36113344, 1.0534151, 5.85968077,
            8.27594493, 2.09006931, 2.1647733, 1.88197732,
        ];
        const REFERENCE_DETAILS: [f64; 11] = [
            -1.11980127,
            0.29462366,
            -0.75732176,
            -0.61512612,
            -0.52980262,
            -3.42034085,
            1.36890244,
            -0.37564252,
            1.25365344,
            -0.05294689,
            1.08607739,
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 6)];
        db3.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-7,
                "reconstructed difference expected to be < 1e-7, but values were ref {}, derived {} at {i}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_db3_even() {
        if !has_avx_with_fma() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db3 = AvxWavelet6TapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db3
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 6);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db3.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 10] = [
            2.25345752, 1.28973105, 2.65038574, 5.04554797, 1.36113344, 1.0534151, 5.85968077,
            8.27594493, 2.25345752, 1.28973105,
        ];
        const REFERENCE_DETAILS: [f64; 10] = [
            -0.28935438,
            0.16391309,
            -0.75732176,
            -0.61512612,
            -0.52980262,
            -3.42034085,
            1.36890244,
            -0.37564252,
            -0.28935438,
            0.16391309,
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 6)];
        db3.execute_inverse(&approx, &details, &mut reconstructed)
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
    fn test_coif1_even() {
        if !has_avx_with_fma() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db3 = AvxWavelet6TapsF64::new(
            BorderMode::Wrap,
            CoifletFamily::Coif1
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 6);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db3.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 10] = [
            0.64709521, 1.74438159, 4.53911719, 3.20774595, 0.30097675, 4.61093707, 6.14348133,
            6.59556141, 0.64709521, 1.74438159,
        ];
        const REFERENCE_DETAILS: [f64; 10] = [
            -0.47031852,
            0.07106881,
            -1.37735608,
            0.23385359,
            0.47256556,
            -3.26886672,
            -2.2968896,
            2.18117025,
            -0.47031852,
            0.07106881,
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 6)];
        db3.execute_inverse(&approx, &details, &mut reconstructed)
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
}
