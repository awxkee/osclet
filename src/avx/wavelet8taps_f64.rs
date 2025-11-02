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
use crate::avx::util::_mm256_hsum_pd;
use crate::border_mode::BorderMode;
use crate::err::OscletError;
use crate::filter_padding::make_arena_1d;
use crate::util::{dwt_length, idwt_length, low_pass_to_high_from_arr};
use crate::{DwtForwardExecutor, DwtInverseExecutor, IncompleteDwtExecutor};
use std::arch::x86_64::*;

pub(crate) struct AvxWavelet8TapsF64 {
    border_mode: BorderMode,
    low_pass: [f64; 8],
    high_pass: [f64; 8],
}

impl AvxWavelet8TapsF64 {
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[f64; 8]) -> Self {
        Self {
            border_mode,
            low_pass: *wavelet,
            high_pass: low_pass_to_high_from_arr(wavelet),
        }
    }
}

impl DwtForwardExecutor<f64> for AvxWavelet8TapsF64 {
    fn execute_forward(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_forward_impl(input, approx, details) }
    }
}

impl AvxWavelet8TapsF64 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_forward_impl(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
    ) -> Result<(), OscletError> {
        let half = dwt_length(input.len(), 8);

        if input.len() < 8 {
            return Err(OscletError::MinFilterSize(input.len(), 8));
        }

        if approx.len() != half {
            return Err(OscletError::ApproxDetailsSize(approx.len()));
        }
        if details.len() != half {
            return Err(OscletError::ApproxDetailsSize(details.len()));
        }

        const FILTER_SIZE: usize = 8;

        let whole_size = (2 * half + FILTER_SIZE - 2) - input.len();
        let left_pad = whole_size / 2;
        let right_pad = whole_size - left_pad;

        let padded_input = make_arena_1d(input, left_pad, right_pad, self.border_mode)?;

        unsafe {
            let h0 = _mm256_loadu_pd(self.low_pass.as_ptr());
            let g0 = _mm256_loadu_pd(self.high_pass.as_ptr());

            let h2 = _mm256_loadu_pd(self.low_pass.get_unchecked(4..).as_ptr());
            let g2 = _mm256_loadu_pd(self.high_pass.get_unchecked(4..).as_ptr());

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * i;
                let input = padded_input.get_unchecked(base..);

                let x01 = _mm256_loadu_pd(input.as_ptr());
                let x45 = _mm256_loadu_pd(input.get_unchecked(4..).as_ptr());

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

impl DwtInverseExecutor<f64> for AvxWavelet8TapsF64 {
    fn execute_inverse(
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

        let rec_len = idwt_length(approx.len(), 8);

        if output.len() != rec_len {
            return Err(OscletError::OutputSizeIsTooSmall(output.len(), rec_len));
        }

        const FILTER_OFFSET: usize = 6;
        const FILTER_LENGTH: usize = 8;

        unsafe {
            let safe_start = FILTER_OFFSET;
            // 2*x - off + len >= output.len()
            // x >= (output.len() + off - len)/2
            let safe_end = ((output.len() + FILTER_OFFSET).saturating_sub(FILTER_LENGTH)) / 2;
            for i in 0..safe_start {
                let (h, g) = (
                    _mm_set1_pd(*approx.get_unchecked(i)),
                    _mm_set1_pd(*details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..8 {
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

            for i in safe_start..safe_end {
                let (h, g) = (
                    _mm256_set1_pd(*approx.get_unchecked(i)),
                    _mm256_set1_pd(*details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                let part = output.get_unchecked_mut(k as usize..);

                let w0 = _mm256_loadu_pd(part.as_ptr());
                let w2 = _mm256_loadu_pd(part.get_unchecked(4..).as_ptr());

                let q0 = _mm256_fmadd_pd(g0, g, _mm256_fmadd_pd(h0, h, w0));
                let q4 = _mm256_fmadd_pd(g2, g, _mm256_fmadd_pd(h2, h, w2));

                _mm256_storeu_pd(part.as_mut_ptr(), q0);
                _mm256_storeu_pd(part.get_unchecked_mut(4..).as_mut_ptr(), q4);
            }

            for i in safe_end..approx.len() {
                let (h, g) = (
                    _mm_set1_pd(*approx.get_unchecked(i)),
                    _mm_set1_pd(*details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..8 {
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

impl IncompleteDwtExecutor<f64> for AvxWavelet8TapsF64 {
    fn filter_length(&self) -> usize {
        8
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
    fn test_db8_odd() {
        if !has_avx_with_fma() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db4 = AvxWavelet8TapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db4
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 8);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 12] = [
            5.40180316, 1.17674293, 2.27895053, 3.08695254, 4.82517499, 0.91029972, 1.96020043,
            5.58301587, 8.40990105, 1.50316223, 2.42249936, 1.81502786,
        ];
        const REFERENCE_DETAILS: [f64; 12] = [
            -1.48628267,
            0.41816403,
            -1.0992322,
            -0.15292615,
            -0.32731146,
            -3.79371528,
            0.7310401,
            0.56575684,
            0.75931276,
            0.22144916,
            0.66611246,
            -0.09543936,
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 8)];
        db4.execute_inverse(&approx, &details, &mut reconstructed)
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
    fn test_db8_even() {
        if !has_avx_with_fma() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db4 = AvxWavelet8TapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db4
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 8);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 11] = [
            8.34997913, 1.83684144, 1.23683239, 3.08695254, 4.82517499, 0.91029972, 1.96020043,
            5.58301587, 8.34997913, 1.83684144, 1.23683239,
        ];
        const REFERENCE_DETAILS: [f64; 11] = [
            -0.54333491,
            0.1170128,
            -1.05129466,
            -0.15292615,
            -0.32731146,
            -3.79371528,
            0.7310401,
            0.56575684,
            -0.54333491,
            0.1170128,
            -1.05129466,
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 8)];
        db4.execute_inverse(&approx, &details, &mut reconstructed)
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
    fn test_db8_even_big() {
        if !has_avx_with_fma() {
            return;
        }
        let data_length = 86;
        let mut input = vec![0.; data_length];
        for i in 0..data_length {
            input[i] = i as f64 / data_length as f64;
        }
        let db4 = AvxWavelet8TapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db4
                .get_wavelet()
                .as_slice()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 8);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 8)];
        db4.execute_inverse(&approx, &details, &mut reconstructed)
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
