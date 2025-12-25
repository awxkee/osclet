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
use crate::avx::util::{_mm256_hsum_ps, shuffle};
use crate::border_mode::{BorderInterpolation, BorderMode};
use crate::err::{OscletError, try_vec};
use crate::mla::fmla;
use crate::util::{dwt_length, idwt_length, low_pass_to_high_from_arr, ten_taps_size_for_input};
use crate::{DwtForwardExecutor, DwtInverseExecutor, DwtSize, IncompleteDwtExecutor};
use std::arch::x86_64::*;

pub(crate) struct AvxWavelet10TapsF32 {
    border_mode: BorderMode,
    low_pass: [f32; 16],
    high_pass: [f32; 16],
}

impl AvxWavelet10TapsF32 {
    pub(crate) fn new(border_mode: BorderMode, w: &[f32; 10]) -> Self {
        let g = low_pass_to_high_from_arr(w);
        Self {
            border_mode,
            low_pass: [
                w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], 0., 0., 0., 0., 0., 0.,
            ],
            high_pass: [
                g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9], 0., 0., 0., 0., 0., 0.,
            ],
        }
    }
}

impl DwtForwardExecutor<f32> for AvxWavelet10TapsF32 {
    fn execute_forward(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
    ) -> Result<(), OscletError> {
        let mut scratch = try_vec![f32::default(); self.required_scratch_size(input.len())];
        unsafe { self.execute_forward_impl(input, approx, details, &mut scratch) }
    }

    fn execute_forward_with_scratch(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_forward_impl(input, approx, details, scratch) }
    }

    fn required_scratch_size(&self, _: usize) -> usize {
        0
    }

    fn dwt_size(&self, input_length: usize) -> DwtSize {
        DwtSize::new(dwt_length(input_length, self.filter_length()))
    }
}

impl AvxWavelet10TapsF32 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_forward_impl(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<(), OscletError> {
        let half = dwt_length(input.len(), 10);

        if input.len() < 10 {
            return Err(OscletError::MinFilterSize(input.len(), 10));
        }

        if approx.len() != half {
            return Err(OscletError::ApproxDetailsSize(approx.len()));
        }
        if details.len() != half {
            return Err(OscletError::ApproxDetailsSize(details.len()));
        }

        let required_size = self.required_scratch_size(input.len());
        if scratch.len() < required_size {
            return Err(OscletError::ScratchSize(required_size, scratch.len()));
        }

        unsafe {
            let h0 = _mm256_loadu_ps(self.low_pass.as_ptr());
            let g0 = _mm256_loadu_ps(self.high_pass.as_ptr());

            let h8 = _mm256_loadu_ps(self.low_pass.get_unchecked(8..).as_ptr());
            let g8 = _mm256_loadu_ps(self.high_pass.get_unchecked(8..).as_ptr());

            let interpolation = BorderInterpolation::new(self.border_mode, 0, input.len() as isize);

            let (front_approx, approx) = approx.split_at_mut(4);
            let (front_detail, details) = details.split_at_mut(4);

            for (i, (approx, detail)) in front_approx
                .iter_mut()
                .zip(front_detail.iter_mut())
                .enumerate()
            {
                let base = 2 * i as isize - 8;

                let x0 = interpolation.interpolate(input, base);
                let x1 = interpolation.interpolate(input, base + 1);
                let x2 = interpolation.interpolate(input, base + 2);
                let x3 = interpolation.interpolate(input, base + 3);
                let x4 = interpolation.interpolate(input, base + 4);
                let x5 = interpolation.interpolate(input, base + 5);
                let x6 = interpolation.interpolate(input, base + 6);
                let x7 = interpolation.interpolate(input, base + 7);
                let x8 = *input.get_unchecked((base + 8) as usize);
                let x9 = *input.get_unchecked((base + 9) as usize);

                let vals0 = _mm256_setr_ps(x0, x1, x2, x3, x4, x5, x6, x7);
                let vals1 = _mm256_setr_ps(x8, x9, 0., 0., 0., 0., 0., 0.);
                let a = _mm256_fmadd_ps(vals1, h8, _mm256_mul_ps(vals0, h0));
                let d = _mm256_fmadd_ps(vals1, g8, _mm256_mul_ps(vals0, g0));

                let a = _mm256_hsum_ps(a);
                let d = _mm256_hsum_ps(d);

                _mm_store_ss(approx, a);
                _mm_store_ss(detail, d);
            }

            let (approx, approx_rem) =
                approx.split_at_mut(ten_taps_size_for_input(input.len(), approx.len()));
            let (details, details_rem) =
                details.split_at_mut(ten_taps_size_for_input(input.len(), details.len()));

            let base_start = approx.len();

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * i;

                let input = input.get_unchecked(base..);

                let x0 = _mm256_loadu_ps(input.as_ptr());
                let x8 = _mm256_castps128_ps256(_mm_castsi128_ps(_mm_loadu_si64(
                    input.get_unchecked(8..).as_ptr().cast(),
                )));

                let a = _mm256_fmadd_ps(x8, h8, _mm256_mul_ps(x0, h0));
                let d = _mm256_fmadd_ps(x8, g8, _mm256_mul_ps(x0, g0));

                let wa = _mm256_hsum_ps(a);
                let wd = _mm256_hsum_ps(d);

                _mm_store_ss(approx as *mut f32, wa);
                _mm_store_ss(detail as *mut f32, wd);
            }

            for (i, (approx, detail)) in approx_rem
                .iter_mut()
                .zip(details_rem.iter_mut())
                .enumerate()
            {
                let base = 2 * (i + base_start);

                let x0 = *input.get_unchecked(base);
                let x1 = interpolation.interpolate(input, base as isize + 1);
                let x2 = interpolation.interpolate(input, base as isize + 2);
                let x3 = interpolation.interpolate(input, base as isize + 3);
                let x4 = interpolation.interpolate(input, base as isize + 4);
                let x5 = interpolation.interpolate(input, base as isize + 5);
                let x6 = interpolation.interpolate(input, base as isize + 6);
                let x7 = interpolation.interpolate(input, base as isize + 7);
                let x8 = interpolation.interpolate(input, base as isize + 8);
                let x9 = interpolation.interpolate(input, base as isize + 9);

                let vals0 = _mm256_setr_ps(x0, x1, x2, x3, x4, x5, x6, x7);
                let vals1 = _mm256_setr_ps(x8, x9, 0., 0., 0., 0., 0., 0.);
                let a = _mm256_fmadd_ps(vals1, h8, _mm256_mul_ps(vals0, h0));
                let d = _mm256_fmadd_ps(vals1, g8, _mm256_mul_ps(vals0, g0));

                let a = _mm256_hsum_ps(a);
                let d = _mm256_hsum_ps(d);

                _mm_store_ss(approx, a);
                _mm_store_ss(detail, d);
            }
        }
        Ok(())
    }
}

impl DwtInverseExecutor<f32> for AvxWavelet10TapsF32 {
    fn execute_inverse(
        &self,
        approx: &[f32],
        details: &[f32],
        output: &mut [f32],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_inverse_impl(approx, details, output) }
    }

    fn idwt_size(&self, input_length: DwtSize) -> usize {
        idwt_length(input_length.approx_length, self.filter_length())
    }
}

impl AvxWavelet10TapsF32 {
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

        let rec_len = idwt_length(approx.len(), 10);

        if output.len() != rec_len {
            return Err(OscletError::OutputSizeIsNotValid(output.len(), rec_len));
        }

        const FILTER_OFFSET: usize = 8;
        const FILTER_LENGTH: usize = 10;

        unsafe {
            let safe_start = FILTER_OFFSET;
            // 2*x - off + len >= output.len()
            // x >= (output.len() + off - len)/2
            let mut safe_end = ((output.len() + FILTER_OFFSET).saturating_sub(FILTER_LENGTH)) / 2;

            if safe_start < safe_end {
                for i in 0..safe_start {
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    for j in 0..10 {
                        let k = k + j as isize;
                        if k >= 0 && k < rec_len as isize {
                            *output.get_unchecked_mut(k as usize) = fmla(
                                self.low_pass[j],
                                h,
                                f32::mul_add(
                                    self.high_pass[j],
                                    g,
                                    *output.get_unchecked(k as usize),
                                ),
                            );
                        }
                    }
                }

                let h0 = _mm256_loadu_ps(self.low_pass.as_ptr());
                let g0 = _mm256_loadu_ps(self.high_pass.as_ptr());

                let h8 = _mm_loadu_ps(self.low_pass.get_unchecked(8..).as_ptr());
                let g8 = _mm_loadu_ps(self.high_pass.get_unchecked(8..).as_ptr());

                let mut ui = safe_start;

                while ui + 4 < safe_end {
                    let (h, g) = (
                        _mm_loadu_ps(approx.get_unchecked(ui)),
                        _mm_loadu_ps(details.get_unchecked(ui)),
                    );

                    let h = _mm256_setr_m128(h, h);
                    let g = _mm256_setr_m128(g, g);

                    let k = 2 * ui as isize - FILTER_OFFSET as isize;
                    let part = output.get_unchecked_mut(k as usize..);

                    macro_rules! step_by {
                        ($part: expr, $h: expr, $g: expr, $w: expr) => {
                            let x0 = _mm256_loadu_ps($part.get_unchecked(2 * $w..).as_ptr());
                            let x8 = _mm_castsi128_ps(_mm_loadu_si64(
                                $part.get_unchecked(8 + 2 * $w..).as_ptr().cast(),
                            ));

                            let wh = _mm256_shuffle_ps::<{ shuffle($w, $w, $w, $w) }>($h, $h);
                            let wg = _mm256_shuffle_ps::<{ shuffle($w, $w, $w, $w) }>($g, $g);

                            let q0 = _mm256_fmadd_ps(g0, wg, _mm256_fmadd_ps(h0, wh, x0));
                            let q8 = _mm_fmadd_ps(
                                g8,
                                _mm256_castps256_ps128(wg),
                                _mm_fmadd_ps(h8, _mm256_castps256_ps128(wh), x8),
                            );

                            _mm256_storeu_ps($part.get_unchecked_mut(2 * $w..).as_mut_ptr(), q0);
                            _mm_storel_pd(
                                $part.get_unchecked_mut(8 + 2 * $w..).as_mut_ptr().cast(),
                                _mm_castps_pd(q8),
                            );
                        };
                    }

                    step_by!(part, h, g, 0);
                    step_by!(part, h, g, 1);
                    step_by!(part, h, g, 2);
                    step_by!(part, h, g, 3);

                    ui += 4;
                }

                for i in ui..safe_end {
                    let (h, g) = (
                        _mm256_set1_ps(*approx.get_unchecked(i)),
                        _mm256_set1_ps(*details.get_unchecked(i)),
                    );
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    let part = output.get_unchecked_mut(k as usize..);

                    let x0 = _mm256_loadu_ps(part.as_ptr());
                    let x8 =
                        _mm_castsi128_ps(_mm_loadu_si64(part.get_unchecked(8..).as_ptr().cast()));

                    let q0 = _mm256_fmadd_ps(g0, g, _mm256_fmadd_ps(h0, h, x0));
                    let q8 = _mm_fmadd_ps(
                        g8,
                        _mm256_castps256_ps128(g),
                        _mm_fmadd_ps(h8, _mm256_castps256_ps128(h), x8),
                    );

                    _mm256_storeu_ps(part.as_mut_ptr(), q0);
                    _mm_storel_pd(
                        part.get_unchecked_mut(8..).as_mut_ptr().cast(),
                        _mm_castps_pd(q8),
                    );
                }
            } else {
                safe_end = 0usize;
            }

            for i in safe_end..approx.len() {
                let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..10 {
                    let k = k + j as isize;
                    if k >= 0 && k < rec_len as isize {
                        *output.get_unchecked_mut(k as usize) = fmla(
                            self.low_pass[j],
                            h,
                            f32::mul_add(self.high_pass[j], g, *output.get_unchecked(k as usize)),
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

impl IncompleteDwtExecutor<f32> for AvxWavelet10TapsF32 {
    fn filter_length(&self) -> usize {
        10
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DaubechiesFamily, WaveletFilterProvider};

    #[test]
    fn test_db5_odd() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db4 = AvxWavelet10TapsF32::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db5
                .get_wavelet()
                .as_ref()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 10);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f32; 13] = [
            7.76308732, 4.31346037, 1.56478564, 2.0152442, 3.56281138, 4.58718134, 0.3541704,
            2.85175073, 5.65669193, 8.10768146, 1.19172576, 2.51812658, 1.90731395,
        ];
        const REFERENCE_DETAILS: [f32; 13] = [
            -1.63683102,
            0.4689461,
            -1.17579949,
            0.13929415,
            -0.43849194,
            -3.73867239,
            0.17971352,
            1.34570881,
            0.03313086,
            0.65749801,
            0.28165624,
            0.29616734,
            0.48934456,
        ];

        approx.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_APPROX[i] - x).abs() < 1e-5,
                "approx difference expected to be < 1e-5, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-5,
                "details difference expected to be < 1e-5, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 10)];
        db4.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-5,
                "reconstructed difference expected to be < 1e-5, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_db5_even() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db5 = AvxWavelet10TapsF32::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db5
                .get_wavelet()
                .as_ref()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 10);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db5.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f32; 12] = [
            5.67889878, 7.9758377, 1.616079, 1.16256715, 3.56281138, 4.58718134, 0.3541704,
            2.85175073, 5.67889878, 7.9758377, 1.616079, 1.16256715,
        ];
        const REFERENCE_DETAILS: [f32; 12] = [
            -1.03271544,
            0.16927413,
            -1.06111809,
            0.12152867,
            -0.43849194,
            -3.73867239,
            0.17971352,
            1.34570881,
            -1.03271544,
            0.16927413,
            -1.06111809,
            0.12152867,
        ];

        approx.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_APPROX[i] - x).abs() < 1e-5,
                "approx difference expected to be < 1e-5, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-5,
                "details difference expected to be < 1e-5, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 10)];
        db5.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-5,
                "reconstructed difference expected to be < 1e-5, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_db8_even_big() {
        let data_length = 86;
        let mut input = vec![0.; data_length];
        for i in 0..data_length {
            input[i] = i as f32 / data_length as f32;
        }
        let db4 = AvxWavelet10TapsF32::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db5
                .get_wavelet()
                .as_ref()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 10);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 10)];
        db4.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-5,
                "reconstructed difference expected to be < 1e-5, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }
}
