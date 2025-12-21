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
use crate::border_mode::{BorderInterpolation, BorderMode};
use crate::err::{OscletError, try_vec};
use crate::mla::fmla;
use crate::util::{dwt_length, idwt_length, low_pass_to_high_from_arr, ten_taps_size_for_input};
use crate::{DwtForwardExecutor, DwtInverseExecutor, DwtSize, IncompleteDwtExecutor};
use std::arch::aarch64::*;

pub(crate) struct NeonWavelet10TapsF64 {
    border_mode: BorderMode,
    low_pass: [f64; 10],
    high_pass: [f64; 10],
}

impl NeonWavelet10TapsF64 {
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[f64; 10]) -> Self {
        Self {
            border_mode,
            low_pass: *wavelet,
            high_pass: low_pass_to_high_from_arr(wavelet),
        }
    }
}

impl DwtForwardExecutor<f64> for NeonWavelet10TapsF64 {
    fn execute_forward(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
    ) -> Result<(), OscletError> {
        let mut scratch = try_vec![f64::default(); self.required_scratch_size(input.len())];
        self.execute_forward_with_scratch(input, approx, details, &mut scratch)
    }

    fn execute_forward_with_scratch(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
        scratch: &mut [f64],
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
            let h0 = vld1q_f64(self.low_pass.as_ptr());
            let g0 = vld1q_f64(self.high_pass.as_ptr());

            let h1 = vld1q_f64(self.low_pass.get_unchecked(2..).as_ptr());
            let g1 = vld1q_f64(self.high_pass.get_unchecked(2..).as_ptr());

            let h2 = vld1q_f64(self.low_pass.get_unchecked(4..).as_ptr());
            let g2 = vld1q_f64(self.high_pass.get_unchecked(4..).as_ptr());

            let h3 = vld1q_f64(self.low_pass.get_unchecked(6..).as_ptr());
            let g3 = vld1q_f64(self.high_pass.get_unchecked(6..).as_ptr());

            let h4 = vld1q_f64(self.low_pass.get_unchecked(8..).as_ptr());
            let g4 = vld1q_f64(self.high_pass.get_unchecked(8..).as_ptr());

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

                let x01 = vld1q_f64([x0, x1].as_ptr());
                let x23 = vld1q_f64([x2, x3].as_ptr());
                let x45 = vld1q_f64([x4, x5].as_ptr());
                let x67 = vld1q_f64([x6, x7].as_ptr());
                let x89 = vld1q_f64([x8, x9].as_ptr());

                let mut wa = vfmaq_f64(
                    vfmaq_f64(
                        vfmaq_f64(vfmaq_f64(vmulq_f64(x01, h0), x23, h1), x45, h2),
                        x67,
                        h3,
                    ),
                    x89,
                    h4,
                );
                let mut wd = vfmaq_f64(
                    vfmaq_f64(
                        vfmaq_f64(vfmaq_f64(vmulq_f64(x01, g0), x23, g1), x45, g2),
                        x67,
                        g3,
                    ),
                    x89,
                    g4,
                );

                wa = vpaddq_f64(wa, wa);
                wd = vpaddq_f64(wd, wd);

                vst1q_lane_f64::<0>(approx, wa);
                vst1q_lane_f64::<0>(detail, wd);
            }

            let (approx, approx_rem) =
                approx.split_at_mut(ten_taps_size_for_input(input.len(), approx.len()));
            let (details, details_rem) =
                details.split_at_mut(ten_taps_size_for_input(input.len(), details.len()));

            let base_start = approx.len();

            let mut processed = 0usize;

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(2)
                .zip(details.chunks_exact_mut(2))
                .enumerate()
            {
                let base0 = 2 * 2 * i;

                let input0 = input.get_unchecked(base0..);

                let xw00 = vld1q_f64(input0.as_ptr());
                let xw01 = vld1q_f64(input0.get_unchecked(2..).as_ptr());
                let xw02 = vld1q_f64(input0.get_unchecked(4..).as_ptr());
                let xw03 = vld1q_f64(input0.get_unchecked(6..).as_ptr());
                let xw04 = vld1q_f64(input0.get_unchecked(8..).as_ptr());
                let xw05 = vld1q_f64(input0.get_unchecked(10..).as_ptr());

                let a0 = vfmaq_f64(
                    vfmaq_f64(
                        vfmaq_f64(vfmaq_f64(vmulq_f64(xw00, h0), xw01, h1), xw02, h2),
                        xw03,
                        h3,
                    ),
                    xw04,
                    h4,
                );
                let d0 = vfmaq_f64(
                    vfmaq_f64(
                        vfmaq_f64(vfmaq_f64(vmulq_f64(xw00, g0), xw01, g1), xw02, g2),
                        xw03,
                        g3,
                    ),
                    xw04,
                    g4,
                );

                let a1 = vfmaq_f64(
                    vfmaq_f64(
                        vfmaq_f64(vfmaq_f64(vmulq_f64(xw01, h0), xw02, h1), xw03, h2),
                        xw04,
                        h3,
                    ),
                    xw05,
                    h4,
                );
                let d1 = vfmaq_f64(
                    vfmaq_f64(
                        vfmaq_f64(vfmaq_f64(vmulq_f64(xw01, g0), xw02, g1), xw03, g2),
                        xw04,
                        g3,
                    ),
                    xw05,
                    g4,
                );

                let q0 = vpaddq_f64(a0, a1);
                let fq0 = vpaddq_f64(d0, d1);

                vst1q_f64(approx.as_mut_ptr(), q0);
                vst1q_f64(detail.as_mut_ptr(), fq0);

                processed += 2;
            }

            let approx = approx.chunks_exact_mut(2).into_remainder();
            let details = details.chunks_exact_mut(2).into_remainder();

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * (i + processed);

                let input = input.get_unchecked(base..);

                let x01 = vld1q_f64(input.as_ptr());
                let x23 = vld1q_f64(input.get_unchecked(2..).as_ptr());
                let x45 = vld1q_f64(input.get_unchecked(4..).as_ptr());
                let x67 = vld1q_f64(input.get_unchecked(6..).as_ptr());
                let x89 = vld1q_f64(input.get_unchecked(8..).as_ptr());

                let mut wa = vfmaq_f64(
                    vfmaq_f64(
                        vfmaq_f64(vfmaq_f64(vmulq_f64(x01, h0), x23, h1), x45, h2),
                        x67,
                        h3,
                    ),
                    x89,
                    h4,
                );
                let mut wd = vfmaq_f64(
                    vfmaq_f64(
                        vfmaq_f64(vfmaq_f64(vmulq_f64(x01, g0), x23, g1), x45, g2),
                        x67,
                        g3,
                    ),
                    x89,
                    g4,
                );

                wa = vpaddq_f64(wa, wa);
                wd = vpaddq_f64(wd, wd);

                vst1q_lane_f64::<0>(approx, wa);
                vst1q_lane_f64::<0>(detail, wd);
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

                let x01 = vld1q_f64([x0, x1].as_ptr());
                let x23 = vld1q_f64([x2, x3].as_ptr());
                let x45 = vld1q_f64([x4, x5].as_ptr());
                let x67 = vld1q_f64([x6, x7].as_ptr());
                let x89 = vld1q_f64([x8, x9].as_ptr());

                let mut wa = vfmaq_f64(
                    vfmaq_f64(
                        vfmaq_f64(vfmaq_f64(vmulq_f64(x01, h0), x23, h1), x45, h2),
                        x67,
                        h3,
                    ),
                    x89,
                    h4,
                );
                let mut wd = vfmaq_f64(
                    vfmaq_f64(
                        vfmaq_f64(vfmaq_f64(vmulq_f64(x01, g0), x23, g1), x45, g2),
                        x67,
                        g3,
                    ),
                    x89,
                    g4,
                );

                wa = vpaddq_f64(wa, wa);
                wd = vpaddq_f64(wd, wd);

                vst1q_lane_f64::<0>(approx, wa);
                vst1q_lane_f64::<0>(detail, wd);
            }
        }
        Ok(())
    }

    fn required_scratch_size(&self, _: usize) -> usize {
        0
    }

    fn dwt_size(&self, input_length: usize) -> DwtSize {
        DwtSize::new(dwt_length(input_length, 10))
    }
}

impl DwtInverseExecutor<f64> for NeonWavelet10TapsF64 {
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
                                fmla(self.high_pass[j], g, *output.get_unchecked(k as usize)),
                            );
                        }
                    }
                }

                let h0 = vld1q_f64(self.low_pass.as_ptr());
                let g0 = vld1q_f64(self.high_pass.as_ptr());

                let h1 = vld1q_f64(self.low_pass.get_unchecked(2..).as_ptr());
                let g1 = vld1q_f64(self.high_pass.get_unchecked(2..).as_ptr());

                let h2 = vld1q_f64(self.low_pass.get_unchecked(4..).as_ptr());
                let g2 = vld1q_f64(self.high_pass.get_unchecked(4..).as_ptr());

                let h3 = vld1q_f64(self.low_pass.get_unchecked(6..).as_ptr());
                let g3 = vld1q_f64(self.high_pass.get_unchecked(6..).as_ptr());

                let h4 = vld1q_f64(self.low_pass.get_unchecked(8..).as_ptr());
                let g4 = vld1q_f64(self.high_pass.get_unchecked(8..).as_ptr());

                let mut ui = safe_start;

                while ui + 2 < safe_end {
                    let (h, g) = (
                        vld1q_f64(approx.get_unchecked(ui)),
                        vld1q_f64(details.get_unchecked(ui)),
                    );
                    let k = 2 * ui as isize - FILTER_OFFSET as isize;
                    let part = output.get_unchecked_mut(k as usize..);

                    let w0 = vld1q_f64(part.as_ptr());
                    let w1 = vld1q_f64(part.get_unchecked(2..).as_ptr());
                    let w2 = vld1q_f64(part.get_unchecked(4..).as_ptr());
                    let w3 = vld1q_f64(part.get_unchecked(6..).as_ptr());
                    let w4 = vld1q_f64(part.get_unchecked(8..).as_ptr());
                    let w5 = vld1q_f64(part.get_unchecked(10..).as_ptr());

                    let q0 = vfmaq_laneq_f64::<0>(vfmaq_laneq_f64::<0>(w0, h0, h), g0, g);
                    let q2 = vfmaq_laneq_f64::<0>(vfmaq_laneq_f64::<0>(w1, h1, h), g1, g);
                    let q4 = vfmaq_laneq_f64::<0>(vfmaq_laneq_f64::<0>(w2, h2, h), g2, g);
                    let q6 = vfmaq_laneq_f64::<0>(vfmaq_laneq_f64::<0>(w3, h3, h), g3, g);
                    let q8 = vfmaq_laneq_f64::<0>(vfmaq_laneq_f64::<0>(w4, h4, h), g4, g);

                    let q2 = vfmaq_laneq_f64::<1>(vfmaq_laneq_f64::<1>(q2, h0, h), g0, g);
                    let q4 = vfmaq_laneq_f64::<1>(vfmaq_laneq_f64::<1>(q4, h1, h), g1, g);
                    let q6 = vfmaq_laneq_f64::<1>(vfmaq_laneq_f64::<1>(q6, h2, h), g2, g);
                    let q8 = vfmaq_laneq_f64::<1>(vfmaq_laneq_f64::<1>(q8, h3, h), g3, g);
                    let q10 = vfmaq_laneq_f64::<1>(vfmaq_laneq_f64::<1>(w5, h4, h), g4, g);

                    vst1q_f64(part.as_mut_ptr(), q0);
                    vst1q_f64(part.get_unchecked_mut(2..).as_mut_ptr(), q2);
                    vst1q_f64(part.get_unchecked_mut(4..).as_mut_ptr(), q4);
                    vst1q_f64(part.get_unchecked_mut(6..).as_mut_ptr(), q6);
                    vst1q_f64(part.get_unchecked_mut(8..).as_mut_ptr(), q8);
                    vst1q_f64(part.get_unchecked_mut(10..).as_mut_ptr(), q10);

                    ui += 2;
                }

                for i in ui..safe_end {
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    let part = output.get_unchecked_mut(k as usize..);

                    let w0 = vld1q_f64(part.as_ptr());
                    let w1 = vld1q_f64(part.get_unchecked(2..).as_ptr());
                    let w2 = vld1q_f64(part.get_unchecked(4..).as_ptr());
                    let w3 = vld1q_f64(part.get_unchecked(6..).as_ptr());
                    let w4 = vld1q_f64(part.get_unchecked(8..).as_ptr());

                    let q0 = vfmaq_n_f64(vfmaq_n_f64(w0, h0, h), g0, g);
                    let q2 = vfmaq_n_f64(vfmaq_n_f64(w1, h1, h), g1, g);
                    let q4 = vfmaq_n_f64(vfmaq_n_f64(w2, h2, h), g2, g);
                    let q6 = vfmaq_n_f64(vfmaq_n_f64(w3, h3, h), g3, g);
                    let q8 = vfmaq_n_f64(vfmaq_n_f64(w4, h4, h), g4, g);

                    vst1q_f64(part.as_mut_ptr(), q0);
                    vst1q_f64(part.get_unchecked_mut(2..).as_mut_ptr(), q2);
                    vst1q_f64(part.get_unchecked_mut(4..).as_mut_ptr(), q4);
                    vst1q_f64(part.get_unchecked_mut(6..).as_mut_ptr(), q6);
                    vst1q_f64(part.get_unchecked_mut(8..).as_mut_ptr(), q8);
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
                            fmla(self.high_pass[j], g, *output.get_unchecked(k as usize)),
                        );
                    }
                }
            }
        }
        Ok(())
    }

    fn idwt_size(&self, input_length: DwtSize) -> usize {
        idwt_length(input_length.approx_length, 10)
    }
}

impl IncompleteDwtExecutor<f64> for NeonWavelet10TapsF64 {
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
        let db4 = NeonWavelet10TapsF64::new(
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

        const REFERENCE_APPROX: [f64; 13] = [
            7.76308732, 4.31346037, 1.56478564, 2.0152442, 3.56281138, 4.58718134, 0.3541704,
            2.85175073, 5.65669193, 8.10768146, 1.19172576, 2.51812658, 1.90731395,
        ];
        const REFERENCE_DETAILS: [f64; 13] = [
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
        let db5 = NeonWavelet10TapsF64::new(
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

        const REFERENCE_APPROX: [f64; 12] = [
            5.67889878, 7.9758377, 1.616079, 1.16256715, 3.56281138, 4.58718134, 0.3541704,
            2.85175073, 5.67889878, 7.9758377, 1.616079, 1.16256715,
        ];
        const REFERENCE_DETAILS: [f64; 12] = [
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
            input[i] = i as f64 / data_length as f64;
        }
        let db4 = NeonWavelet10TapsF64::new(
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
