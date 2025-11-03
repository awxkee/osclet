/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
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
use crate::border_mode::BorderMode;
use crate::err::OscletError;
use crate::filter_padding::make_arena_1d;
use crate::mla::fmla;
use crate::util::{dwt_length, idwt_length, low_pass_to_high_from_arr};
use crate::{DwtForwardExecutor, DwtInverseExecutor, IncompleteDwtExecutor};
use std::arch::aarch64::{
    vfmaq_f64, vfmaq_laneq_f64, vfmaq_n_f64, vld1q_f64, vmulq_f64, vpaddq_f64, vst1q_f64,
    vst1q_lane_f64,
};

pub(crate) struct NeonWavelet8TapsF64 {
    border_mode: BorderMode,
    low_pass: [f64; 8],
    high_pass: [f64; 8],
}

impl NeonWavelet8TapsF64 {
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[f64; 8]) -> Self {
        let g = low_pass_to_high_from_arr(wavelet);
        Self {
            border_mode,
            low_pass: [
                wavelet[0], wavelet[1], wavelet[2], wavelet[3], wavelet[4], wavelet[5], wavelet[6],
                wavelet[7],
            ],
            high_pass: [g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7]],
        }
    }
}

impl DwtForwardExecutor<f64> for NeonWavelet8TapsF64 {
    fn execute_forward(
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
            let h0 = vld1q_f64(self.low_pass.as_ptr());
            let g0 = vld1q_f64(self.high_pass.as_ptr());

            let h1 = vld1q_f64(self.low_pass.get_unchecked(2..).as_ptr());
            let g1 = vld1q_f64(self.high_pass.get_unchecked(2..).as_ptr());

            let h2 = vld1q_f64(self.low_pass.get_unchecked(4..).as_ptr());
            let g2 = vld1q_f64(self.high_pass.get_unchecked(4..).as_ptr());

            let h3 = vld1q_f64(self.low_pass.get_unchecked(6..).as_ptr());
            let g3 = vld1q_f64(self.high_pass.get_unchecked(6..).as_ptr());

            let mut processed = 0usize;

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(4)
                .zip(details.chunks_exact_mut(4))
                .enumerate()
            {
                let base0 = 2 * 4 * i;

                let input0 = padded_input.get_unchecked(base0..);

                let xw00 = vld1q_f64(input0.as_ptr());
                let xw01 = vld1q_f64(input0.get_unchecked(2..).as_ptr());
                let xw02 = vld1q_f64(input0.get_unchecked(4..).as_ptr());
                let xw03 = vld1q_f64(input0.get_unchecked(6..).as_ptr());
                let xw04 = vld1q_f64(input0.get_unchecked(8..).as_ptr());
                let xw05 = vld1q_f64(input0.get_unchecked(10..).as_ptr());
                let xw06 = vld1q_f64(input0.get_unchecked(12..).as_ptr());

                let a0 = vfmaq_f64(
                    vfmaq_f64(vfmaq_f64(vmulq_f64(xw00, h0), xw01, h1), xw02, h2),
                    xw03,
                    h3,
                );
                let d0 = vfmaq_f64(
                    vfmaq_f64(vfmaq_f64(vmulq_f64(xw00, g0), xw01, g1), xw02, g2),
                    xw03,
                    g3,
                );

                let a1 = vfmaq_f64(
                    vfmaq_f64(vfmaq_f64(vmulq_f64(xw01, h0), xw02, h1), xw03, h2),
                    xw04,
                    h3,
                );
                let d1 = vfmaq_f64(
                    vfmaq_f64(vfmaq_f64(vmulq_f64(xw01, g0), xw02, g1), xw03, g2),
                    xw04,
                    g3,
                );

                let a2 = vfmaq_f64(
                    vfmaq_f64(vfmaq_f64(vmulq_f64(xw02, h0), xw03, h1), xw04, h2),
                    xw05,
                    h3,
                );
                let d2 = vfmaq_f64(
                    vfmaq_f64(vfmaq_f64(vmulq_f64(xw02, g0), xw03, g1), xw04, g2),
                    xw05,
                    g3,
                );

                let a3 = vfmaq_f64(
                    vfmaq_f64(vfmaq_f64(vmulq_f64(xw03, h0), xw04, h1), xw05, h2),
                    xw06,
                    h3,
                );
                let d3 = vfmaq_f64(
                    vfmaq_f64(vfmaq_f64(vmulq_f64(xw03, g0), xw04, g1), xw05, g2),
                    xw06,
                    g3,
                );

                let q0 = vpaddq_f64(a0, a1);
                let q1 = vpaddq_f64(a2, a3);

                let fq0 = vpaddq_f64(d0, d1);
                let fq1 = vpaddq_f64(d2, d3);

                vst1q_f64(approx.as_mut_ptr(), q0);
                vst1q_f64(approx.get_unchecked_mut(2..).as_mut_ptr(), q1);
                vst1q_f64(detail.as_mut_ptr(), fq0);
                vst1q_f64(detail.get_unchecked_mut(2..).as_mut_ptr(), fq1);

                processed += 4;
            }

            let approx = approx.chunks_exact_mut(4).into_remainder();
            let details = details.chunks_exact_mut(4).into_remainder();
            let padded_input = padded_input.get_unchecked(processed * 2..);

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * i;

                let input = padded_input.get_unchecked(base..);

                let x01 = vld1q_f64(input.as_ptr());
                let x23 = vld1q_f64(input.get_unchecked(2..).as_ptr());
                let x45 = vld1q_f64(input.get_unchecked(4..).as_ptr());
                let x67 = vld1q_f64(input.get_unchecked(6..).as_ptr());

                let mut wa = vfmaq_f64(
                    vfmaq_f64(vfmaq_f64(vmulq_f64(x01, h0), x23, h1), x45, h2),
                    x67,
                    h3,
                );
                let mut wd = vfmaq_f64(
                    vfmaq_f64(vfmaq_f64(vmulq_f64(x01, g0), x23, g1), x45, g2),
                    x67,
                    g3,
                );

                wa = vpaddq_f64(wa, wa);
                wd = vpaddq_f64(wd, wd);

                vst1q_lane_f64::<0>(approx, wa);
                vst1q_lane_f64::<0>(detail, wd);
            }
        }
        Ok(())
    }
}

impl DwtInverseExecutor<f64> for NeonWavelet8TapsF64 {
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
            let mut safe_end = ((output.len() + FILTER_OFFSET).saturating_sub(FILTER_LENGTH)) / 2;

            if safe_start < safe_end {
                for i in 0..safe_start {
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    for j in 0..8 {
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

                let mut ui = safe_start;

                while ui + 2 < safe_end {
                    let (h, g) = (
                        vld1q_f64(approx.get_unchecked(ui)),
                        vld1q_f64(details.get_unchecked(ui)),
                    );
                    let k = 2 * ui as isize - FILTER_OFFSET as isize;
                    let part0 = output.get_unchecked_mut(k as usize..);
                    let q0 = vld1q_f64(part0.as_ptr());
                    let q1 = vld1q_f64(part0.get_unchecked(2..).as_ptr());
                    let q2 = vld1q_f64(part0.get_unchecked(4..).as_ptr());
                    let q3 = vld1q_f64(part0.get_unchecked(6..).as_ptr());
                    let q4 = vld1q_f64(part0.get_unchecked(8..).as_ptr());

                    let w0 = vfmaq_laneq_f64::<0>(vfmaq_laneq_f64::<0>(q0, h0, h), g0, g);
                    let w1 = vfmaq_laneq_f64::<0>(vfmaq_laneq_f64::<0>(q1, h1, h), g1, g);
                    let w2 = vfmaq_laneq_f64::<0>(vfmaq_laneq_f64::<0>(q2, h2, h), g2, g);
                    let w3 = vfmaq_laneq_f64::<0>(vfmaq_laneq_f64::<0>(q3, h2, h), g2, g);

                    let w4 = vfmaq_laneq_f64::<1>(vfmaq_laneq_f64::<1>(w1, h0, h), g0, g);
                    let w5 = vfmaq_laneq_f64::<1>(vfmaq_laneq_f64::<1>(w2, h1, h), g1, g);
                    let w6 = vfmaq_laneq_f64::<1>(vfmaq_laneq_f64::<1>(w3, h2, h), g2, g);
                    let w7 = vfmaq_laneq_f64::<1>(vfmaq_laneq_f64::<1>(q4, h2, h), g2, g);

                    vst1q_f64(part0.as_mut_ptr(), w0);
                    vst1q_f64(part0.get_unchecked_mut(2..).as_mut_ptr(), w4);
                    vst1q_f64(part0.get_unchecked_mut(4..).as_mut_ptr(), w5);
                    vst1q_f64(part0.get_unchecked_mut(6..).as_mut_ptr(), w6);
                    vst1q_f64(part0.get_unchecked_mut(8..).as_mut_ptr(), w7);
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

                    let q0 = vfmaq_n_f64(vfmaq_n_f64(w0, h0, h), g0, g);
                    let q2 = vfmaq_n_f64(vfmaq_n_f64(w1, h1, h), g1, g);
                    let q4 = vfmaq_n_f64(vfmaq_n_f64(w2, h2, h), g2, g);
                    let q6 = vfmaq_n_f64(vfmaq_n_f64(w3, h3, h), g3, g);

                    vst1q_f64(part.as_mut_ptr(), q0);
                    vst1q_f64(part.get_unchecked_mut(2..).as_mut_ptr(), q2);
                    vst1q_f64(part.get_unchecked_mut(4..).as_mut_ptr(), q4);
                    vst1q_f64(part.get_unchecked_mut(6..).as_mut_ptr(), q6);
                }
            } else {
                safe_end = 0usize;
            }

            for i in safe_end..approx.len() {
                let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..8 {
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
}

impl IncompleteDwtExecutor<f64> for NeonWavelet8TapsF64 {
    fn filter_length(&self) -> usize {
        8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DaubechiesFamily, WaveletFilterProvider};

    #[test]
    fn test_db4_odd() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db4 = NeonWavelet8TapsF64::new(
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
    fn test_db4_even() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db4 = NeonWavelet8TapsF64::new(
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
}
