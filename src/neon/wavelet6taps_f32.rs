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
use std::arch::aarch64::*;

pub(crate) struct NeonWavelet6TapsF32 {
    border_mode: BorderMode,
    low_pass: [f32; 8],
    high_pass: [f32; 8],
}

impl NeonWavelet6TapsF32 {
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[f32; 6]) -> Self {
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

impl DwtForwardExecutor<f32> for NeonWavelet6TapsF32 {
    fn execute_forward(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
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
            let h = vld1q_f32(self.low_pass.as_ptr());
            let h2 = vld1q_f32(self.low_pass.get_unchecked(4..).as_ptr());
            let g = vld1q_f32(self.high_pass.as_ptr());
            let g2 = vld1q_f32(self.high_pass.get_unchecked(4..).as_ptr());

            let mut processed = 0usize;

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(4)
                .zip(details.chunks_exact_mut(4))
                .enumerate()
            {
                let base0 = 2 * 4 * i;

                let input0 = padded_input.get_unchecked(base0..);

                let xw01 = vld1q_f32(input0.as_ptr());
                let xw23 = vld1q_f32(input0.get_unchecked(4..).as_ptr());
                let xw45 = vld1q_f32(input0.get_unchecked(8..).as_ptr());

                let xw0002 = vcombine_f32(vget_low_f32(xw23), vdup_n_f32(0.));
                let xw0003 = vcombine_f32(vget_high_f32(xw23), vdup_n_f32(0.));
                let xw0004 = vcombine_f32(vget_low_f32(xw45), vdup_n_f32(0.));
                let xw0005 = vcombine_f32(vget_high_f32(xw45), vdup_n_f32(0.));

                let xw1 = vcombine_f32(vget_high_f32(xw01), vget_low_f32(xw23));
                let xw2 = vcombine_f32(vget_high_f32(xw23), vget_low_f32(xw45));

                let a0 = vfmaq_f32(vmulq_f32(xw01, h), xw0002, h2);
                let d0 = vfmaq_f32(vmulq_f32(xw01, g), xw0002, g2);

                let a1 = vfmaq_f32(vmulq_f32(xw1, h), xw0003, h2);
                let d1 = vfmaq_f32(vmulq_f32(xw1, g), xw0003, g2);

                let a2 = vfmaq_f32(vmulq_f32(xw23, h), xw0004, h2);
                let d2 = vfmaq_f32(vmulq_f32(xw23, g), xw0004, g2);

                let a3 = vfmaq_f32(vmulq_f32(xw2, h), xw0005, h2);
                let d3 = vfmaq_f32(vmulq_f32(xw2, g), xw0005, g2);

                let wa = vpaddq_f32(vpaddq_f32(a0, a1), vpaddq_f32(a2, a3));
                let wd = vpaddq_f32(vpaddq_f32(d0, d1), vpaddq_f32(d2, d3));

                vst1q_f32(approx.as_mut_ptr(), wa);
                vst1q_f32(detail.as_mut_ptr(), wd);

                processed += 4;
            }

            let approx = approx.chunks_exact_mut(4).into_remainder();
            let details = details.chunks_exact_mut(4).into_remainder();
            let padded_input = padded_input.get_unchecked(processed * 2..);

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * i;
                let input = padded_input.get_unchecked(base..);

                let xw = vld1q_f32(input.as_ptr());
                let xw1 = vcombine_f32(vld1_f32(input.get_unchecked(4..).as_ptr()), vdup_n_f32(0.));

                let a = vfmaq_f32(vmulq_f32(xw, h), xw1, h2);
                let d = vfmaq_f32(vmulq_f32(xw, g), xw1, g2);

                let a0 = vpadds_f32(vadd_f32(vget_low_f32(a), vget_high_f32(a)));
                let d0 = vpadds_f32(vadd_f32(vget_low_f32(d), vget_high_f32(d)));

                *approx = a0;
                *detail = d0;
            }
        }
        Ok(())
    }
}

impl DwtInverseExecutor<f32> for NeonWavelet6TapsF32 {
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
                let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..6 {
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

            let h0 = vld1q_f32(self.low_pass.as_ptr());
            let h2 = vld1q_f32(self.low_pass.get_unchecked(4..).as_ptr());
            let g0 = vld1q_f32(self.high_pass.as_ptr());
            let g2 = vld1q_f32(self.high_pass.get_unchecked(4..).as_ptr());

            let mut ui = safe_start;

            while ui + 2 < safe_end {
                let (h, g) = (
                    vld1_f32(approx.get_unchecked(ui)),
                    vld1_f32(details.get_unchecked(ui)),
                );
                let k = 2 * ui as isize - FILTER_OFFSET as isize;
                let part0 = output.get_unchecked_mut(k as usize..);
                let q0 = vld1q_f32(part0.as_ptr());
                let q1 = vld1q_f32(part0.get_unchecked(4..).as_ptr());

                let w0 = vfmaq_lane_f32::<0>(vfmaq_lane_f32::<0>(q0, h0, h), g0, g);
                let w1 = vfmaq_lane_f32::<0>(vfmaq_lane_f32::<0>(q1, h2, h), g2, g);

                let interim_w = vcombine_f32(vget_high_f32(w0), vget_low_f32(w1));

                let w3 = vfmaq_lane_f32::<1>(vfmaq_lane_f32::<1>(interim_w, h0, h), g0, g);
                let w4 = vfmaq_lane_f32::<1>(
                    vfmaq_lane_f32::<1>(vcombine_f32(vget_high_f32(q1), vdup_n_f32(0.)), h2, h),
                    g2,
                    g,
                );

                vst1_f32(part0.as_mut_ptr(), vget_low_f32(w0));
                vst1q_f32(part0.get_unchecked_mut(2..).as_mut_ptr(), w3);
                vst1_f32(part0.get_unchecked_mut(6..).as_mut_ptr(), vget_low_f32(w4));
                ui += 2;
            }

            for i in ui..safe_end {
                let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                let part = output.get_unchecked_mut(k as usize..);

                let xw0 = vld1q_f32(part.as_ptr());
                let xw1 = vcombine_f32(vld1_f32(part.get_unchecked(4..).as_ptr()), vdup_n_f32(0.));

                let q0 = vfmaq_n_f32(vfmaq_n_f32(xw0, h0, h), g0, g);
                let q2 = vfmaq_n_f32(vfmaq_n_f32(xw1, h2, h), g2, g);

                vst1q_f32(part.as_mut_ptr(), q0);
                vst1_f32(part.get_unchecked_mut(4..).as_mut_ptr(), vget_low_f32(q2));
            }

            for i in safe_end..approx.len() {
                let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..6 {
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

impl IncompleteDwtExecutor<f32> for NeonWavelet6TapsF32 {
    fn filter_length(&self) -> usize {
        6
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coiflet::CoifletFamily;
    use crate::{DaubechiesFamily, WaveletFilterProvider};

    #[test]
    fn test_db3_odd() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db3 = NeonWavelet6TapsF32::new(
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

        const REFERENCE_APPROX: [f32; 11] = [
            0.8483726, 2.5241373, 2.65038574, 5.04554797, 1.36113344, 1.0534151, 5.85968077,
            8.27594493, 2.09006931, 2.1647733, 1.88197732,
        ];
        const REFERENCE_DETAILS: [f32; 11] = [
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 6)];
        db3.execute_inverse(&approx, &details, &mut reconstructed)
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
    fn test_db3_even() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db3 = NeonWavelet6TapsF32::new(
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

        const REFERENCE_APPROX: [f32; 10] = [
            2.25345752, 1.28973105, 2.65038574, 5.04554797, 1.36113344, 1.0534151, 5.85968077,
            8.27594493, 2.25345752, 1.28973105,
        ];
        const REFERENCE_DETAILS: [f32; 10] = [
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 6)];
        db3.execute_inverse(&approx, &details, &mut reconstructed)
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
    fn test_coif1_even() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db3 = NeonWavelet6TapsF32::new(
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

        const REFERENCE_APPROX: [f32; 10] = [
            0.64709521, 1.74438159, 4.53911719, 3.20774595, 0.30097675, 4.61093707, 6.14348133,
            6.59556141, 0.64709521, 1.74438159,
        ];
        const REFERENCE_DETAILS: [f32; 10] = [
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 6)];
        db3.execute_inverse(&approx, &details, &mut reconstructed)
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
