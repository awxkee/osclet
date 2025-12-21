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
use crate::err::{OscletError, try_vec};
use crate::mla::fmla;
use crate::util::{dwt_length, idwt_length, low_pass_to_high_from_arr, two_taps_size_for_input};
use crate::{BorderMode, DwtForwardExecutor, DwtInverseExecutor, DwtSize, IncompleteDwtExecutor};
use num_traits::AsPrimitive;
use std::arch::aarch64::*;

pub(crate) struct NeonWavelet2TapsF64 {
    border_mode: BorderMode,
    low_pass: [f64; 2],
    high_pass: [f64; 2],
}

impl NeonWavelet2TapsF64 {
    pub fn new(border_mode: BorderMode, wavelet: &[f64; 2]) -> Self {
        Self {
            border_mode,
            low_pass: *wavelet,
            high_pass: low_pass_to_high_from_arr(wavelet),
        }
    }
}

impl DwtForwardExecutor<f64> for NeonWavelet2TapsF64 {
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

        let required_size = self.required_scratch_size(input.len());
        if scratch.len() < required_size {
            return Err(OscletError::ScratchSize(required_size, scratch.len()));
        }

        unsafe {
            let l0 = vld1q_f64(self.low_pass.as_ptr());
            let h0 = vld1q_f64(self.high_pass.as_ptr());

            let (approx, approx_rem) =
                approx.split_at_mut(two_taps_size_for_input(input.len(), approx.len()));
            let (details, details_rem) =
                details.split_at_mut(two_taps_size_for_input(input.len(), details.len()));

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(2)
                .zip(details.chunks_exact_mut(2))
                .enumerate()
            {
                let base0 = 2 * 2 * i;

                let input0 = input.get_unchecked(base0..);
                let input1 = input.get_unchecked(base0 + 2..);

                let xw0 = vld1q_f64(input0.as_ptr());
                let xw1 = vld1q_f64(input1.as_ptr());

                let a0 = vmulq_f64(xw0, l0);
                let d0 = vmulq_f64(xw0, h0);

                let a1 = vmulq_f64(xw1, l0);
                let d1 = vmulq_f64(xw1, h0);

                vst1q_f64(approx.as_mut_ptr(), vpaddq_f64(a0, a1));
                vst1q_f64(detail.as_mut_ptr(), vpaddq_f64(d0, d1));
            }

            let processed = 2 * approx.chunks_exact_mut(2).len() * 2;

            let approx = approx.chunks_exact_mut(2).into_remainder();
            let details = details.chunks_exact_mut(2).into_remainder();

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = processed + 2 * i;

                let input = input.get_unchecked(base..);

                let xw = vld1q_f64(input.as_ptr());

                let a = vpaddd_f64(vmulq_f64(xw, l0));
                let d = vpaddd_f64(vmulq_f64(xw, h0));

                *approx = a;
                *detail = d;
            }

            if !details_rem.is_empty() && !approx_rem.is_empty() {
                let i = half - 1;
                let x0 = input.get_unchecked(2 * i);
                let x1 = self.border_mode.interpolate(
                    input,
                    2 * i as isize + 1,
                    0,
                    input.len() as isize,
                );

                let mut a = 0.0f64.as_();
                let mut d = 0.0f64.as_();

                a = fmla(self.low_pass[0], *x0, a);
                d = fmla(self.high_pass[0], *x0, d);

                a = fmla(self.low_pass[1], x1, a);
                d = fmla(self.high_pass[1], x1, d);

                *approx_rem.last_mut().unwrap() = a;
                *details_rem.last_mut().unwrap() = d;
            }
        }
        Ok(())
    }

    fn required_scratch_size(&self, _: usize) -> usize {
        0
    }

    fn dwt_size(&self, input_length: usize) -> DwtSize {
        DwtSize::new(dwt_length(input_length, self.filter_length()))
    }
}

impl DwtInverseExecutor<f64> for NeonWavelet2TapsF64 {
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

        let rec_len = idwt_length(approx.len(), 2);

        if output.len() != rec_len {
            return Err(OscletError::OutputSizeIsNotValid(output.len(), rec_len));
        }

        const FILTER_OFFSET: usize = 0;
        const FILTER_LENGTH: usize = 2;

        unsafe {
            let safe_start = FILTER_OFFSET;
            // 2*x - off + len >= output.len()
            // x >= (output.len() + off - len)/2
            let mut safe_end = ((output.len() + FILTER_OFFSET).saturating_sub(FILTER_LENGTH)) / 2;

            if safe_start < safe_end {
                for i in 0..safe_start {
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    for j in 0..2 {
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

                let l0 = vld1q_f64(self.low_pass.as_ptr());
                let h0 = vld1q_f64(self.high_pass.as_ptr());

                let mut uq = safe_start;

                while uq + 2 < safe_end {
                    let (h, g) = (
                        vld1q_f64(approx.get_unchecked(uq..).as_ptr()),
                        vld1q_f64(details.get_unchecked(uq..).as_ptr()),
                    );
                    let k0 = 2 * uq as isize - FILTER_OFFSET as isize;
                    let part0_src = output.get_unchecked(k0 as usize..);
                    let part1_src = output.get_unchecked(k0 as usize + 2..);
                    let xw0 = vld1q_f64(part0_src.as_ptr());
                    let xw1 = vld1q_f64(part1_src.as_ptr());
                    let q0 = vfmaq_laneq_f64::<0>(vfmaq_laneq_f64::<0>(xw0, h0, g), l0, h);
                    let q1 = vfmaq_laneq_f64::<1>(vfmaq_laneq_f64::<1>(xw1, h0, g), l0, h);
                    vst1q_f64(output.get_unchecked_mut(k0 as usize..).as_mut_ptr(), q0);
                    vst1q_f64(output.get_unchecked_mut(k0 as usize + 2..).as_mut_ptr(), q1);
                    uq += 2;
                }

                for i in uq..safe_end {
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    let part = output.get_unchecked_mut(k as usize..);
                    let xw = vld1q_f64(part.as_ptr());
                    let q = vfmaq_n_f64(vfmaq_n_f64(xw, h0, g), l0, h);
                    vst1q_f64(part.as_mut_ptr(), q);
                }
            } else {
                safe_end = 0usize;
            }

            for i in safe_end..approx.len() {
                let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..2 {
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
        idwt_length(input_length.approx_length, self.filter_length())
    }
}

impl IncompleteDwtExecutor<f64> for NeonWavelet2TapsF64 {
    fn filter_length(&self) -> usize {
        2
    }
}

pub(crate) struct NeonWavelet2TapsF32 {
    border_mode: BorderMode,
    low_pass: [f32; 4],
    high_pass: [f32; 4],
}

impl NeonWavelet2TapsF32 {
    pub fn new(border_mode: BorderMode, wavelet: &[f32; 2]) -> Self {
        let k = low_pass_to_high_from_arr(wavelet);
        Self {
            border_mode,
            low_pass: [wavelet[0], wavelet[1], wavelet[0], wavelet[1]],
            high_pass: [k[0], k[1], k[0], k[1]],
        }
    }
}

impl DwtForwardExecutor<f32> for NeonWavelet2TapsF32 {
    fn execute_forward(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
    ) -> Result<(), OscletError> {
        let mut scratch = try_vec![f32::default(); self.required_scratch_size(input.len())];
        self.execute_forward_with_scratch(input, approx, details, &mut scratch)
    }

    fn execute_forward_with_scratch(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
        scratch: &mut [f32],
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

        let required_size = self.required_scratch_size(input.len());
        if scratch.len() < required_size {
            return Err(OscletError::ScratchSize(required_size, scratch.len()));
        }

        let (approx, approx_rem) =
            approx.split_at_mut(two_taps_size_for_input(input.len(), approx.len()));
        let (details, details_rem) =
            details.split_at_mut(two_taps_size_for_input(input.len(), details.len()));

        unsafe {
            let l0 = vld1q_f32(self.low_pass.as_ptr());
            let h0 = vld1q_f32(self.high_pass.as_ptr());

            let mut processed = 0usize;

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(8)
                .zip(details.chunks_exact_mut(8))
                .enumerate()
            {
                let base0 = 2 * 8 * i;

                let input0 = input.get_unchecked(base0..);

                let xw0 = vld1q_f32(input0.as_ptr());
                let xw1 = vld1q_f32(input0.get_unchecked(4..).as_ptr());
                let xw2 = vld1q_f32(input0.get_unchecked(8..).as_ptr());
                let xw3 = vld1q_f32(input0.get_unchecked(12..).as_ptr());

                let a0 = vmulq_f32(xw0, l0);
                let d0 = vmulq_f32(xw0, h0);

                let a1 = vmulq_f32(xw1, l0);
                let d1 = vmulq_f32(xw1, h0);

                let a2 = vmulq_f32(xw2, l0);
                let d2 = vmulq_f32(xw2, h0);

                let a3 = vmulq_f32(xw3, l0);
                let d3 = vmulq_f32(xw3, h0);

                vst1q_f32(approx.as_mut_ptr(), vpaddq_f32(a0, a1));
                vst1q_f32(detail.as_mut_ptr(), vpaddq_f32(d0, d1));
                vst1q_f32(
                    approx.get_unchecked_mut(4..).as_mut_ptr(),
                    vpaddq_f32(a2, a3),
                );
                vst1q_f32(
                    detail.get_unchecked_mut(4..).as_mut_ptr(),
                    vpaddq_f32(d2, d3),
                );

                processed += 8;
            }

            let approx = approx.chunks_exact_mut(8).into_remainder();
            let details = details.chunks_exact_mut(8).into_remainder();
            let padded_input = input.get_unchecked(processed * 2..);
            processed = 0;

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(4)
                .zip(details.chunks_exact_mut(4))
                .enumerate()
            {
                let base0 = 2 * 4 * i;

                let input0 = padded_input.get_unchecked(base0..);
                let input1 = padded_input.get_unchecked(base0 + 4..);

                let xw0 = vld1q_f32(input0.as_ptr());
                let xw1 = vld1q_f32(input1.as_ptr());

                let a0 = vmulq_f32(xw0, l0);
                let d0 = vmulq_f32(xw0, h0);

                let a1 = vmulq_f32(xw1, l0);
                let d1 = vmulq_f32(xw1, h0);

                vst1q_f32(approx.as_mut_ptr(), vpaddq_f32(a0, a1));
                vst1q_f32(detail.as_mut_ptr(), vpaddq_f32(d0, d1));

                processed += 4;
            }

            let approx = approx.chunks_exact_mut(4).into_remainder();
            let details = details.chunks_exact_mut(4).into_remainder();
            let padded_input = padded_input.get_unchecked(processed * 2..);

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * i;

                let input = padded_input.get_unchecked(base..);

                let xw = vld1_f32(input.as_ptr());

                let a = vmul_f32(xw, vget_low_f32(l0));
                let d = vmul_f32(xw, vget_low_f32(h0));

                let q0 = vpadd_f32(a, d);

                vst1_lane_f32::<0>(approx, q0);
                vst1_lane_f32::<1>(detail, q0);
            }

            if !details_rem.is_empty() && !approx_rem.is_empty() {
                let i = half - 1;
                let x0 = input.get_unchecked(2 * i);
                let x1 = self.border_mode.interpolate(
                    input,
                    2 * i as isize + 1,
                    0,
                    input.len() as isize,
                );

                let mut a = 0.0f64.as_();
                let mut d = 0.0f64.as_();

                a = fmla(self.low_pass[0], *x0, a);
                d = fmla(self.high_pass[0], *x0, d);

                a = fmla(self.low_pass[1], x1, a);
                d = fmla(self.high_pass[1], x1, d);

                *approx_rem.last_mut().unwrap() = a;
                *details_rem.last_mut().unwrap() = d;
            }
        }
        Ok(())
    }

    fn required_scratch_size(&self, _: usize) -> usize {
        0
    }

    fn dwt_size(&self, input_length: usize) -> DwtSize {
        DwtSize::new(dwt_length(input_length, self.filter_length()))
    }
}

impl DwtInverseExecutor<f32> for NeonWavelet2TapsF32 {
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
            return Err(OscletError::OutputSizeIsNotValid(output.len(), rec_len));
        }

        const FILTER_OFFSET: usize = 0;
        const FILTER_LENGTH: usize = 2;

        unsafe {
            let safe_start = FILTER_OFFSET;
            // 2*x - off + len >= output.len()
            // x >= (output.len() + off - len)/2
            let mut safe_end = ((output.len() + FILTER_OFFSET).saturating_sub(FILTER_LENGTH)) / 2;

            if safe_start < safe_end {
                for i in 0..safe_start {
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    for j in 0..2 {
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

                let l0 = vld1q_f32(self.low_pass.as_ptr());
                let h0 = vld1q_f32(self.high_pass.as_ptr());

                let mut uq = safe_start;
                static SH: [u8; 16] = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7];
                let sh = vld1q_u8(SH.as_ptr());

                while uq + 2 < safe_end {
                    let (h, g) = (
                        vld1_f32(approx.get_unchecked(uq..).as_ptr()),
                        vld1_f32(details.get_unchecked(uq..).as_ptr()),
                    );
                    let fh = vreinterpretq_f32_u8(vqtbl1q_u8(
                        vreinterpretq_u8_f32(vcombine_f32(h, h)),
                        sh,
                    ));
                    let fg = vreinterpretq_f32_u8(vqtbl1q_u8(
                        vreinterpretq_u8_f32(vcombine_f32(g, g)),
                        sh,
                    ));
                    let k0 = 2 * uq as isize - FILTER_OFFSET as isize;
                    let part0_src = output.get_unchecked(k0 as usize..);
                    let xw0 = vld1q_f32(part0_src.as_ptr());
                    let q0 = vfmaq_f32(vfmaq_f32(xw0, h0, fg), l0, fh);
                    vst1q_f32(output.get_unchecked_mut(k0 as usize..).as_mut_ptr(), q0);
                    uq += 2;
                }

                for i in uq..safe_end {
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    let part = output.get_unchecked_mut(k as usize..);
                    let xw = vld1_f32(part.as_ptr());
                    let q = vfma_n_f32(vfma_n_f32(xw, vget_low_f32(h0), g), vget_low_f32(l0), h);
                    vst1_f32(part.as_mut_ptr(), q);
                }
            } else {
                safe_end = 0usize;
            }

            for i in safe_end..approx.len() {
                let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..2 {
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
        idwt_length(input_length.approx_length, self.filter_length())
    }
}

impl IncompleteDwtExecutor<f32> for NeonWavelet2TapsF32 {
    fn filter_length(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DaubechiesFamily, WaveletFilterProvider};

    #[test]
    fn test_db1_odd() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db1 = NeonWavelet2TapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db1
                .get_wavelet()
                .as_ref()
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
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db1 = NeonWavelet2TapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db1
                .get_wavelet()
                .as_ref()
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
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db1 = NeonWavelet2TapsF32::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db1
                .get_wavelet()
                .as_ref()
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
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db1 = NeonWavelet2TapsF32::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db1
                .get_wavelet()
                .as_ref()
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
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.0,
            1.0,
        ];
        let db1 = NeonWavelet2TapsF32::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db1
                .get_wavelet()
                .as_ref()
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
