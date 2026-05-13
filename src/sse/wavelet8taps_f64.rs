/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
use crate::sse::sse_vector_d::SseVectorD;
use crate::util::{dwt_length, eight_taps_size_for_input, idwt_length, low_pass_to_high_from_arr};
use crate::{DwtForwardExecutor, DwtInverseExecutor, DwtSize, IncompleteDwtExecutor};

pub(crate) struct SseWavelet8TapsF64 {
    border_mode: BorderMode,
    low_pass: [f64; 8],
    high_pass: [f64; 8],
}

impl SseWavelet8TapsF64 {
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

impl DwtForwardExecutor<f64> for SseWavelet8TapsF64 {
    fn execute_forward(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
    ) -> Result<(), OscletError> {
        let mut scratch = try_vec![f64::default(); self.required_scratch_size(input.len())];
        unsafe { self.execute_forward_impl(input, approx, details, &mut scratch) }
    }

    fn execute_forward_with_scratch(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
        scratch: &mut [f64],
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

impl SseWavelet8TapsF64 {
    #[target_feature(enable = "sse4.2")]
    fn execute_forward_impl(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
        scratch: &mut [f64],
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

        let required_size = self.required_scratch_size(input.len());
        if scratch.len() < required_size {
            return Err(OscletError::ScratchSize(required_size, scratch.len()));
        }

        unsafe {
            let h0 = SseVectorD::load(&self.low_pass);
            let g0 = SseVectorD::load(&self.high_pass);

            let h1 = SseVectorD::load(&self.low_pass[2..]);
            let g1 = SseVectorD::load(&self.high_pass[2..]);

            let h2 = SseVectorD::load(&self.low_pass[4..]);
            let g2 = SseVectorD::load(&self.high_pass[4..]);

            let h3 = SseVectorD::load(&self.low_pass[6..]);
            let g3 = SseVectorD::load(&self.high_pass[6..]);

            let interpolation = BorderInterpolation::new(self.border_mode, 0, input.len() as isize);

            let (front_approx, approx) = approx.split_at_mut(3);
            let (front_detail, details) = details.split_at_mut(3);

            for (i, (approx, detail)) in front_approx
                .iter_mut()
                .zip(front_detail.iter_mut())
                .enumerate()
            {
                let base = 2 * i as isize - 6;

                let x0 = interpolation.interpolate(input, base);
                let x1 = interpolation.interpolate(input, base + 1);
                let x2 = interpolation.interpolate(input, base + 2);
                let x3 = interpolation.interpolate(input, base + 3);
                let x4 = interpolation.interpolate(input, base + 4);
                let x5 = interpolation.interpolate(input, base + 5);
                let x6 = *input.get_unchecked((base + 6) as usize);
                let x7 = *input.get_unchecked((base + 7) as usize);

                let x01 = SseVectorD::from_elements(x0, x1);
                let x23 = SseVectorD::from_elements(x2, x3);
                let x45 = SseVectorD::from_elements(x4, x5);
                let x67 = SseVectorD::from_elements(x6, x7);

                let mut wa = x67.mul_add(h3, x45.mul_add(h2, x23.mul_add(h1, x01 * h0)));
                let mut wd = x67.mul_add(g3, x45.mul_add(g2, x23.mul_add(g1, x01 * g0)));

                wa = wa.hsum();
                wd = wd.hsum();

                wa.write1(approx);
                wd.write1(detail);
            }

            let (approx, approx_rem) =
                approx.split_at_mut(eight_taps_size_for_input(input.len(), approx.len()));
            let (details, details_rem) =
                details.split_at_mut(eight_taps_size_for_input(input.len(), details.len()));

            let base_start = approx.len();

            let mut processed = 0usize;

            for (i, (approx, detail)) in approx
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(details.as_chunks_mut::<4>().0.iter_mut())
                .enumerate()
            {
                let base0 = 2 * 4 * i;

                let input0 = input.get_unchecked(base0..);

                let xw00 = SseVectorD::load(input0);
                let xw01 = SseVectorD::load(input0.get_unchecked(2..));
                let xw02 = SseVectorD::load(input0.get_unchecked(4..));
                let xw03 = SseVectorD::load(input0.get_unchecked(6..));
                let xw04 = SseVectorD::load(input0.get_unchecked(8..));
                let xw05 = SseVectorD::load(input0.get_unchecked(10..));
                let xw06 = SseVectorD::load(input0.get_unchecked(12..));

                let a0 = xw03.mul_add(h3, xw02.mul_add(h2, xw01.mul_add(h1, xw00 * h0)));
                let d0 = xw03.mul_add(g3, xw02.mul_add(g2, xw01.mul_add(g1, xw00 * g0)));

                let a1 = xw04.mul_add(h3, xw03.mul_add(h2, xw02.mul_add(h1, xw01 * h0)));
                let d1 = xw04.mul_add(g3, xw03.mul_add(g2, xw02.mul_add(g1, xw01 * g0)));

                let a2 = xw05.mul_add(h3, xw04.mul_add(h2, xw03.mul_add(h1, xw02 * h0)));
                let d2 = xw05.mul_add(g3, xw04.mul_add(g2, xw03.mul_add(g1, xw02 * g0)));

                let a3 = xw06.mul_add(h3, xw05.mul_add(h2, xw04.mul_add(h1, xw03 * h0)));
                let d3 = xw06.mul_add(g3, xw05.mul_add(g2, xw04.mul_add(g1, xw03 * g0)));

                let q0 = a0.hadd(a1);
                let q1 = a2.hadd(a3);

                let fq0 = d0.hadd(d1);
                let fq1 = d2.hadd(d3);

                q0.write(approx);
                q1.write(approx.get_unchecked_mut(2..));
                fq0.write(detail);
                fq1.write(detail.get_unchecked_mut(2..));

                processed += 4;
            }

            let approx = approx.chunks_exact_mut(4).into_remainder();
            let details = details.chunks_exact_mut(4).into_remainder();

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * (i + processed);

                let input = input.get_unchecked(base..);

                let x01 = SseVectorD::load(input);
                let x23 = SseVectorD::load(input.get_unchecked(2..));
                let x45 = SseVectorD::load(input.get_unchecked(4..));
                let x67 = SseVectorD::load(input.get_unchecked(6..));

                let mut wa = x67.mul_add(h3, x45.mul_add(h2, x23.mul_add(h1, x01 * h0)));
                let mut wd = x67.mul_add(g3, x45.mul_add(g2, x23.mul_add(g1, x01 * g0)));

                wa = wa.hsum();
                wd = wd.hsum();

                wa.write1(approx);
                wd.write1(detail);
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

                let x01 = SseVectorD::from_elements(x0, x1);
                let x23 = SseVectorD::from_elements(x2, x3);
                let x45 = SseVectorD::from_elements(x4, x5);
                let x67 = SseVectorD::from_elements(x6, x7);

                let mut wa = x67.mul_add(h3, x45.mul_add(h2, x23.mul_add(h1, x01 * h0)));
                let mut wd = x67.mul_add(g3, x45.mul_add(g2, x23.mul_add(g1, x01 * g0)));

                wa = wa.hsum();
                wd = wd.hsum();

                wa.write1(approx);
                wd.write1(detail);
            }
        }
        Ok(())
    }
}

impl DwtInverseExecutor<f64> for SseWavelet8TapsF64 {
    fn execute_inverse(
        &self,
        approx: &[f64],
        details: &[f64],
        output: &mut [f64],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_inverse_impl(approx, details, output) }
    }

    fn idwt_size(&self, input_length: DwtSize) -> usize {
        idwt_length(input_length.approx_length, self.filter_length())
    }
}

impl SseWavelet8TapsF64 {
    #[target_feature(enable = "sse4.2")]
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

        let rec_len = idwt_length(approx.len(), 8);

        if output.len() != rec_len {
            return Err(OscletError::OutputSizeIsNotValid(output.len(), rec_len));
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

                let h0 = SseVectorD::load(&self.low_pass);
                let g0 = SseVectorD::load(&self.high_pass);

                let h1 = SseVectorD::load(&self.low_pass[2..]);
                let g1 = SseVectorD::load(&self.high_pass[2..]);

                let h2 = SseVectorD::load(&self.low_pass[4..]);
                let g2 = SseVectorD::load(&self.high_pass[4..]);

                let h3 = SseVectorD::load(&self.low_pass[6..]);
                let g3 = SseVectorD::load(&self.high_pass[6..]);

                let mut ui = safe_start;

                while ui + 2 <= safe_end {
                    let (h, g) = (
                        SseVectorD::load(approx.get_unchecked(ui..)),
                        SseVectorD::load(details.get_unchecked(ui..)),
                    );
                    let k = 2 * ui as isize - FILTER_OFFSET as isize;
                    let part0 = output.get_unchecked_mut(k as usize..);
                    let q0 = SseVectorD::load(part0);
                    let q1 = SseVectorD::load(part0.get_unchecked(2..));
                    let q2 = SseVectorD::load(part0.get_unchecked(4..));
                    let q3 = SseVectorD::load(part0.get_unchecked(6..));
                    let q4 = SseVectorD::load(part0.get_unchecked(8..));

                    let wh0 = h.duplicate_lo();
                    let wg0 = g.duplicate_lo();

                    let w0 = g0.mul_add(wg0, h0.mul_add(wh0, q0));
                    let w1 = g1.mul_add(wg0, h1.mul_add(wh0, q1));
                    let w2 = g2.mul_add(wg0, h2.mul_add(wh0, q2));
                    let w3 = g3.mul_add(wg0, h3.mul_add(wh0, q3));

                    let wh1 = h.duplicate_hi();
                    let wg1 = g.duplicate_hi();

                    let w4 = g0.mul_add(wg1, h0.mul_add(wh1, w1));
                    let w5 = g1.mul_add(wg1, h1.mul_add(wh1, w2));
                    let w6 = g2.mul_add(wg1, h2.mul_add(wh1, w3));
                    let w7 = g3.mul_add(wg1, h3.mul_add(wh1, q4));

                    w0.write(part0);
                    w4.write(part0.get_unchecked_mut(2..));
                    w5.write(part0.get_unchecked_mut(4..));
                    w6.write(part0.get_unchecked_mut(6..));
                    w7.write(part0.get_unchecked_mut(8..));
                    ui += 2;
                }

                for i in ui..safe_end {
                    let (h, g) = (
                        SseVectorD::load1(approx.get_unchecked(i..)),
                        SseVectorD::load1(details.get_unchecked(i..)),
                    );
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    let part = output.get_unchecked_mut(k as usize..);

                    let w0 = SseVectorD::load(part);
                    let w1 = SseVectorD::load(part.get_unchecked(2..));
                    let w2 = SseVectorD::load(part.get_unchecked(4..));
                    let w3 = SseVectorD::load(part.get_unchecked(6..));

                    let q0 = g0.mul_add(g, h0.mul_add(h, w0));
                    let q2 = g1.mul_add(g, h1.mul_add(h, w1));
                    let q4 = g2.mul_add(g, h2.mul_add(h, w2));
                    let q6 = g3.mul_add(g, h3.mul_add(h, w3));

                    q0.write(part);
                    q2.write(part.get_unchecked_mut(2..));
                    q4.write(part.get_unchecked_mut(4..));
                    q6.write(part.get_unchecked_mut(6..));
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

impl IncompleteDwtExecutor<f64> for SseWavelet8TapsF64 {
    fn filter_length(&self) -> usize {
        8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factory::has_valid_sse;
    use crate::{DaubechiesFamily, WaveletFilterProvider};

    #[test]
    fn test_db4_odd() {
        if !has_valid_sse() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db4 = SseWavelet8TapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db4
                .get_wavelet()
                .as_ref()
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
        if !has_valid_sse() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db4 = SseWavelet8TapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db4
                .get_wavelet()
                .as_ref()
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
