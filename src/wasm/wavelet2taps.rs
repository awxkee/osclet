/*
 * // Copyright (c) Radzivon Bartoshyk 03/2025. All rights reserved.
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
use crate::wasm::wasm_vector::WasmVector;
use crate::wasm::wasm_vector_d::WasmVectorD;
use crate::{BorderMode, DwtForwardExecutor, DwtInverseExecutor, DwtSize, IncompleteDwtExecutor};
use num_traits::AsPrimitive;

pub(crate) struct WasmWavelet2TapsF64 {
    border_mode: BorderMode,
    low_pass: [f64; 2],
    high_pass: [f64; 2],
}

impl WasmWavelet2TapsF64 {
    pub fn new(border_mode: BorderMode, wavelet: &[f64; 2]) -> Self {
        Self {
            border_mode,
            low_pass: *wavelet,
            high_pass: low_pass_to_high_from_arr(wavelet),
        }
    }
}

impl DwtForwardExecutor<f64> for WasmWavelet2TapsF64 {
    fn execute_forward(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
    ) -> Result<(), OscletError> {
        let mut scratch = try_vec![f64::default(); self.required_scratch_size(input.len())];
        self.execute_forward_impl(input, approx, details, &mut scratch)
    }

    fn execute_forward_with_scratch(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), OscletError> {
        self.execute_forward_impl(input, approx, details, scratch)
    }

    fn required_scratch_size(&self, _: usize) -> usize {
        0
    }

    fn dwt_size(&self, input_length: usize) -> DwtSize {
        DwtSize::new(dwt_length(input_length, self.filter_length()))
    }
}

impl WasmWavelet2TapsF64 {
    #[target_feature(enable = "simd128")]
    fn execute_forward_impl(
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

        let (approx, approx_rem) =
            approx.split_at_mut(two_taps_size_for_input(input.len(), approx.len()));
        let (details, details_rem) =
            details.split_at_mut(two_taps_size_for_input(input.len(), details.len()));

        unsafe {
            let l0 = WasmVectorD::load(&self.low_pass);
            let h0 = WasmVectorD::load(&self.high_pass);

            for (i, (approx, detail)) in approx
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(details.as_chunks_mut::<2>().0.iter_mut())
                .enumerate()
            {
                let base0 = 2 * 2 * i;

                let input0 = input.get_unchecked(base0..);
                let input1 = input.get_unchecked(base0 + 2..);

                let xw0 = WasmVectorD::load(input0);
                let xw1 = WasmVectorD::load(input1);

                let a0 = xw0 * l0;
                let d0 = xw0 * h0;

                let a1 = xw1 * l0;
                let d1 = xw1 * h0;

                a0.hadd(a1).write(approx);
                d0.hadd(d1).write(detail);
            }

            let processed = 2 * approx.as_chunks_mut::<2>().0.iter_mut().len() * 2;

            let approx = approx.as_chunks_mut::<2>().1;
            let details = details.as_chunks_mut::<2>().1;

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = processed + 2 * i;

                let input = input.get_unchecked(base..);

                let xw = WasmVectorD::load(input);

                let a = (xw * l0).hsum();
                let d = (xw * h0).hsum();

                a.write1(approx);
                d.write1(detail);
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
}

impl DwtInverseExecutor<f64> for WasmWavelet2TapsF64 {
    fn execute_inverse(
        &self,
        approx: &[f64],
        details: &[f64],
        output: &mut [f64],
    ) -> Result<(), OscletError> {
        self.execute_inverse_impl(approx, details, output)
    }

    fn idwt_size(&self, input_length: DwtSize) -> usize {
        idwt_length(input_length.approx_length, self.filter_length())
    }
}

impl WasmWavelet2TapsF64 {
    #[target_feature(enable = "simd128")]
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

                let l0 = WasmVectorD::load(&self.low_pass);
                let h0 = WasmVectorD::load(&self.high_pass);

                let mut uq = safe_start;
                //
                while uq + 2 < safe_end {
                    let (h, g) = (
                        WasmVectorD::load(approx.get_unchecked(uq..)),
                        WasmVectorD::load(details.get_unchecked(uq..)),
                    );
                    let k0 = 2 * uq as isize - FILTER_OFFSET as isize;
                    let part0_src = output.get_unchecked(k0 as usize..);
                    let part1_src = output.get_unchecked(k0 as usize + 2..);
                    let xw0 = WasmVectorD::load(part0_src);
                    let xw1 = WasmVectorD::load(part1_src);
                    let q0 = l0.mul_add(h.duplicate_lo(), h0.mul_add(g.duplicate_lo(), xw0));
                    let q1 = l0.mul_add(h.duplicate_hi(), h0.mul_add(g.duplicate_hi(), xw1));
                    q0.write(output.get_unchecked_mut(k0 as usize..));
                    q1.write(output.get_unchecked_mut(k0 as usize + 2..));
                    uq += 2;
                }

                for i in uq..safe_end {
                    let (h, g) = (
                        WasmVectorD::load1(approx.get_unchecked(i..)),
                        WasmVectorD::load1(details.get_unchecked(i..)),
                    );
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    let part = output.get_unchecked_mut(k as usize..);
                    let xw = WasmVectorD::load(part);
                    let q = l0.mul_add(h, h0.mul_add(g, xw));
                    q.write(part);
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
}

impl IncompleteDwtExecutor<f64> for WasmWavelet2TapsF64 {
    fn filter_length(&self) -> usize {
        2
    }
}

pub(crate) struct WasmWavelet2TapsF32 {
    border_mode: BorderMode,
    low_pass: [f32; 4],
    high_pass: [f32; 4],
}

impl WasmWavelet2TapsF32 {
    pub fn new(border_mode: BorderMode, wavelet: &[f32; 2]) -> Self {
        let k = low_pass_to_high_from_arr(wavelet);
        Self {
            border_mode,
            low_pass: [wavelet[0], wavelet[1], wavelet[0], wavelet[1]],
            high_pass: [k[0], k[1], k[0], k[1]],
        }
    }
}

impl DwtForwardExecutor<f32> for WasmWavelet2TapsF32 {
    fn execute_forward(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
    ) -> Result<(), OscletError> {
        let mut scratch = try_vec![f32::default(); self.required_scratch_size(input.len())];
        self.execute_forward_impl(input, approx, details, &mut scratch)
    }

    fn execute_forward_with_scratch(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
        scratch: &mut [f32],
    ) -> Result<(), OscletError> {
        self.execute_forward_impl(input, approx, details, scratch)
    }

    fn required_scratch_size(&self, _: usize) -> usize {
        0
    }

    fn dwt_size(&self, input_length: usize) -> DwtSize {
        DwtSize::new(dwt_length(input_length, self.filter_length()))
    }
}

impl WasmWavelet2TapsF32 {
    #[target_feature(enable = "simd128")]
    fn execute_forward_impl(
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
            let l0 = WasmVector::load(self.low_pass.as_slice());
            let h0 = WasmVector::load(self.high_pass.as_slice());

            let mut processed = 0usize;

            for (i, (approx, detail)) in approx
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(details.as_chunks_mut::<8>().0.iter_mut())
                .enumerate()
            {
                let base0 = 2 * 8 * i;

                let input0 = input.get_unchecked(base0..);

                let xw0 = WasmVector::load(input0);
                let xw1 = WasmVector::load(input0.get_unchecked(4..));
                let xw2 = WasmVector::load(input0.get_unchecked(8..));
                let xw3 = WasmVector::load(input0.get_unchecked(12..));

                let a0 = xw0 * l0;
                let d0 = xw0 * h0;

                let a1 = xw1 * l0;
                let d1 = xw1 * h0;

                let a2 = xw2 * l0;
                let d2 = xw2 * h0;

                let a3 = xw3 * l0;
                let d3 = xw3 * h0;

                a0.hadd(a1).write(approx);
                d0.hadd(d1).write(detail);
                a2.hadd(a3).write(approx.get_unchecked_mut(4..));
                d2.hadd(d3).write(detail.get_unchecked_mut(4..));

                processed += 8;
            }

            let approx = approx.as_chunks_mut::<8>().1;
            let details = details.as_chunks_mut::<8>().1;
            let padded_input = input.get_unchecked(processed * 2..);
            processed = 0;

            for (i, (approx, detail)) in approx
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(details.as_chunks_mut::<4>().0.iter_mut())
                .enumerate()
            {
                let base0 = 2 * 4 * i;

                let input0 = padded_input.get_unchecked(base0..);
                let input1 = padded_input.get_unchecked(base0 + 4..);

                let xw0 = WasmVector::load(input0);
                let xw1 = WasmVector::load(input1);

                let a0 = xw0 * l0;
                let d0 = xw0 * h0;

                let a1 = xw1 * l0;
                let d1 = xw1 * h0;

                a0.hadd(a1).write(approx);
                d0.hadd(d1).write(detail);

                processed += 4;
            }

            let approx = approx.as_chunks_mut::<4>().1;
            let details = details.as_chunks_mut::<4>().1;
            let padded_input = padded_input.get_unchecked(processed * 2..);

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * i;

                let input = padded_input.get_unchecked(base..);

                let xw = WasmVector::load2(input);

                let a = (xw * l0).hsum();
                let d = (xw * h0).hsum();

                a.write1(approx);
                d.write1(detail);
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
}

impl DwtInverseExecutor<f32> for WasmWavelet2TapsF32 {
    fn execute_inverse(
        &self,
        approx: &[f32],
        details: &[f32],
        output: &mut [f32],
    ) -> Result<(), OscletError> {
        self.execute_inverse_impl(approx, details, output)
    }

    fn idwt_size(&self, input_length: DwtSize) -> usize {
        idwt_length(input_length.approx_length, self.filter_length())
    }
}

impl WasmWavelet2TapsF32 {
    #[target_feature(enable = "simd128")]
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

                let l0 = WasmVector::load(self.low_pass.as_slice());
                let h0 = WasmVector::load(self.high_pass.as_slice());

                let mut uq = safe_start;

                while uq + 2 < safe_end {
                    let (h, g) = (
                        WasmVector::load2(approx.get_unchecked(uq..)),
                        WasmVector::load2(details.get_unchecked(uq..)),
                    );
                    let fh = h.distribute_00_11();
                    let fg = g.distribute_00_11();
                    let k0 = 2 * uq as isize - FILTER_OFFSET as isize;
                    let part0_src = output.get_unchecked(k0 as usize..);
                    let xw0 = WasmVector::load(part0_src);
                    let q0 = l0.mul_add(fh, h0.mul_add(fg, xw0));
                    q0.write(output.get_unchecked_mut(k0 as usize..));
                    uq += 2;
                }

                for i in uq..safe_end {
                    let (h, g) = (
                        WasmVector::load1(approx.get_unchecked(i..)),
                        WasmVector::load1(details.get_unchecked(i..)),
                    );
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    let part = output.get_unchecked_mut(k as usize..);
                    let xw = WasmVector::load2(part);
                    let q = l0.mul_add(h, h0.mul_add(g, xw));
                    q.write2(part);
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
}

impl IncompleteDwtExecutor<f32> for WasmWavelet2TapsF32 {
    fn filter_length(&self) -> usize {
        2
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use crate::{DaubechiesFamily, WaveletFilterProvider};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn test_db1_odd() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db1 = WasmWavelet2TapsF64::new(
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

    #[wasm_bindgen_test]
    fn test_db1_even() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db1 = WasmWavelet2TapsF64::new(
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

    #[wasm_bindgen_test]
    fn test_db1_odd_f32() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db1 = WasmWavelet2TapsF32::new(
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

    #[wasm_bindgen_test]
    fn test_db1_even_f32() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db1 = WasmWavelet2TapsF32::new(
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

    #[wasm_bindgen_test]
    fn test_db1_even_f32_2() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.0,
            1.0,
        ];
        let db1 = WasmWavelet2TapsF32::new(
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
