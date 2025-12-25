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
use crate::sse::util::{
    _mm_fma_ps, _mm_hsum_ps, _mm_load2_ps, _mm_swap_hilo, _mm_unpack2hi_ps, shuffle,
};
use crate::util::{dwt_length, eight_taps_size_for_input, idwt_length, low_pass_to_high_from_arr};
use crate::{DwtForwardExecutor, DwtInverseExecutor, DwtSize, IncompleteDwtExecutor};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) struct SseWavelet8TapsF32 {
    border_mode: BorderMode,
    low_pass: [f32; 8],
    high_pass: [f32; 8],
}

impl SseWavelet8TapsF32 {
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[f32; 8]) -> Self {
        Self {
            border_mode,
            low_pass: *wavelet,
            high_pass: low_pass_to_high_from_arr(wavelet),
        }
    }
}

impl DwtForwardExecutor<f32> for SseWavelet8TapsF32 {
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

impl SseWavelet8TapsF32 {
    #[target_feature(enable = "sse4.2")]
    fn execute_forward_impl(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
        scratch: &mut [f32],
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
            let h = _mm_loadu_ps(self.low_pass.as_ptr());
            let h2 = _mm_loadu_ps(self.low_pass.get_unchecked(4..).as_ptr());
            let g = _mm_loadu_ps(self.high_pass.as_ptr());
            let g2 = _mm_loadu_ps(self.high_pass.get_unchecked(4..).as_ptr());

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
                let x6 = input.get_unchecked((base + 6) as usize);
                let x7 = input.get_unchecked((base + 7) as usize);

                let val0 = _mm_setr_ps(x0, x1, x2, x3);
                let val1 = _mm_setr_ps(x4, x5, *x6, *x7);

                let a = _mm_fma_ps(val1, h2, _mm_mul_ps(val0, h));
                let d = _mm_fma_ps(val1, g2, _mm_mul_ps(val0, g));

                let ha = _mm_hsum_ps(a);
                let hd = _mm_hsum_ps(d);

                _mm_store_ss(approx, ha);
                _mm_store_ss(detail, hd);
            }

            let (approx, approx_rem) =
                approx.split_at_mut(eight_taps_size_for_input(input.len(), approx.len()));
            let (details, details_rem) =
                details.split_at_mut(eight_taps_size_for_input(input.len(), details.len()));

            let base_start = approx.len();

            let mut processed = 0usize;

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(2)
                .zip(details.chunks_exact_mut(2))
                .enumerate()
            {
                let base = 2 * 2 * i;
                let input = input.get_unchecked(base..);

                let xw = _mm_loadu_ps(input.as_ptr());
                let xw1 = _mm_loadu_ps(input.get_unchecked(4..).as_ptr());
                let xw2 = _mm_load2_ps(input.get_unchecked(8..).as_ptr());

                let xw1_0 = _mm_swap_hilo(xw, xw1);
                let xw1_1 = _mm_swap_hilo(xw1, xw2);

                let a0 = _mm_fma_ps(xw1, h2, _mm_mul_ps(xw, h));
                let d0 = _mm_fma_ps(xw1, g2, _mm_mul_ps(xw, g));

                let a1 = _mm_fma_ps(xw1_1, h2, _mm_mul_ps(xw1_0, h));
                let d1 = _mm_fma_ps(xw1_1, g2, _mm_mul_ps(xw1_0, g));

                let xa = _mm_hadd_ps(a0, a1);
                let xd = _mm_hadd_ps(d0, d1);

                let va = _mm_hadd_ps(xa, _mm_unpack2hi_ps(xa, xa));
                let vd = _mm_hadd_ps(xd, _mm_unpack2hi_ps(xd, xd));

                _mm_storel_pd(approx.as_mut_ptr().cast(), _mm_castps_pd(va));
                _mm_storel_pd(detail.as_mut_ptr().cast(), _mm_castps_pd(vd));

                processed += 2;
            }

            let approx = approx.chunks_exact_mut(2).into_remainder();
            let details = details.chunks_exact_mut(2).into_remainder();

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * (i + processed);
                let input = input.get_unchecked(base..);

                let xw = _mm_loadu_ps(input.as_ptr());
                let xw1 = _mm_loadu_ps(input.get_unchecked(4..).as_ptr());

                let a = _mm_fma_ps(xw1, h2, _mm_mul_ps(xw, h));
                let d = _mm_fma_ps(xw1, g2, _mm_mul_ps(xw, g));

                let a0 = _mm_hsum_ps(a);
                let d0 = _mm_hsum_ps(d);

                _mm_store_ss(approx, a0);
                _mm_store_ss(detail, d0);
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

                let val0 = _mm_setr_ps(x0, x1, x2, x3);
                let val1 = _mm_setr_ps(x4, x5, x6, x7);

                let a = _mm_fma_ps(val1, h2, _mm_mul_ps(val0, h));
                let d = _mm_fma_ps(val1, g2, _mm_mul_ps(val0, g));

                let ha = _mm_hsum_ps(a);
                let hd = _mm_hsum_ps(d);

                _mm_store_ss(approx, ha);
                _mm_store_ss(detail, hd);
            }
        }
        Ok(())
    }
}

impl DwtInverseExecutor<f32> for SseWavelet8TapsF32 {
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

impl SseWavelet8TapsF32 {
    #[target_feature(enable = "sse4.2")]
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

                let h0 = _mm_loadu_ps(self.low_pass.as_ptr());
                let h2 = _mm_loadu_ps(self.low_pass.get_unchecked(4..).as_ptr());
                let g0 = _mm_loadu_ps(self.high_pass.as_ptr());
                let g2 = _mm_loadu_ps(self.high_pass.get_unchecked(4..).as_ptr());

                let mut ui = safe_start;

                while ui + 2 < safe_end {
                    let (h, g) = (
                        _mm_load2_ps(approx.get_unchecked(ui)),
                        _mm_load2_ps(details.get_unchecked(ui)),
                    );
                    let k = 2 * ui as isize - FILTER_OFFSET as isize;
                    let part0 = output.get_unchecked_mut(k as usize..);
                    let q0 = _mm_loadu_ps(part0.as_ptr());
                    let q1 = _mm_loadu_ps(part0.get_unchecked(4..).as_ptr());
                    let q2 = _mm_load2_ps(part0.get_unchecked(8..).as_ptr());

                    let wg0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(g, g);
                    let wh0 = _mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(h, h);

                    let w0 = _mm_fma_ps(g0, wg0, _mm_fma_ps(h0, wh0, q0));
                    let w1 = _mm_fma_ps(g2, wg0, _mm_fma_ps(h2, wh0, q1));

                    let interim_w0 = _mm_swap_hilo(w0, w1);
                    let interim_w1 = _mm_swap_hilo(w1, q2);

                    let wg1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(g, g);
                    let wh1 = _mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(h, h);

                    let w3 = _mm_fma_ps(g0, wg1, _mm_fma_ps(h0, wh1, interim_w0));
                    let w4 = _mm_fma_ps(g2, wg1, _mm_fma_ps(h2, wh1, interim_w1));

                    _mm_storel_pd(part0.as_mut_ptr().cast(), _mm_castps_pd(w0));
                    _mm_storeu_ps(part0.get_unchecked_mut(2..).as_mut_ptr(), w3);
                    _mm_storeu_ps(part0.get_unchecked_mut(6..).as_mut_ptr(), w4);
                    ui += 2;
                }

                for i in ui..safe_end {
                    let (h, g) = (
                        _mm_load1_ps(approx.get_unchecked(i)),
                        _mm_load1_ps(details.get_unchecked(i)),
                    );
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    let part = output.get_unchecked_mut(k as usize..);
                    let xw0 = _mm_loadu_ps(part.as_ptr());
                    let xw1 = _mm_loadu_ps(part.get_unchecked(4..).as_ptr());

                    let q0 = _mm_fma_ps(g0, g, _mm_fma_ps(h0, h, xw0));
                    let q2 = _mm_fma_ps(g2, g, _mm_fma_ps(h2, h, xw1));

                    _mm_storeu_ps(part.as_mut_ptr(), q0);
                    _mm_storeu_ps(part.get_unchecked_mut(4..).as_mut_ptr(), q2);
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

impl IncompleteDwtExecutor<f32> for SseWavelet8TapsF32 {
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
    fn test_db8_odd() {
        if !has_valid_sse() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db4 = SseWavelet8TapsF32::new(
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

        const REFERENCE_APPROX: [f32; 12] = [
            5.40180316, 1.17674293, 2.27895053, 3.08695254, 4.82517499, 0.91029972, 1.96020043,
            5.58301587, 8.40990105, 1.50316223, 2.42249936, 1.81502786,
        ];
        const REFERENCE_DETAILS: [f32; 12] = [
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
                (REFERENCE_APPROX[i] - x).abs() < 1e-3,
                "approx difference expected to be < 1e-3, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-3,
                "details difference expected to be < 1e-3, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 8)];
        db4.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-3,
                "reconstructed difference expected to be < 1e-3, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_db8_even() {
        if !has_valid_sse() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db4 = SseWavelet8TapsF32::new(
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

        const REFERENCE_APPROX: [f32; 11] = [
            8.34997913, 1.83684144, 1.23683239, 3.08695254, 4.82517499, 0.91029972, 1.96020043,
            5.58301587, 8.34997913, 1.83684144, 1.23683239,
        ];
        const REFERENCE_DETAILS: [f32; 11] = [
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
                (REFERENCE_APPROX[i] - x).abs() < 1e-3,
                "approx difference expected to be < 1e-3, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-3,
                "details difference expected to be < 1e-3, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 8)];
        db4.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-3,
                "reconstructed difference expected to be < 1e-3, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_db8_even_big() {
        if !has_valid_sse() {
            return;
        }
        let data_length = 86;
        let mut input = vec![0.; data_length];
        for i in 0..data_length {
            input[i] = i as f32 / data_length as f32;
        }
        let db4 = SseWavelet8TapsF32::new(
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 8)];
        db4.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-3,
                "reconstructed difference expected to be < 1e-3, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }
}
