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
use crate::avx::util::{
    _mm_unpack2lo_ps, _mm256_hpadd2_ps, _mm256_hsum_ps, _mm256_permute_ps64, shuffle,
};
use crate::border_mode::{BorderInterpolation, BorderMode};
use crate::err::{OscletError, try_vec};
use crate::util::{dwt_length, eight_taps_size_for_input, idwt_length, low_pass_to_high_from_arr};
use crate::{DwtForwardExecutor, DwtInverseExecutor, DwtSize, IncompleteDwtExecutor};
use std::arch::x86_64::*;

pub(crate) struct AvxWavelet8TapsF32 {
    border_mode: BorderMode,
    low_pass: [f32; 8],
    high_pass: [f32; 8],
}

impl AvxWavelet8TapsF32 {
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[f32; 8]) -> Self {
        Self {
            border_mode,
            low_pass: *wavelet,
            high_pass: low_pass_to_high_from_arr(wavelet),
        }
    }
}

impl DwtForwardExecutor<f32> for AvxWavelet8TapsF32 {
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

impl AvxWavelet8TapsF32 {
    #[target_feature(enable = "avx2", enable = "fma")]
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
            let interpolation = BorderInterpolation::new(self.border_mode, 0, input.len() as isize);

            let (front_approx, approx) = approx.split_at_mut(3);
            let (front_detail, details) = details.split_at_mut(3);

            let h0 = _mm256_loadu_ps(self.low_pass.as_ptr());
            let g0 = _mm256_loadu_ps(self.high_pass.as_ptr());

            let mut processed = 0usize;

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

                let vals = _mm256_setr_ps(x0, x1, x2, x3, x4, x5, *x6, *x7);
                let a = _mm256_mul_ps(vals, h0);
                let d = _mm256_mul_ps(vals, g0);

                let a = _mm256_hsum_ps(a);
                let d = _mm256_hsum_ps(d);

                _mm_store_ss(approx, a);
                _mm_store_ss(detail, d);
            }

            let (approx, approx_rem) =
                approx.split_at_mut(eight_taps_size_for_input(input.len(), approx.len()));
            let (details, details_rem) =
                details.split_at_mut(eight_taps_size_for_input(input.len(), details.len()));

            let base_start = approx.len();

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(4)
                .zip(details.chunks_exact_mut(4))
                .enumerate()
            {
                let base0 = 2 * 4 * i;

                let input0 = input.get_unchecked(base0..);

                let xw00 = _mm256_loadu_ps(input0.as_ptr());
                let xw01 = _mm256_loadu_ps(input0.get_unchecked(2..).as_ptr());
                let xw02 = _mm256_loadu_ps(input0.get_unchecked(4..).as_ptr());
                let xw03 = _mm256_loadu_ps(input0.get_unchecked(6..).as_ptr());

                let a0 = _mm256_mul_ps(xw00, h0);
                let d0 = _mm256_mul_ps(xw00, g0);

                let a1 = _mm256_mul_ps(xw01, h0);
                let d1 = _mm256_mul_ps(xw01, g0);

                let a2 = _mm256_mul_ps(xw02, h0);
                let d2 = _mm256_mul_ps(xw02, g0);

                let a3 = _mm256_mul_ps(xw03, h0);
                let d3 = _mm256_mul_ps(xw03, g0);

                let wa0 = _mm256_hpadd2_ps(_mm256_hpadd2_ps(a0, a1), _mm256_hpadd2_ps(a2, a3));
                let wd0 = _mm256_hpadd2_ps(_mm256_hpadd2_ps(d0, d1), _mm256_hpadd2_ps(d2, d3));

                let wa2 = _mm_hadd_ps(_mm256_castps256_ps128(wa0), _mm256_extractf128_ps::<1>(wa0));
                let wd2 = _mm_hadd_ps(_mm256_castps256_ps128(wd0), _mm256_extractf128_ps::<1>(wd0));

                _mm_storeu_ps(approx.as_mut_ptr(), wa2);
                _mm_storeu_ps(detail.as_mut_ptr(), wd2);

                processed += 4;
            }

            let approx = approx.chunks_exact_mut(4).into_remainder();
            let details = details.chunks_exact_mut(4).into_remainder();

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * (i + processed);

                let input = input.get_unchecked(base..);

                let x0 = _mm256_loadu_ps(input.as_ptr());

                let a = _mm256_mul_ps(x0, h0);
                let d = _mm256_mul_ps(x0, g0);

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

                let vals = _mm256_setr_ps(x0, x1, x2, x3, x4, x5, x6, x7);
                let a = _mm256_mul_ps(vals, h0);
                let d = _mm256_mul_ps(vals, g0);

                let a = _mm256_hsum_ps(a);
                let d = _mm256_hsum_ps(d);

                _mm_store_ss(approx, a);
                _mm_store_ss(detail, d);
            }
        }
        Ok(())
    }
}

impl DwtInverseExecutor<f32> for AvxWavelet8TapsF32 {
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

impl AvxWavelet8TapsF32 {
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
                    let (h, g) = (
                        _mm_set_ss(*approx.get_unchecked(i)),
                        _mm_set_ss(*details.get_unchecked(i)),
                    );
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    for j in 0..8 {
                        let k = k + j as isize;
                        if k >= 0 && k < rec_len as isize {
                            let mut w = _mm_fmadd_ss(
                                _mm_set1_ps(self.high_pass[j]),
                                g,
                                _mm_load_ss(output.get_unchecked(k as usize) as *const f32),
                            );
                            w = _mm_fmadd_ss(_mm_set1_ps(self.low_pass[j]), h, w);
                            _mm_store_ss(output.get_unchecked_mut(k as usize) as *mut f32, w);
                        }
                    }
                }

                let h0 = _mm256_loadu_ps(self.low_pass.as_ptr());
                let g0 = _mm256_loadu_ps(self.high_pass.as_ptr());

                // this is ugly as we need for each step 8 items, 6 after previous stage
                // and append new 2 to the end. Repeat 4 times.

                let mut ui = safe_start;

                while ui + 4 < safe_end {
                    let (h, g) = (
                        _mm_loadu_ps(approx.get_unchecked(ui)),
                        _mm_loadu_ps(details.get_unchecked(ui)),
                    );

                    let h = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(h), h);
                    let g = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(g), g);

                    let k = 2 * ui as isize - FILTER_OFFSET as isize;
                    let part0 = output.get_unchecked_mut(k as usize..);
                    let q0 = _mm256_loadu_ps(part0.as_ptr());
                    let q1 = _mm256_setr_m128(
                        _mm_loadu_ps(part0.get_unchecked(8..).as_ptr()),
                        _mm_castsi128_ps(_mm_loadu_si64(part0.get_unchecked(12..).as_ptr().cast())),
                    );

                    let wh0 = _mm256_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(h, h);
                    let wg0 = _mm256_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(g, g);

                    let w0 = _mm256_fmadd_ps(g0, wg0, _mm256_fmadd_ps(h0, wh0, q0)); // first done item the lowest two w0

                    let interim_w0 = _mm256_permute_ps64::<{ shuffle(0, 3, 2, 1) }>(w0);
                    let t0 = _mm256_permute_ps64::<{ shuffle(1, 0, 1, 0) }>(q1);
                    let iw0 = _mm256_blend_ps::<0b11000000>(interim_w0, t0);

                    let wh1 = _mm256_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(h, h);
                    let wg1 = _mm256_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(g, g);

                    let w1 = _mm256_fmadd_ps(g0, wg1, _mm256_fmadd_ps(h0, wh1, iw0)); // second item done the lowest two w1

                    let packed_lo1 =
                        _mm_unpack2lo_ps(_mm256_castps256_ps128(w0), _mm256_castps256_ps128(w1));

                    let interim_w1 = _mm256_permute_ps64::<{ shuffle(0, 3, 2, 1) }>(w1);
                    let t1 = _mm256_permute_ps64::<{ shuffle(0, 1, 0, 1) }>(q1);
                    let iw2 = _mm256_blend_ps::<0b11000000>(interim_w1, t1);

                    let wh2 = _mm256_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(h, h);
                    let wg2 = _mm256_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(g, g);

                    let w2 = _mm256_fmadd_ps(g0, wg2, _mm256_fmadd_ps(h0, wh2, iw2)); // third item done the lowest two w2

                    let interim_w2 = _mm256_permute_ps64::<{ shuffle(0, 3, 2, 1) }>(w2);
                    let t2 = _mm256_permute_ps64::<{ shuffle(3, 2, 3, 2) }>(q1);
                    let iw3 = _mm256_blend_ps::<0b11000000>(interim_w2, t2);

                    let wh3 = _mm256_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(h, h);
                    let wg3 = _mm256_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(g, g);

                    let w3 = _mm256_fmadd_ps(g0, wg3, _mm256_fmadd_ps(h0, wh3, iw3)); // fourth item done the lowest two w2

                    let packed_lo2 =
                        _mm_unpack2lo_ps(_mm256_castps256_ps128(w2), _mm256_castps256_ps128(w3));

                    let packed =
                        _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(packed_lo1), packed_lo2);
                    _mm256_storeu_ps(part0.as_mut_ptr(), packed);
                    let tail = _mm256_permute_ps64::<{ shuffle(0, 3, 2, 1) }>(w3);
                    _mm_storeu_ps(
                        part0.get_unchecked_mut(8..).as_mut_ptr(),
                        _mm256_castps256_ps128(tail),
                    );
                    _mm_storel_pd(
                        part0.get_unchecked_mut(12..).as_mut_ptr().cast(),
                        _mm_castps_pd(_mm256_extractf128_ps::<1>(tail)),
                    );
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

                    let q0 = _mm256_fmadd_ps(g0, g, _mm256_fmadd_ps(h0, h, x0));

                    _mm256_storeu_ps(part.as_mut_ptr(), q0);
                }
            } else {
                safe_end = 0usize;
            }

            for i in safe_end..approx.len() {
                let (h, g) = (
                    _mm_set_ss(*approx.get_unchecked(i)),
                    _mm_set_ss(*details.get_unchecked(i)),
                );
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..8 {
                    let k = k + j as isize;
                    if k >= 0 && k < rec_len as isize {
                        let mut w = _mm_fmadd_ss(
                            _mm_set1_ps(self.high_pass[j]),
                            g,
                            _mm_load_ss(output.get_unchecked(k as usize) as *const f32),
                        );
                        w = _mm_fmadd_ss(_mm_set1_ps(self.low_pass[j]), h, w);
                        _mm_store_ss(output.get_unchecked_mut(k as usize) as *mut f32, w);
                    }
                }
            }
        }
        Ok(())
    }
}

impl IncompleteDwtExecutor<f32> for AvxWavelet8TapsF32 {
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
        let db4 = AvxWavelet8TapsF32::new(
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
        if !has_avx_with_fma() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db4 = AvxWavelet8TapsF32::new(
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
        if !has_avx_with_fma() {
            return;
        }
        let data_length = 86;
        let mut input = vec![0.; data_length];
        for i in 0..data_length {
            input[i] = i as f32 / data_length as f32;
        }
        let db4 = AvxWavelet8TapsF32::new(
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
