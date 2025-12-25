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
use crate::border_mode::BorderMode;
use crate::err::{OscletError, try_vec};
use crate::filter_padding::write_arena_1d;
use crate::mla::fmla;
use crate::sse::util::{_mm_fma_pd, _mm_hsum_pd};
use crate::util::{dwt_length, idwt_length, low_pass_to_high};
use crate::{DwtForwardExecutor, DwtInverseExecutor, DwtSize, IncompleteDwtExecutor};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) struct SseWaveletNTapsF64 {
    border_mode: BorderMode,
    low_pass: Vec<f64>,
    high_pass: Vec<f64>,
    filter_length: usize,
}

impl SseWaveletNTapsF64 {
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[f64]) -> Self {
        Self {
            border_mode,
            filter_length: wavelet.len(),
            high_pass: low_pass_to_high(wavelet),
            low_pass: wavelet.to_vec(),
        }
    }
}

impl DwtForwardExecutor<f64> for SseWaveletNTapsF64 {
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

    fn required_scratch_size(&self, input_length: usize) -> usize {
        let half = dwt_length(input_length, self.filter_length);
        let whole_pad_size = (2 * half + self.filter_length - 2) - input_length;
        let left_pad = whole_pad_size / 2;
        let right_pad = whole_pad_size - left_pad;
        left_pad + right_pad + input_length
    }

    fn dwt_size(&self, input_length: usize) -> DwtSize {
        DwtSize::new(dwt_length(input_length, self.filter_length()))
    }
}

impl SseWaveletNTapsF64 {
    #[target_feature(enable = "sse4.2")]
    fn execute_forward_impl(
        &self,
        input: &[f64],
        approx: &mut [f64],
        details: &mut [f64],
        scratch: &mut [f64],
    ) -> Result<(), OscletError> {
        let half = dwt_length(input.len(), self.filter_length);

        if input.len() < self.filter_length {
            return Err(OscletError::MinFilterSize(input.len(), self.filter_length));
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

        let (padded_input, _) = scratch.split_at_mut(required_size);

        let whole_pad_size = (2 * half + self.filter_length - 2) - input.len();
        let left_pad = whole_pad_size / 2;
        let right_pad = whole_pad_size - left_pad;

        write_arena_1d(input, padded_input, left_pad, right_pad, self.border_mode)?;

        unsafe {
            let mut processed = 0usize;

            for (i, (approx, detail)) in approx
                .chunks_exact_mut(2)
                .zip(details.chunks_exact_mut(2))
                .enumerate()
            {
                let base = 2 * 2 * i;

                let input = padded_input.get_unchecked(base..);

                let mut a0 = _mm_setzero_pd();
                let mut d0 = _mm_setzero_pd();

                let mut a1 = _mm_setzero_pd();
                let mut d1 = _mm_setzero_pd();

                let h = &self.low_pass;
                let g = &self.high_pass;

                let mut u = 0usize;

                while u + 4 < self.filter_length {
                    let q0 = _mm_loadu_pd(input.get_unchecked(u..).as_ptr());
                    let q1 = _mm_loadu_pd(input.get_unchecked(u + 2..).as_ptr());
                    let q2 = _mm_loadu_pd(input.get_unchecked(u + 4..).as_ptr());

                    let h0 = _mm_loadu_pd(h.get_unchecked(u..).as_ptr());
                    let h1 = _mm_loadu_pd(h.get_unchecked(u + 2..).as_ptr());

                    let g0 = _mm_loadu_pd(g.get_unchecked(u..).as_ptr());
                    let g1 = _mm_loadu_pd(g.get_unchecked(u + 2..).as_ptr());

                    a0 = _mm_fma_pd(h1, q1, _mm_fma_pd(h0, q0, a0));
                    d0 = _mm_fma_pd(g1, q1, _mm_fma_pd(g0, q0, d0));

                    a1 = _mm_fma_pd(h1, q2, _mm_fma_pd(h0, q1, a1));
                    d1 = _mm_fma_pd(g1, q2, _mm_fma_pd(g0, q1, d1));

                    u += 4;
                }

                let mut xa = _mm_hadd_pd(a0, a1);
                let mut xd = _mm_hadd_pd(d0, d1);

                while u < self.filter_length {
                    let w = _mm_setr_pd(*input.get_unchecked(u), *input.get_unchecked(u + 2));
                    xa = _mm_fma_pd(
                        w,
                        _mm_load1_pd(self.low_pass.get_unchecked(u..).as_ptr()),
                        xa,
                    );
                    xd = _mm_fma_pd(
                        w,
                        _mm_load1_pd(self.high_pass.get_unchecked(u..).as_ptr()),
                        xd,
                    );
                    u += 1;
                }

                _mm_storeu_pd(approx.as_mut_ptr(), xa);
                _mm_storeu_pd(detail.as_mut_ptr(), xd);

                processed += 2;
            }

            let approx = approx.chunks_exact_mut(2).into_remainder();
            let details = details.chunks_exact_mut(2).into_remainder();
            let padded_input = padded_input.get_unchecked(processed * 2..);

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let base = 2 * i;

                let input = padded_input.get_unchecked(base..base + self.filter_length);

                let mut a = _mm_setzero_pd();
                let mut d = _mm_setzero_pd();

                for ((src, g), h) in input
                    .chunks_exact(4)
                    .zip(self.high_pass.chunks_exact(4))
                    .zip(self.low_pass.chunks_exact(4))
                {
                    let q0 = _mm_loadu_pd(src.as_ptr());
                    let q1 = _mm_loadu_pd(src.get_unchecked(2..).as_ptr());

                    a = _mm_fma_pd(
                        _mm_loadu_pd(h.get_unchecked(2..).as_ptr()),
                        q1,
                        _mm_fma_pd(_mm_loadu_pd(h.as_ptr()), q0, a),
                    );
                    d = _mm_fma_pd(
                        _mm_loadu_pd(g.get_unchecked(2..).as_ptr()),
                        q1,
                        _mm_fma_pd(_mm_loadu_pd(g.as_ptr()), q0, d),
                    );
                }

                let input = input.chunks_exact(4).remainder();
                let high_pass = self.high_pass.chunks_exact(4).remainder();
                let low_pass = self.low_pass.chunks_exact(4).remainder();

                let mut a = _mm_hsum_pd(a);
                let mut d = _mm_hsum_pd(d);

                for ((src, g), h) in input.iter().zip(high_pass.iter()).zip(low_pass.iter()) {
                    a = _mm_fma_pd(_mm_load_sd(h), _mm_load_sd(src), a);
                    d = _mm_fma_pd(_mm_load_sd(g), _mm_load_sd(src), d);
                }

                _mm_store_sd(approx, a);
                _mm_store_sd(detail, d);
            }
        }
        Ok(())
    }
}

impl DwtInverseExecutor<f64> for SseWaveletNTapsF64 {
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

impl SseWaveletNTapsF64 {
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

        let rec_len = idwt_length(approx.len(), self.filter_length);

        if output.len() != rec_len {
            return Err(OscletError::OutputSizeIsNotValid(output.len(), rec_len));
        }

        let whole_pad_size = (2 * approx.len() + self.filter_length - 2) - output.len();
        let filter_offset = whole_pad_size / 2;

        unsafe {
            let safe_start = filter_offset;
            // 2*x - off + len >= output.len()
            // x >= (output.len() + off - len)/2
            let mut safe_end =
                ((output.len() + filter_offset).saturating_sub(self.filter_length)) / 2;

            if safe_start < safe_end {
                for i in 0..safe_start.min(safe_end) {
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - filter_offset as isize;
                    for (j, (&wg, &wh)) in
                        self.high_pass.iter().zip(self.low_pass.iter()).enumerate()
                    {
                        let k = k + j as isize;
                        if k >= 0 && k < rec_len as isize {
                            *output.get_unchecked_mut(k as usize) =
                                fmla(wh, h, fmla(wg, g, *output.get_unchecked(k as usize)));
                        }
                    }
                }

                for i in safe_start..safe_end {
                    let (h, g) = (
                        _mm_load1_pd(approx.get_unchecked(i)),
                        _mm_load1_pd(details.get_unchecked(i)),
                    );
                    let k = 2 * i as isize - filter_offset as isize;
                    let part =
                        output.get_unchecked_mut(k as usize..k as usize + self.filter_length);

                    for ((wg, wh), dst) in self
                        .high_pass
                        .chunks_exact(8)
                        .zip(self.low_pass.chunks_exact(8))
                        .zip(part.chunks_exact_mut(8))
                    {
                        let xw0 = _mm_loadu_pd(dst.as_ptr());
                        let xw1 = _mm_loadu_pd(dst.get_unchecked(2..).as_ptr());
                        let xw2 = _mm_loadu_pd(dst.get_unchecked(4..).as_ptr());
                        let xw3 = _mm_loadu_pd(dst.get_unchecked(6..).as_ptr());

                        let q0 = _mm_fma_pd(
                            _mm_loadu_pd(wg.as_ptr()),
                            g,
                            _mm_fma_pd(_mm_loadu_pd(wh.as_ptr()), h, xw0),
                        );
                        let q1 = _mm_fma_pd(
                            _mm_loadu_pd(wg.get_unchecked(2..).as_ptr()),
                            g,
                            _mm_fma_pd(_mm_loadu_pd(wh.get_unchecked(2..).as_ptr()), h, xw1),
                        );
                        let q2 = _mm_fma_pd(
                            _mm_loadu_pd(wg.get_unchecked(4..).as_ptr()),
                            g,
                            _mm_fma_pd(_mm_loadu_pd(wh.get_unchecked(4..).as_ptr()), h, xw2),
                        );
                        let q3 = _mm_fma_pd(
                            _mm_loadu_pd(wg.get_unchecked(6..).as_ptr()),
                            g,
                            _mm_fma_pd(_mm_loadu_pd(wh.get_unchecked(6..).as_ptr()), h, xw3),
                        );

                        _mm_storeu_pd(dst.as_mut_ptr(), q0);
                        _mm_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr(), q1);
                        _mm_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr(), q2);
                        _mm_storeu_pd(dst.get_unchecked_mut(6..).as_mut_ptr(), q3);
                    }

                    let part = part.chunks_exact_mut(8).into_remainder();
                    let high_pass = self.high_pass.chunks_exact(8).remainder();
                    let low_pass = self.low_pass.chunks_exact(8).remainder();

                    for ((wg, wh), dst) in high_pass
                        .chunks_exact(4)
                        .zip(low_pass.chunks_exact(4))
                        .zip(part.chunks_exact_mut(4))
                    {
                        let xw0 = _mm_loadu_pd(dst.as_ptr());
                        let xw1 = _mm_loadu_pd(dst.get_unchecked(2..).as_ptr());

                        let q0 = _mm_fma_pd(
                            _mm_loadu_pd(wg.as_ptr()),
                            g,
                            _mm_fma_pd(_mm_loadu_pd(wh.as_ptr()), h, xw0),
                        );
                        let q1 = _mm_fma_pd(
                            _mm_loadu_pd(wg.get_unchecked(2..).as_ptr()),
                            g,
                            _mm_fma_pd(_mm_loadu_pd(wh.get_unchecked(2..).as_ptr()), h, xw1),
                        );

                        _mm_storeu_pd(dst.as_mut_ptr(), q0);
                        _mm_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr(), q1);
                    }

                    let part = part.chunks_exact_mut(4).into_remainder();
                    let high_pass = high_pass.chunks_exact(4).remainder();
                    let low_pass = low_pass.chunks_exact(4).remainder();

                    for ((wg, wh), dst) in
                        high_pass.iter().zip(low_pass.iter()).zip(part.iter_mut())
                    {
                        let q = _mm_fma_pd(
                            _mm_load_sd(wh),
                            h,
                            _mm_fma_pd(_mm_load_sd(wg), g, _mm_load_sd(dst)),
                        );
                        _mm_store_sd(dst, q);
                    }
                }
            } else {
                safe_end = 0usize;
            }

            for i in safe_end..approx.len() {
                let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                let k = 2 * i as isize - filter_offset as isize;
                for (j, (&wg, &wh)) in self.high_pass.iter().zip(self.low_pass.iter()).enumerate() {
                    let k = k + j as isize;
                    if k >= 0 && k < rec_len as isize {
                        *output.get_unchecked_mut(k as usize) =
                            fmla(wh, h, fmla(wg, g, *output.get_unchecked(k as usize)));
                    }
                }
            }
        }
        Ok(())
    }
}

impl IncompleteDwtExecutor<f64> for SseWaveletNTapsF64 {
    fn filter_length(&self) -> usize {
        self.filter_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factory::has_valid_sse;
    use crate::{DaubechiesFamily, WaveletFilterProvider};

    #[test]
    fn test_db6_odd() {
        if !has_valid_sse() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db4 = SseWaveletNTapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db6
                .get_wavelet()
                .as_ref()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 12);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 14] = [
            4.84994058,
            8.33313622,
            3.34414304,
            1.93194666,
            1.79287863,
            3.99279204,
            4.36571463,
            -0.10632014,
            3.49224622,
            6.06109206,
            7.4732367,
            1.1238786,
            2.46569841,
            2.15261755,
        ];
        const REFERENCE_DETAILS: [f64; 14] = [
            -1.42520254,
            0.36472228,
            -1.08190245,
            0.19274396,
            -0.57938398,
            -3.35136629,
            -0.47340917,
            1.88452864,
            -0.588979,
            1.1753127,
            -0.08996084,
            0.67678864,
            0.26439685,
            1.33359115,
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 12)];
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
    fn test_db6_even() {
        if !has_valid_sse() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db4 = SseWaveletNTapsF64::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db6
                .get_wavelet()
                .as_ref()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 12);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 13] = [
            3.48400304,
            6.11271891,
            7.31500174,
            1.51528892,
            1.11009737,
            3.99279204,
            4.36571463,
            -0.10632014,
            3.48400304,
            6.11271891,
            7.31500174,
            1.51528892,
            1.11009737,
        ];
        const REFERENCE_DETAILS: [f64; 13] = [
            -1.44245557,
            0.33438642,
            -0.98263644,
            0.14896913,
            -0.57278943,
            -3.35136629,
            -0.47340917,
            1.88452864,
            -1.44245557,
            0.33438642,
            -0.98263644,
            0.14896913,
            -0.57278943,
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 12)];
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
    fn test_db22_even() {
        if !has_valid_sse() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 1.0,
            2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 1.0, 2.0,
            3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 1.0, 2.0, 3.0,
            4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 1.0, 2.0, 3.0, 4.0,
            2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 1.0, 2.0, 3.0, 4.0, 2.0,
            1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0,
            0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0,
            1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0,
            2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4,
            6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5,
            2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4,
            6.4, 5.2, 0.6, 0.5, 1.3, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4,
            5.2, 0.6, 0.5, 1.3, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2,
            0.6, 0.5, 1.3, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6,
            0.5, 1.3, 1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 5.4, 1.0, 2.0, 3.0, 4.0, 2.0,
            1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 1.3, 1.3,
        ];
        let wavelet = DaubechiesFamily::Db11.get_wavelet();
        let db4 = SseWaveletNTapsF64::new(BorderMode::Wrap, wavelet.as_ref());
        let out_length = dwt_length(input.len(), wavelet.len());
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), wavelet.len())];
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
