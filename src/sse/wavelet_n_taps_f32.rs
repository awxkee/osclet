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
use crate::sse::sse_vector::SseVector;
use crate::util::{dwt_length, idwt_length, low_pass_to_high};
use crate::{DwtForwardExecutor, DwtInverseExecutor, DwtSize, IncompleteDwtExecutor};

pub(crate) struct SseWaveletNTapsF32 {
    border_mode: BorderMode,
    low_pass: Vec<f32>,
    high_pass: Vec<f32>,
    filter_length: usize,
}

impl SseWaveletNTapsF32 {
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[f32]) -> Self {
        Self {
            border_mode,
            filter_length: wavelet.len(),
            high_pass: low_pass_to_high(wavelet),
            low_pass: wavelet.to_vec(),
        }
    }
}

impl DwtForwardExecutor<f32> for SseWaveletNTapsF32 {
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

impl SseWaveletNTapsF32 {
    #[target_feature(enable = "sse4.2")]
    fn execute_forward_impl(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
        scratch: &mut [f32],
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
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(details.as_chunks_mut::<4>().0.iter_mut())
                .enumerate()
            {
                let mut a0 = SseVector::zero();
                let mut d0 = SseVector::zero();

                let mut a1 = SseVector::zero();
                let mut d1 = SseVector::zero();

                let mut a2 = SseVector::zero();
                let mut d2 = SseVector::zero();

                let mut a3 = SseVector::zero();
                let mut d3 = SseVector::zero();

                let base = 2 * 4 * i;

                let input = padded_input.get_unchecked(base..);

                let mut u = 0usize;

                while u + 4 < self.filter_length {
                    let q0 = SseVector::load(input.get_unchecked(u..));
                    let q2 = SseVector::load(input.get_unchecked(u + 4..));
                    let pq2 = SseVector::load2(input.get_unchecked(u + 8..));

                    let q1 = q0.swap_hilo(q2);
                    let q3 = q2.swap_hilo(pq2);

                    let fh = SseVector::load(self.low_pass.get_unchecked(u..));
                    let fg = SseVector::load(self.high_pass.get_unchecked(u..));

                    a0 = fh.mul_add(q0, a0);
                    d0 = fg.mul_add(q0, d0);

                    a1 = fh.mul_add(q1, a1);
                    d1 = fg.mul_add(q1, d1);

                    a2 = fh.mul_add(q2, a2);
                    d2 = fg.mul_add(q2, d2);

                    a3 = fh.mul_add(q3, a3);
                    d3 = fg.mul_add(q3, d3);

                    u += 4;
                }

                let mut xa0 = a0.hadd(a1).hadd(a2.hadd(a3));
                let mut xd0 = d0.hadd(d1).hadd(d2.hadd(d3));

                while u < self.filter_length {
                    let w = SseVector::from_elements(
                        *input.get_unchecked(u),
                        *input.get_unchecked(u + 2),
                        *input.get_unchecked(u + 4),
                        *input.get_unchecked(u + 6),
                    );
                    xa0 = w.mul_add(SseVector::load1(self.low_pass.get_unchecked(u..)), xa0);
                    xd0 = w.mul_add(SseVector::load1(self.high_pass.get_unchecked(u..)), xd0);
                    u += 1;
                }

                xa0.write(approx);
                xd0.write(detail);

                processed += 4;
            }

            let approx = approx.as_chunks_mut::<4>().1;
            let details = details.as_chunks_mut::<4>().1;
            let padded_input = padded_input.get_unchecked(processed * 2..);

            processed = 0usize;

            for (i, (approx, detail)) in approx
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(details.as_chunks_mut::<2>().0.iter_mut())
                .enumerate()
            {
                let mut a0 = SseVector::zero();
                let mut d0 = SseVector::zero();

                let mut a1 = SseVector::zero();
                let mut d1 = SseVector::zero();

                let base = 2 * 2 * i;

                let input = padded_input.get_unchecked(base..);

                let mut u = 0usize;

                while u + 4 < self.filter_length {
                    let q0 = SseVector::load(input.get_unchecked(u..));
                    let pq1 = SseVector::load2(input.get_unchecked(u + 4..));

                    let q1 = q0.swap_hilo(pq1);

                    let fh = SseVector::load(self.low_pass.get_unchecked(u..));
                    let fg = SseVector::load(self.high_pass.get_unchecked(u..));

                    a0 = fh.mul_add(q0, a0);
                    d0 = fg.mul_add(q0, d0);

                    a1 = fh.mul_add(q1, a1);
                    d1 = fg.mul_add(q1, d1);

                    u += 4;
                }

                let mut xa0 = a0
                    .hadd(a0.swap_hilo(SseVector::zero()))
                    .hadd(a1.hadd(a1.swap_hilo(SseVector::zero())));

                xa0 = xa0.pack_evens_in_lo();

                let mut xd0 = d0
                    .hadd(d0.swap_hilo(SseVector::zero()))
                    .hadd(d1.hadd(d1.swap_hilo(SseVector::zero())));

                xd0 = xd0.pack_evens_in_lo();

                while u < self.filter_length {
                    let w = SseVector::from_elements(
                        *input.get_unchecked(u),
                        *input.get_unchecked(u + 2),
                        0.,
                        0.,
                    );
                    xa0 = w.mul_add(SseVector::load1(self.low_pass.get_unchecked(u..)), xa0);
                    xd0 = w.mul_add(SseVector::load1(self.high_pass.get_unchecked(u..)), xd0);
                    u += 1;
                }

                xa0.write2(approx);
                xd0.write2(detail);

                processed += 2;
            }

            let approx = approx.as_chunks_mut::<2>().1;
            let details = details.as_chunks_mut::<2>().1;
            let padded_input = padded_input.get_unchecked(processed * 2..);

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let mut a = SseVector::zero();
                let mut d = SseVector::zero();
                let base = 2 * i;

                let input = padded_input.get_unchecked(base..base + self.filter_length);

                for ((src, g), h) in input
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .zip(self.high_pass.as_chunks::<16>().0.iter())
                    .zip(self.low_pass.as_chunks::<16>().0.iter())
                {
                    let q0 = SseVector::load(src);
                    let q1 = SseVector::load(src.get_unchecked(4..));
                    let q2 = SseVector::load(src.get_unchecked(8..));
                    let q3 = SseVector::load(src.get_unchecked(12..));

                    a = SseVector::load(h).mul_add(q0, a);
                    d = SseVector::load(g).mul_add(q0, d);

                    a = SseVector::load(h.get_unchecked(4..)).mul_add(q1, a);
                    d = SseVector::load(g.get_unchecked(4..)).mul_add(q1, d);

                    a = SseVector::load(h.get_unchecked(8..)).mul_add(q2, a);
                    d = SseVector::load(g.get_unchecked(8..)).mul_add(q2, d);

                    a = SseVector::load(h.get_unchecked(12..)).mul_add(q3, a);
                    d = SseVector::load(g.get_unchecked(12..)).mul_add(q3, d);
                }

                let input = input.as_chunks::<16>().1;
                let high_pass = self.high_pass.as_chunks::<16>().1;
                let low_pass = self.low_pass.as_chunks::<16>().1;

                for ((src, g), h) in input
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(high_pass.as_chunks::<4>().0.iter())
                    .zip(low_pass.as_chunks::<4>().0.iter())
                {
                    let q0 = SseVector::load(src);

                    a = SseVector::load(h).mul_add(q0, a);
                    d = SseVector::load(g).mul_add(q0, d);
                }

                let input = input.as_chunks::<4>().1;
                let high_pass = high_pass.as_chunks::<4>().1;
                let low_pass = low_pass.as_chunks::<4>().1;

                let mut a = a.hsum();
                let mut d = d.hsum();

                for ((src, g), h) in input.iter().zip(high_pass.iter()).zip(low_pass.iter()) {
                    let s = SseVector::load1_lane(src);
                    a = SseVector::load1_lane(h).mul_add(s, a);
                    d = SseVector::load1_lane(g).mul_add(s, d);
                }

                a.write1(approx);
                d.write1(detail);
            }
        }
        Ok(())
    }
}

impl DwtInverseExecutor<f32> for SseWaveletNTapsF32 {
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

impl SseWaveletNTapsF32 {
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
                        SseVector::load1(approx.get_unchecked(i..)),
                        SseVector::load1(details.get_unchecked(i..)),
                    );
                    let k = 2 * i as isize - filter_offset as isize;
                    let part =
                        output.get_unchecked_mut(k as usize..k as usize + self.filter_length);

                    for ((wg, wh), dst) in self
                        .high_pass
                        .as_chunks::<16>()
                        .0
                        .iter()
                        .zip(self.low_pass.as_chunks::<16>().0.iter())
                        .zip(part.as_chunks_mut::<16>().0.iter_mut())
                    {
                        let xw0 = SseVector::load(dst);
                        let xw1 = SseVector::load(dst.get_unchecked(4..));
                        let xw2 = SseVector::load(dst.get_unchecked(8..));
                        let xw3 = SseVector::load(dst.get_unchecked(12..));

                        let q0 =
                            SseVector::load(wg).mul_add(g, SseVector::load(wh).mul_add(h, xw0));
                        let q1 = SseVector::load(wg.get_unchecked(4..))
                            .mul_add(g, SseVector::load(wh.get_unchecked(4..)).mul_add(h, xw1));
                        let q2 = SseVector::load(wg.get_unchecked(8..))
                            .mul_add(g, SseVector::load(wh.get_unchecked(8..)).mul_add(h, xw2));
                        let q3 = SseVector::load(wg.get_unchecked(12..))
                            .mul_add(g, SseVector::load(wh.get_unchecked(12..)).mul_add(h, xw3));

                        q0.write(dst);
                        q1.write(dst.get_unchecked_mut(4..));
                        q2.write(dst.get_unchecked_mut(8..));
                        q3.write(dst.get_unchecked_mut(12..));
                    }

                    let part = part.as_chunks_mut::<16>().1;
                    let high_pass = self.high_pass.as_chunks::<16>().1;
                    let low_pass = self.low_pass.as_chunks::<16>().1;

                    for ((wg, wh), dst) in high_pass
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(low_pass.as_chunks::<4>().0.iter())
                        .zip(part.as_chunks_mut::<4>().0.iter_mut())
                    {
                        let xw0 = SseVector::load(dst);
                        let q0 =
                            SseVector::load(wg).mul_add(g, SseVector::load(wh).mul_add(h, xw0));
                        q0.write(dst);
                    }

                    let part = part.as_chunks_mut::<4>().1;
                    let high_pass = self.high_pass.as_chunks::<4>().1;
                    let low_pass = self.low_pass.as_chunks::<4>().1;

                    for ((wg, wh), dst) in
                        high_pass.iter().zip(low_pass.iter()).zip(part.iter_mut())
                    {
                        let q = SseVector::load1_lane(wh).mul_add(
                            h,
                            SseVector::load1_lane(wg).mul_add(g, SseVector::load1_lane(dst)),
                        );
                        q.write1(dst);
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

impl IncompleteDwtExecutor<f32> for SseWaveletNTapsF32 {
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
        let db4 = SseWaveletNTapsF32::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db6.get_wavelet().as_ref(),
        );
        let out_length = dwt_length(input.len(), 12);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f32; 14] = [
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
        const REFERENCE_DETAILS: [f32; 14] = [
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 12)];
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
    fn test_db6_even() {
        if !has_valid_sse() {
            return;
        }
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db4 = SseWaveletNTapsF32::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db6.get_wavelet().as_ref(),
        );
        let out_length = dwt_length(input.len(), 12);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db4.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f32; 13] = [
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
        const REFERENCE_DETAILS: [f32; 13] = [
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 12)];
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
        let db4 = SseWaveletNTapsF32::new(BorderMode::Wrap, wavelet.as_ref());
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
