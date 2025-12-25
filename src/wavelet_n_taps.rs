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
use crate::err::{OscletError, try_vec};
use crate::filter_padding::write_arena_1d;
use crate::mla::fmla;
use crate::util::{dwt_length, idwt_length, low_pass_to_high};
use crate::{
    DwtForwardExecutor, DwtInverseExecutor, DwtSize, IncompleteDwtExecutor, WaveletSample,
};
use num_traits::AsPrimitive;
use std::marker::PhantomData;

pub(crate) struct WaveletNTaps<T> {
    phantom_data: PhantomData<T>,
    border_mode: BorderMode,
    low_pass: Vec<T>,
    high_pass: Vec<T>,
    filter_length: usize,
}

impl<T: WaveletSample> WaveletNTaps<T>
where
    f64: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[T]) -> Self {
        Self {
            border_mode,
            filter_length: wavelet.len(),
            high_pass: low_pass_to_high(wavelet),
            low_pass: wavelet.to_vec(),
            phantom_data: PhantomData,
        }
    }
}

impl<T: WaveletSample> DwtForwardExecutor<T> for WaveletNTaps<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_forward(
        &self,
        input: &[T],
        approx: &mut [T],
        details: &mut [T],
    ) -> Result<(), OscletError> {
        let mut scratch = try_vec![T::default(); self.required_scratch_size(input.len())];
        self.execute_forward_with_scratch(input, approx, details, &mut scratch)
    }

    fn execute_forward_with_scratch(
        &self,
        input: &[T],
        approx: &mut [T],
        details: &mut [T],
        scratch: &mut [T],
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
            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let mut a = 0.0f64.as_();
                let mut d = 0.0f64.as_();
                let base = 2 * i;

                let input = padded_input.get_unchecked(base..base + self.filter_length);

                for ((&src, &g), &h) in input
                    .iter()
                    .zip(self.high_pass.iter())
                    .zip(self.low_pass.iter())
                {
                    a = fmla(h, src, a);
                    d = fmla(g, src, d);
                }

                *approx = a;
                *detail = d;
            }
        }
        Ok(())
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

impl<T: WaveletSample> DwtInverseExecutor<T> for WaveletNTaps<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_inverse(
        &self,
        approx: &[T],
        details: &[T],
        output: &mut [T],
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
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - filter_offset as isize;
                    let part =
                        output.get_unchecked_mut(k as usize..k as usize + self.filter_length);
                    for ((&wg, &wh), dst) in self
                        .high_pass
                        .iter()
                        .zip(self.low_pass.iter())
                        .zip(part.iter_mut())
                    {
                        *dst = fmla(wh, h, fmla(wg, g, *dst));
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

    fn idwt_size(&self, input_length: DwtSize) -> usize {
        idwt_length(input_length.approx_length, self.filter_length())
    }
}

impl<T: WaveletSample> IncompleteDwtExecutor<T> for WaveletNTaps<T>
where
    f64: AsPrimitive<T>,
{
    fn filter_length(&self) -> usize {
        self.filter_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DaubechiesFamily, WaveletFilterProvider};

    #[test]
    fn test_db6_odd() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db4 = WaveletNTaps::new(
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
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db4 = WaveletNTaps::new(
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
}
