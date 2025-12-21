/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
use crate::util::{dwt_length, idwt_length, low_pass_to_high_from_arr, twelve_taps_size_for_input};
use crate::{
    DwtForwardExecutor, DwtInverseExecutor, DwtSize, IncompleteDwtExecutor, WaveletSample,
};
use num_traits::AsPrimitive;
use std::marker::PhantomData;

pub(crate) struct Wavelet12Taps<T> {
    phantom_data: PhantomData<T>,
    border_mode: BorderMode,
    low_pass: [T; 12],
    high_pass: [T; 12],
}

impl<T: WaveletSample> Wavelet12Taps<T>
where
    f64: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[T; 12]) -> Self {
        Self {
            border_mode,
            low_pass: *wavelet,
            high_pass: low_pass_to_high_from_arr(wavelet),
            phantom_data: PhantomData,
        }
    }
}

impl<T: WaveletSample> DwtForwardExecutor<T> for Wavelet12Taps<T>
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
        let half = dwt_length(input.len(), 12);

        if input.len() < 12 {
            return Err(OscletError::MinFilterSize(input.len(), 12));
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

            let (front_approx, approx) = approx.split_at_mut(5);
            let (front_detail, details) = details.split_at_mut(5);

            for (i, (approx, detail)) in front_approx
                .iter_mut()
                .zip(front_detail.iter_mut())
                .enumerate()
            {
                let mut a = 0.0f64.as_();
                let mut d = 0.0f64.as_();
                let base = 2 * i as isize - 10;

                let x0 = interpolation.interpolate(input, base);
                let x1 = interpolation.interpolate(input, base + 1);
                let x2 = interpolation.interpolate(input, base + 2);
                let x3 = interpolation.interpolate(input, base + 3);
                let x4 = interpolation.interpolate(input, base + 4);
                let x5 = interpolation.interpolate(input, base + 5);
                let x6 = interpolation.interpolate(input, base + 6);
                let x7 = interpolation.interpolate(input, base + 7);
                let x8 = interpolation.interpolate(input, base + 8);
                let x9 = interpolation.interpolate(input, base + 9);
                let x10 = *input.get_unchecked((base + 10) as usize);
                let x11 = *input.get_unchecked((base + 11) as usize);

                a = fmla(self.low_pass[0], x0, a);
                d = fmla(self.high_pass[0], x0, d);

                a = fmla(self.low_pass[1], x1, a);
                d = fmla(self.high_pass[1], x1, d);

                a = fmla(self.low_pass[2], x2, a);
                d = fmla(self.high_pass[2], x2, d);

                a = fmla(self.low_pass[3], x3, a);
                d = fmla(self.high_pass[3], x3, d);

                a = fmla(self.low_pass[4], x4, a);
                d = fmla(self.high_pass[4], x4, d);

                a = fmla(self.low_pass[5], x5, a);
                d = fmla(self.high_pass[5], x5, d);

                a = fmla(self.low_pass[6], x6, a);
                d = fmla(self.high_pass[6], x6, d);

                a = fmla(self.low_pass[7], x7, a);
                d = fmla(self.high_pass[7], x7, d);

                a = fmla(self.low_pass[8], x8, a);
                d = fmla(self.high_pass[8], x8, d);

                a = fmla(self.low_pass[9], x9, a);
                d = fmla(self.high_pass[9], x9, d);

                a = fmla(self.low_pass[10], x10, a);
                d = fmla(self.high_pass[10], x10, d);

                a = fmla(self.low_pass[11], x11, a);
                d = fmla(self.high_pass[11], x11, d);

                *approx = a;
                *detail = d;
            }

            let (approx, approx_rem) =
                approx.split_at_mut(twelve_taps_size_for_input(input.len(), approx.len()));
            let (details, details_rem) =
                details.split_at_mut(twelve_taps_size_for_input(input.len(), details.len()));

            let base_start = approx.len();

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let mut a = 0.0f64.as_();
                let mut d = 0.0f64.as_();
                let base = 2 * i;

                let input = input.get_unchecked(base..);

                let x0 = *input.get_unchecked(0);
                let x1 = *input.get_unchecked(1);
                let x2 = *input.get_unchecked(2);
                let x3 = *input.get_unchecked(3);
                let x4 = *input.get_unchecked(4);
                let x5 = *input.get_unchecked(5);
                let x6 = *input.get_unchecked(6);
                let x7 = *input.get_unchecked(7);
                let x8 = *input.get_unchecked(8);
                let x9 = *input.get_unchecked(9);
                let x10 = *input.get_unchecked(10);
                let x11 = *input.get_unchecked(11);

                a = fmla(self.low_pass[0], x0, a);
                d = fmla(self.high_pass[0], x0, d);

                a = fmla(self.low_pass[1], x1, a);
                d = fmla(self.high_pass[1], x1, d);

                a = fmla(self.low_pass[2], x2, a);
                d = fmla(self.high_pass[2], x2, d);

                a = fmla(self.low_pass[3], x3, a);
                d = fmla(self.high_pass[3], x3, d);

                a = fmla(self.low_pass[4], x4, a);
                d = fmla(self.high_pass[4], x4, d);

                a = fmla(self.low_pass[5], x5, a);
                d = fmla(self.high_pass[5], x5, d);

                a = fmla(self.low_pass[6], x6, a);
                d = fmla(self.high_pass[6], x6, d);

                a = fmla(self.low_pass[7], x7, a);
                d = fmla(self.high_pass[7], x7, d);

                a = fmla(self.low_pass[8], x8, a);
                d = fmla(self.high_pass[8], x8, d);

                a = fmla(self.low_pass[9], x9, a);
                d = fmla(self.high_pass[9], x9, d);

                a = fmla(self.low_pass[10], x10, a);
                d = fmla(self.high_pass[10], x10, d);

                a = fmla(self.low_pass[11], x11, a);
                d = fmla(self.high_pass[11], x11, d);

                *approx = a;
                *detail = d;
            }

            for (i, (approx, detail)) in approx_rem
                .iter_mut()
                .zip(details_rem.iter_mut())
                .enumerate()
            {
                let mut a = 0.0f64.as_();
                let mut d = 0.0f64.as_();
                let base = 2 * (i + base_start);

                let x0 = *input.get_unchecked(base);
                let x1 = interpolation.interpolate(input, base as isize + 1);
                let x2 = interpolation.interpolate(input, base as isize + 2);
                let x3 = interpolation.interpolate(input, base as isize + 3);
                let x4 = interpolation.interpolate(input, base as isize + 4);
                let x5 = interpolation.interpolate(input, base as isize + 5);
                let x6 = interpolation.interpolate(input, base as isize + 6);
                let x7 = interpolation.interpolate(input, base as isize + 7);
                let x8 = interpolation.interpolate(input, base as isize + 8);
                let x9 = interpolation.interpolate(input, base as isize + 9);
                let x10 = interpolation.interpolate(input, base as isize + 10);
                let x11 = interpolation.interpolate(input, base as isize + 11);

                a = fmla(self.low_pass[0], x0, a);
                d = fmla(self.high_pass[0], x0, d);

                a = fmla(self.low_pass[1], x1, a);
                d = fmla(self.high_pass[1], x1, d);

                a = fmla(self.low_pass[2], x2, a);
                d = fmla(self.high_pass[2], x2, d);

                a = fmla(self.low_pass[3], x3, a);
                d = fmla(self.high_pass[3], x3, d);

                a = fmla(self.low_pass[4], x4, a);
                d = fmla(self.high_pass[4], x4, d);

                a = fmla(self.low_pass[5], x5, a);
                d = fmla(self.high_pass[5], x5, d);

                a = fmla(self.low_pass[6], x6, a);
                d = fmla(self.high_pass[6], x6, d);

                a = fmla(self.low_pass[7], x7, a);
                d = fmla(self.high_pass[7], x7, d);

                a = fmla(self.low_pass[8], x8, a);
                d = fmla(self.high_pass[8], x8, d);

                a = fmla(self.low_pass[9], x9, a);
                d = fmla(self.high_pass[9], x9, d);

                a = fmla(self.low_pass[10], x10, a);
                d = fmla(self.high_pass[10], x10, d);

                a = fmla(self.low_pass[11], x11, a);
                d = fmla(self.high_pass[11], x11, d);

                *approx = a;
                *detail = d;
            }
        }
        Ok(())
    }

    fn required_scratch_size(&self, _: usize) -> usize {
        0
    }

    fn dwt_size(&self, input_length: usize) -> DwtSize {
        DwtSize::new(dwt_length(input_length, 12))
    }
}

impl<T: WaveletSample> DwtInverseExecutor<T> for Wavelet12Taps<T>
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

        let rec_len = idwt_length(approx.len(), 12);

        if output.len() != rec_len {
            return Err(OscletError::OutputSizeIsNotValid(output.len(), rec_len));
        }

        const FILTER_OFFSET: usize = 10;
        const FILTER_LENGTH: usize = 12;

        unsafe {
            let safe_start = FILTER_OFFSET;
            // 2*x - off + len >= output.len()
            // x >= (output.len() + off - len)/2
            let mut safe_end = ((output.len() + FILTER_OFFSET).saturating_sub(FILTER_LENGTH)) / 2;

            if safe_start < safe_end {
                for i in 0..safe_start {
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    for j in 0..12 {
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

                for i in safe_start..safe_end {
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    let part = output.get_unchecked_mut(k as usize..);
                    for i in 0..12 {
                        *part.get_unchecked_mut(i) = fmla(
                            self.low_pass[i],
                            h,
                            fmla(self.high_pass[i], g, *part.get_unchecked(i)),
                        );
                    }
                }
            } else {
                safe_end = 0usize;
            }

            for i in safe_end..approx.len() {
                let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..12 {
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
        idwt_length(input_length.approx_length, 12)
    }
}

impl<T: WaveletSample> IncompleteDwtExecutor<T> for Wavelet12Taps<T>
where
    f64: AsPrimitive<T>,
{
    fn filter_length(&self) -> usize {
        12
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
        let db4 = Wavelet12Taps::new(
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
                (REFERENCE_APPROX[i] - x).abs() < 1e-5,
                "approx difference expected to be < 1e-5, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-5,
                "details difference expected to be < 1e-5, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 12)];
        db4.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-5,
                "reconstructed difference expected to be < 1e-5, but values were ref {}, derived {}",
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
        let db5 = Wavelet12Taps::new(
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
        db5.execute_forward(&input, &mut approx, &mut details)
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
                (REFERENCE_APPROX[i] - x).abs() < 1e-5,
                "approx difference expected to be < 1e-5, but values were ref {}, derived {}",
                REFERENCE_APPROX[i],
                x
            );
        });
        details.iter().enumerate().for_each(|(i, x)| {
            assert!(
                (REFERENCE_DETAILS[i] - x).abs() < 1e-5,
                "details difference expected to be < 1e-5, but values were ref {}, derived {}",
                REFERENCE_DETAILS[i],
                x
            );
        });

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 12)];
        db5.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-5,
                "reconstructed difference expected to be < 1e-5, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }

    #[test]
    fn test_db6_even_big() {
        let data_length = 86;
        let mut input = vec![0.; data_length];
        for i in 0..data_length {
            input[i] = i as f32 / data_length as f32;
        }
        let db6 = Wavelet12Taps::new(
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
        db6.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 12)];
        db6.execute_inverse(&approx, &details, &mut reconstructed)
            .unwrap();
        reconstructed.iter().take(input.len()).enumerate().for_each(|(i, x)| {
            assert!(
                (input[i] - x).abs() < 1e-5,
                "reconstructed difference expected to be < 1e-5, but values were ref {}, derived {}",
                input[i],
                x
            );
        });
    }
}
