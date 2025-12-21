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
use crate::{
    BorderMode, DwtForwardExecutor, DwtInverseExecutor, DwtSize, IncompleteDwtExecutor,
    WaveletSample,
};
use num_traits::AsPrimitive;
use std::marker::PhantomData;

#[allow(unused)]
pub(crate) struct Wavelet2Taps<T> {
    phantom_data: PhantomData<T>,
    border_mode: BorderMode,
    low_pass: [T; 2],
    high_pass: [T; 2],
}

impl<T: WaveletSample> Wavelet2Taps<T>
where
    f64: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[T; 2]) -> Self {
        Self {
            border_mode,
            low_pass: *wavelet,
            high_pass: low_pass_to_high_from_arr(wavelet),
            phantom_data: PhantomData,
        }
    }
}

impl<T: WaveletSample> DwtForwardExecutor<T> for Wavelet2Taps<T>
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
            let (approx, approx_rem) =
                approx.split_at_mut(two_taps_size_for_input(input.len(), approx.len()));
            let (details, details_rem) =
                details.split_at_mut(two_taps_size_for_input(input.len(), details.len()));

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let mut a = 0.0f64.as_();
                let mut d = 0.0f64.as_();
                let base = 2 * i;

                let input = input.get_unchecked(base..);

                let x0 = input.get_unchecked(0);
                let x1 = input.get_unchecked(1);

                a = fmla(self.low_pass[0], *x0, a);
                d = fmla(self.high_pass[0], *x0, d);

                a = fmla(self.low_pass[1], *x1, a);
                d = fmla(self.high_pass[1], *x1, d);

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

impl<T: WaveletSample> DwtInverseExecutor<T> for Wavelet2Taps<T>
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

                for i in safe_start..safe_end {
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    let part = output.get_unchecked_mut(k as usize..);
                    *part.get_unchecked_mut(0) = fmla(
                        self.low_pass[0],
                        h,
                        fmla(self.high_pass[0], g, *part.get_unchecked(0)),
                    );
                    *part.get_unchecked_mut(1) = fmla(
                        self.low_pass[1],
                        h,
                        fmla(self.high_pass[1], g, *part.get_unchecked(1)),
                    );
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

impl<T: WaveletSample> IncompleteDwtExecutor<T> for Wavelet2Taps<T>
where
    f64: AsPrimitive<T>,
{
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
        let db1 = Wavelet2Taps::new(
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
        let db1 = Wavelet2Taps::new(
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
}
