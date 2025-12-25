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
use crate::border_mode::{BorderInterpolation, BorderMode};
use crate::err::{OscletError, try_vec};
use crate::mla::fmla;
use crate::util::{dwt_length, idwt_length, low_pass_to_high_from_arr, sixth_taps_size_for_input};
use crate::{
    DwtForwardExecutor, DwtInverseExecutor, DwtSize, IncompleteDwtExecutor, WaveletSample,
};
use num_traits::AsPrimitive;
use std::marker::PhantomData;

pub(crate) struct Wavelet6Taps<T> {
    phantom_data: PhantomData<T>,
    border_mode: BorderMode,
    low_pass: [T; 6],
    high_pass: [T; 6],
}

impl<T: WaveletSample> Wavelet6Taps<T>
where
    f64: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(border_mode: BorderMode, wavelet: &[T; 6]) -> Self {
        Self {
            border_mode,
            low_pass: *wavelet,
            high_pass: low_pass_to_high_from_arr(wavelet),
            phantom_data: PhantomData,
        }
    }
}

impl<T: WaveletSample> DwtForwardExecutor<T> for Wavelet6Taps<T>
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
        let half = dwt_length(input.len(), 6);

        if input.len() < 6 {
            return Err(OscletError::MinFilterSize(input.len(), 6));
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

            let (front_approx, approx) = approx.split_at_mut(2);
            let (front_detail, details) = details.split_at_mut(2);

            for (i, (approx, detail)) in front_approx
                .iter_mut()
                .zip(front_detail.iter_mut())
                .enumerate()
            {
                let mut a = 0.0f64.as_();
                let mut d = 0.0f64.as_();
                let base = 2 * i as isize - 4;

                let x0 = interpolation.interpolate(input, base);
                let x1 = interpolation.interpolate(input, base + 1);
                let x2 = interpolation.interpolate(input, base + 2);
                let x3 = interpolation.interpolate(input, base + 3);
                let x4 = input.get_unchecked((base + 4) as usize);
                let x5 = input.get_unchecked((base + 5) as usize);

                a = fmla(self.low_pass[0], x0, a);
                d = fmla(self.high_pass[0], x0, d);

                a = fmla(self.low_pass[1], x1, a);
                d = fmla(self.high_pass[1], x1, d);

                a = fmla(self.low_pass[2], x2, a);
                d = fmla(self.high_pass[2], x2, d);

                a = fmla(self.low_pass[3], x3, a);
                d = fmla(self.high_pass[3], x3, d);

                a = fmla(self.low_pass[4], *x4, a);
                d = fmla(self.high_pass[4], *x4, d);

                a = fmla(self.low_pass[5], *x5, a);
                d = fmla(self.high_pass[5], *x5, d);

                *approx = a;
                *detail = d;
            }

            let (approx, approx_rem) =
                approx.split_at_mut(sixth_taps_size_for_input(input.len(), approx.len()));
            let (details, details_rem) =
                details.split_at_mut(sixth_taps_size_for_input(input.len(), details.len()));

            for (i, (approx, detail)) in approx.iter_mut().zip(details.iter_mut()).enumerate() {
                let mut a = 0.0f64.as_();
                let mut d = 0.0f64.as_();
                let base = 2 * i;

                let input = input.get_unchecked(base..);

                let x0 = input.get_unchecked(0);
                let x1 = input.get_unchecked(1);
                let x2 = input.get_unchecked(2);
                let x3 = input.get_unchecked(3);
                let x4 = input.get_unchecked(4);
                let x5 = input.get_unchecked(5);

                a = fmla(self.low_pass[0], *x0, a);
                d = fmla(self.high_pass[0], *x0, d);

                a = fmla(self.low_pass[1], *x1, a);
                d = fmla(self.high_pass[1], *x1, d);

                a = fmla(self.low_pass[2], *x2, a);
                d = fmla(self.high_pass[2], *x2, d);

                a = fmla(self.low_pass[3], *x3, a);
                d = fmla(self.high_pass[3], *x3, d);

                a = fmla(self.low_pass[4], *x4, a);
                d = fmla(self.high_pass[4], *x4, d);

                a = fmla(self.low_pass[5], *x5, a);
                d = fmla(self.high_pass[5], *x5, d);

                *approx = a;
                *detail = d;
            }

            let base_start = approx.len();

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
        DwtSize::new(dwt_length(input_length, self.filter_length()))
    }
}

impl<T: WaveletSample> DwtInverseExecutor<T> for Wavelet6Taps<T>
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

        let rec_len = idwt_length(approx.len(), 6);

        if output.len() != rec_len {
            return Err(OscletError::OutputSizeIsNotValid(output.len(), rec_len));
        }

        const FILTER_OFFSET: usize = 4;
        const FILTER_LENGTH: usize = 6;

        unsafe {
            let safe_start = FILTER_OFFSET;
            // 2*x - off + len >= output.len()
            // x >= (output.len() + off - len)/2
            let mut safe_end = ((output.len() + FILTER_OFFSET).saturating_sub(FILTER_LENGTH)) / 2;

            if safe_start < safe_end {
                for i in 0..safe_start {
                    let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                    let k = 2 * i as isize - FILTER_OFFSET as isize;
                    for j in 0..6 {
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
                    *part.get_unchecked_mut(2) = fmla(
                        self.low_pass[2],
                        h,
                        fmla(self.high_pass[2], g, *part.get_unchecked(2)),
                    );
                    *part.get_unchecked_mut(3) = fmla(
                        self.low_pass[3],
                        h,
                        fmla(self.high_pass[3], g, *part.get_unchecked(3)),
                    );
                    *part.get_unchecked_mut(4) = fmla(
                        self.low_pass[4],
                        h,
                        fmla(self.high_pass[4], g, *part.get_unchecked(4)),
                    );
                    *part.get_unchecked_mut(5) = fmla(
                        self.low_pass[5],
                        h,
                        fmla(self.high_pass[5], g, *part.get_unchecked(5)),
                    );
                }
            } else {
                safe_end = 0usize;
            }

            for i in safe_end..approx.len() {
                let (h, g) = (*approx.get_unchecked(i), *details.get_unchecked(i));
                let k = 2 * i as isize - FILTER_OFFSET as isize;
                for j in 0..6 {
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

impl<T: WaveletSample> IncompleteDwtExecutor<T> for Wavelet6Taps<T>
where
    f64: AsPrimitive<T>,
{
    fn filter_length(&self) -> usize {
        6
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coiflet::CoifletFamily;
    use crate::{DaubechiesFamily, WaveletFilterProvider};

    #[test]
    fn test_db3_odd() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3, 2.5,
        ];
        let db3 = Wavelet6Taps::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db3
                .get_wavelet()
                .as_ref()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 6);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db3.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 11] = [
            0.8483726, 2.5241373, 2.65038574, 5.04554797, 1.36113344, 1.0534151, 5.85968077,
            8.27594493, 2.09006931, 2.1647733, 1.88197732,
        ];
        const REFERENCE_DETAILS: [f64; 11] = [
            -1.11980127,
            0.29462366,
            -0.75732176,
            -0.61512612,
            -0.52980262,
            -3.42034085,
            1.36890244,
            -0.37564252,
            1.25365344,
            -0.05294689,
            1.08607739,
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 6)];
        db3.execute_inverse(&approx, &details, &mut reconstructed)
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
    fn test_db3_even() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db3 = Wavelet6Taps::new(
            BorderMode::Wrap,
            DaubechiesFamily::Db3
                .get_wavelet()
                .as_ref()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 6);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db3.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 10] = [
            2.25345752, 1.28973105, 2.65038574, 5.04554797, 1.36113344, 1.0534151, 5.85968077,
            8.27594493, 2.25345752, 1.28973105,
        ];
        const REFERENCE_DETAILS: [f64; 10] = [
            -0.28935438,
            0.16391309,
            -0.75732176,
            -0.61512612,
            -0.52980262,
            -3.42034085,
            1.36890244,
            -0.37564252,
            -0.28935438,
            0.16391309,
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 6)];
        db3.execute_inverse(&approx, &details, &mut reconstructed)
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
    fn test_coif1_even() {
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 0.0, 1.0, 2.4, 6.5, 2.4, 6.4, 5.2, 0.6, 0.5, 1.3,
        ];
        let db3 = Wavelet6Taps::new(
            BorderMode::Wrap,
            CoifletFamily::Coif1
                .get_wavelet()
                .as_ref()
                .try_into()
                .unwrap(),
        );
        let out_length = dwt_length(input.len(), 6);
        let mut approx = vec![0.0; out_length];
        let mut details = vec![0.0; out_length];
        db3.execute_forward(&input, &mut approx, &mut details)
            .unwrap();

        const REFERENCE_APPROX: [f64; 10] = [
            0.64709521, 1.74438159, 4.53911719, 3.20774595, 0.30097675, 4.61093707, 6.14348133,
            6.59556141, 0.64709521, 1.74438159,
        ];
        const REFERENCE_DETAILS: [f64; 10] = [
            -0.47031852,
            0.07106881,
            -1.37735608,
            0.23385359,
            0.47256556,
            -3.26886672,
            -2.2968896,
            2.18117025,
            -0.47031852,
            0.07106881,
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

        let mut reconstructed = vec![0.0; idwt_length(approx.len(), 6)];
        db3.execute_inverse(&approx, &details, &mut reconstructed)
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
