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
use crate::{
    Dwt, DwtExecutor, DwtForwardExecutor, DwtInverseExecutor, DwtSize, IncompleteDwtExecutor,
    MultiDwt,
};
use num_traits::{AsPrimitive, MulAdd};
use std::ops::{Add, Mul};
use std::sync::Arc;

pub(crate) struct CompletedDwtExecutor<T> {
    intercepted: Arc<dyn IncompleteDwtExecutor<T> + Send + Sync>,
}

impl<T> CompletedDwtExecutor<T> {
    pub(crate) fn new(intercepted: Arc<dyn IncompleteDwtExecutor<T> + Send + Sync>) -> Self {
        Self { intercepted }
    }
}

impl<T> DwtForwardExecutor<T> for CompletedDwtExecutor<T> {
    fn execute_forward(
        &self,
        input: &[T],
        approx: &mut [T],
        details: &mut [T],
    ) -> Result<(), OscletError> {
        self.intercepted.execute_forward(input, approx, details)
    }

    fn execute_forward_with_scratch(
        &self,
        input: &[T],
        approx: &mut [T],
        details: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), OscletError> {
        self.intercepted
            .execute_forward_with_scratch(input, approx, details, scratch)
    }

    fn required_scratch_size(&self, input_length: usize) -> usize {
        self.intercepted.required_scratch_size(input_length)
    }

    fn dwt_size(&self, input_length: usize) -> DwtSize {
        self.intercepted.dwt_size(input_length)
    }
}

impl<T> DwtInverseExecutor<T> for CompletedDwtExecutor<T> {
    fn execute_inverse(
        &self,
        approx: &[T],
        details: &[T],
        output: &mut [T],
    ) -> Result<(), OscletError> {
        self.intercepted.execute_inverse(approx, details, output)
    }

    fn idwt_size(&self, input_length: DwtSize) -> usize {
        self.intercepted.idwt_size(input_length)
    }
}

impl<T> IncompleteDwtExecutor<T> for CompletedDwtExecutor<T> {
    fn filter_length(&self) -> usize {
        self.intercepted.filter_length()
    }
}

impl<T: Copy + Default + MulAdd<T, Output = T> + Add<T, Output = T> + Mul<T, Output = T> + 'static>
    DwtExecutor<T> for CompletedDwtExecutor<T>
where
    f64: AsPrimitive<T>,
{
    fn dwt(&self, signal: &[T], level: usize) -> Result<Dwt<T>, OscletError> {
        if level == 0 || level == 1 {
            let dwt_size = self.intercepted.dwt_size(signal.len());
            let mut approx = try_vec![T::default(); dwt_size.approx_length];
            let mut details = try_vec![T::default(); dwt_size.details_length];

            self.intercepted
                .execute_forward(signal, &mut approx, &mut details)?;

            Ok(Dwt {
                approximations: approx,
                details,
            })
        } else {
            let mut current_signal = signal.to_vec();
            let mut approx = vec![];
            let mut details = vec![];

            let filter_length = self.intercepted.filter_length();

            for _ in 0..level {
                if filter_length > current_signal.len() {
                    return Err(OscletError::BufferWasTooSmallForLevel);
                }

                let dwt_size = self.intercepted.dwt_size(signal.len());

                approx = try_vec![T::default(); dwt_size.approx_length];
                details = try_vec![T::default(); dwt_size.details_length];

                // Forward DWT on current signal
                self.intercepted
                    .execute_forward(&current_signal, &mut approx, &mut details)?;

                // Next level uses only the approximation
                current_signal = approx.to_vec();
            }

            Ok(Dwt {
                approximations: approx,
                details,
            })
        }
    }

    fn multi_dwt(&self, signal: &[T], levels: usize) -> Result<MultiDwt<T>, OscletError> {
        if levels == 0 || levels == 1 {
            let dwt_size = self.intercepted.dwt_size(signal.len());
            let mut approx = try_vec![T::default(); dwt_size.approx_length];
            let mut details = try_vec![T::default(); dwt_size.details_length];

            self.intercepted
                .execute_forward(signal, &mut approx, &mut details)?;

            Ok(MultiDwt {
                levels: vec![Dwt {
                    approximations: approx,
                    details,
                }],
            })
        } else {
            let mut current_signal = signal.to_vec();
            let mut approx;
            let mut details;

            let filter_length = self.intercepted.filter_length();

            let mut levels_store = Vec::with_capacity(levels);

            for _ in 0..levels {
                if filter_length > signal.len() {
                    return Err(OscletError::BufferWasTooSmallForLevel);
                }

                let dwt_size = self.intercepted.dwt_size(signal.len());

                approx = try_vec![T::default(); dwt_size.approx_length];
                details = try_vec![T::default(); dwt_size.details_length];

                // Forward DWT on current signal
                self.intercepted
                    .execute_forward(&current_signal, &mut approx, &mut details)?;

                // Next level uses only the approximation
                current_signal = approx.to_vec();

                levels_store.push(Dwt {
                    approximations: approx,
                    details,
                });
            }

            Ok(MultiDwt {
                levels: levels_store,
            })
        }
    }

    fn idwt(&self, dwt: &Dwt<T>) -> Result<Vec<T>, OscletError> {
        if dwt.details.len() != dwt.approximations.len() {
            return Err(OscletError::ApproxDetailsNotMatches(
                dwt.approximations.len(),
                dwt.details.len(),
            ));
        }
        let output_length = self.intercepted.idwt_size(DwtSize {
            approx_length: dwt.approximations.len(),
            details_length: dwt.details.len(),
        });
        let mut output = try_vec![T::default(); output_length];

        self.intercepted
            .execute_inverse(&dwt.approximations, &dwt.details, &mut output)?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::completed::CompletedDwtExecutor;
    use crate::wavelet8taps::Wavelet8Taps;
    use crate::{BorderMode, DaubechiesFamily, DwtExecutor, WaveletFilterProvider};
    use std::sync::Arc;

    #[test]
    fn test_db8_even_big() {
        let data_length = 86;
        let mut input = vec![0.; data_length];
        for i in 0..data_length {
            input[i] = i as f32 / data_length as f32;
        }
        let db4 = CompletedDwtExecutor {
            intercepted: Arc::new(Wavelet8Taps::new(
                BorderMode::Wrap,
                DaubechiesFamily::Db4
                    .get_wavelet()
                    .as_ref()
                    .try_into()
                    .unwrap(),
            )),
        };
        let dwt = db4.dwt(&input, 1).unwrap();

        let reconstructed = db4.idwt(&dwt).unwrap();
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
