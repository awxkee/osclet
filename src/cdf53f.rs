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
use crate::err::{OscletError, try_vec};
use crate::{
    Dwt, DwtExecutor, DwtForwardExecutor, DwtInverseExecutor, IncompleteDwtExecutor, MultiDwt,
};
use num_traits::{AsPrimitive, MulAdd};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

#[derive(Default)]
pub(crate) struct Cdf53<T> {
    phantom0: PhantomData<T>,
}

impl<
    T: Copy
        + 'static
        + MulAdd<T, Output = T>
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Default
        + Sub<T, Output = T>,
> DwtForwardExecutor<T> for Cdf53<T>
where
    f64: AsPrimitive<T>,
{
    /// Perform the forward CDF 5/3 DWT (lifting scheme) on a 1D signal.
    /// Splits the signal into approximation (low-pass) and detail (high-pass) coefficients.
    fn execute_forward(
        &self,
        input: &[T],
        approx: &mut [T],
        details: &mut [T],
    ) -> Result<(), OscletError> {
        let n = input.len();
        if n < 3 {
            return Err(OscletError::MinFilterSize(n, 3));
        }
        if approx.len() != n.div_ceil(2) {
            return Err(OscletError::ApproxSizeNotMatches(approx.len(), n));
        }
        if details.len() != n / 2 {
            return Err(OscletError::ApproxSizeNotMatches(details.len(), n));
        }

        let r = (input[0] + input[1]) * 0.25f64.as_();
        details[0] = input[0] - r;

        // Split into even (approx) and odd (detail)
        // Predict step: d[i] = odd - floor((even_left + even_right)/2)
        let signal_n = &input[1..];
        let signal_n2 = &input[2..];

        for (((dst, &s_previous), &s_current), &s_next) in details
            .iter_mut()
            .zip(input.iter().step_by(2))
            .zip(signal_n.iter().step_by(2))
            .zip(signal_n2.iter().step_by(2))
        {
            let left = s_previous;
            let right = s_next;
            let predicted = (left + right) * 0.5f64.as_();
            *dst = s_current - predicted;
        }

        // Handle last odd index if n is even (boundary)
        if n.is_multiple_of(2) {
            let i = n - 1;
            let left = input[i - 1];
            let predicted = (left + left) * 0.5f64.as_(); // right = left at boundary
            *details.last_mut().unwrap() = input[i] - predicted;
        }

        for (dst, src) in approx.iter_mut().zip(input.iter().step_by(2)) {
            *dst = *src;
        }

        // Update step: s[i] = even + floor((d_left + d_right + 2)/4)
        let d_prod = (details[0] + details[0]) * 0.25f64.as_();
        approx[0] = approx[0] + d_prod;

        let details_next = &details[1..];
        let approx_len = approx.len();
        let approx_k = &mut approx[1..approx_len - 1];

        for ((dst, detail), detail_next) in approx_k
            .iter_mut()
            .zip(details.iter())
            .zip(details_next.iter())
        {
            let d_left = *detail;
            let d_right = *detail_next;
            let update = (d_left + d_right) * 0.25f64.as_();
            *dst = *dst + update;
        }

        if approx.len() > 1 {
            let i = approx.len() - 1;
            let d_left = details[i - 1];
            let d_right = *details.last().unwrap();
            let update = (d_left + d_right) * 0.25f64.as_();
            approx[i] = approx[i] + update;
        }
        Ok(())
    }
}

impl<
    T: Copy
        + 'static
        + MulAdd<T, Output = T>
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Default
        + Sub<T, Output = T>,
> DwtInverseExecutor<T> for Cdf53<T>
where
    f64: AsPrimitive<T>,
{
    /// Perform the inverse CDF 5/3 DWT (lifting scheme) to reconstruct the original signal.
    fn execute_inverse(
        &self,
        approx: &[T],
        details: &[T],
        output: &mut [T],
    ) -> Result<(), OscletError> {
        let n = approx.len() + details.len();
        if n != output.len() {
            return Err(OscletError::OutputSizeIsTooSmall(output.len(), n));
        }
        if n < 3 {
            return Err(OscletError::MinFilterSize(n, 3));
        }

        // Inverse update step: s[i] = even - floor((d_left + d_right + 2)/4)
        let mut approx_inv = approx.to_vec();
        approx_inv[0] = approx_inv[0] - (details[0] + details[0]) * 0.25f64.as_();
        let approx_k = &mut approx_inv[1..];
        let detail_next = &details[1..];

        let mut iters = 1usize;

        for ((dst, detail), detail_next) in approx_k
            .iter_mut()
            .zip(details.iter())
            .zip(detail_next.iter())
        {
            let d_left = *detail;
            let d_right = *detail_next;
            let update: T = (d_left + d_right) * 0.25f64.as_();
            *dst = *dst - update;
            iters += 1;
        }

        if iters < approx.len() {
            let i = approx.len() - 1;
            let d_left = details[i - 1];
            let d_right = if i < details.len() {
                details[i]
            } else {
                details[details.len() - 1]
            };
            let update = (d_left + d_right) * 0.25f64.as_();
            approx_inv[i] = approx_inv[i] - update;
        }

        // Inverse predict step: odd = detail + floor((even_left + even_right)/2)
        let approx_next = &approx_inv[1..];
        let mut odd_values: Vec<T> = try_vec![T::default(); details.len()];

        for (((dst, &current), &next), &detail) in odd_values
            .iter_mut()
            .zip(approx_inv.iter())
            .zip(approx_next.iter())
            .zip(details.iter())
        {
            let left = current;
            let right = next;
            let predicted = (left + right) * 0.5f64.as_();
            *dst = detail + predicted;
        }

        if !details.is_empty() {
            let i = details.len() - 1;
            let left = approx_inv[i];
            let right = if i + 1 < approx_inv.len() {
                approx_inv[i + 1]
            } else {
                left
            };
            let predicted = (left + right) * 0.5f64.as_();
            odd_values[i] = details[i] + predicted;
        }

        // Interleave even and odd
        for ((dst, &even), &odd) in output
            .chunks_exact_mut(2)
            .zip(approx_inv.iter())
            .zip(odd_values.iter())
        {
            dst[0] = even;
            dst[1] = odd;
        }
        let chunks = output.chunks_exact_mut(2).into_remainder();
        if let Some(dst) = chunks.get_mut(0) {
            // Only one even value left at the end
            *dst = *approx_inv.last().unwrap();
        }

        Ok(())
    }
}

impl<
    T: Copy
        + 'static
        + MulAdd<T, Output = T>
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Default
        + Sub<T, Output = T>
        + Send
        + Sync,
> IncompleteDwtExecutor<T> for Cdf53<T>
where
    f64: AsPrimitive<T>,
{
    fn filter_length(&self) -> usize {
        5
    }
}

macro_rules! define_dwt_cdf_float {
    ($for_clazz: ident, $min_size: expr) => {
        impl<
            T: Copy
                + 'static
                + MulAdd<T, Output = T>
                + Add<T, Output = T>
                + Mul<T, Output = T>
                + Default
                + Sub<T, Output = T>
                + Send
                + Sync,
        > DwtExecutor<T> for $for_clazz<T>
        where
            f64: AsPrimitive<T>,
        {
            fn dwt(&self, signal: &[T], level: usize) -> Result<Dwt<T>, OscletError> {
                if signal.len() < $min_size {
                    return Err(OscletError::BufferWasTooSmallForLevel);
                }
                if level == 0 || level == 1 {
                    let mut approx = try_vec![T::default(); signal.len().div_ceil(2)];
                    let mut details = try_vec![T::default(); signal.len() / 2];

                    self.execute_forward(signal, &mut approx, &mut details)?;

                    Ok(Dwt {
                        approximations: approx,
                        details,
                    })
                } else {
                    let mut current_signal = signal.to_vec();
                    let mut approx = vec![];
                    let mut details = vec![];

                    for _ in 0..level {
                        if current_signal.len() < $min_size {
                            return Err(OscletError::BufferWasTooSmallForLevel);
                        }

                        approx = try_vec![T::default(); current_signal.len().div_ceil(2)];
                        details = try_vec![T::default(); current_signal.len() / 2];

                        // Forward DWT on current signal
                        self.execute_forward(&current_signal, &mut approx, &mut details)?;

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
                if signal.len() < $min_size {
                    return Err(OscletError::BufferWasTooSmallForLevel);
                }
                if levels == 0 || levels == 1 {
                    let mut approx = try_vec![T::default(); signal.len().div_ceil(2)];
                    let mut details = try_vec![T::default(); signal.len() / 2];

                    self.execute_forward(signal, &mut approx, &mut details)?;

                    Ok(MultiDwt {
                        levels: vec![Dwt {
                            approximations: approx,
                            details,
                        }],
                    })
                } else {
                    let mut current_signal = signal.to_vec();
                    let mut approx = vec![];
                    let mut details = vec![];

                    let mut levels_store = Vec::with_capacity(levels);

                    for _ in 0..levels {
                        if current_signal.len() < $min_size {
                            return Err(OscletError::BufferWasTooSmallForLevel);
                        }

                        approx = try_vec![T::default(); current_signal.len().div_ceil(2)];
                        details = try_vec![T::default(); current_signal.len() / 2];

                        // Forward DWT on current signal
                        self.execute_forward(&current_signal, &mut approx, &mut details)?;

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
                let mut output = try_vec![T::default(); dwt.details.len() + dwt.approximations.len()];
                self.execute_inverse(&dwt.approximations, &dwt.details, &mut output)?;
                Ok(output)
            }
        }
    };
}

pub(crate) use define_dwt_cdf_float;

define_dwt_cdf_float!(Cdf53, 3);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdf53() {
        let i16_cdf53 = Cdf53::<f32> {
            phantom0: Default::default(),
        };
        let o_signal = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 52, 63, 13, 255, 63, 42, 32, 12, 52, 54, 23, 125, 23,
            255, 43, 23, 123, 54, 34, 255, 255, 23, 255, 32, 13, 15, 65, 23, 5, 7, 7, 3, 9,
        ]
        .iter()
        .map(|&x| x as f32)
        .collect::<Vec<_>>();

        let mut approx: Vec<f32> = vec![0.; (o_signal.len() + 1) / 2];
        let mut details: Vec<f32> = vec![0.; o_signal.len() / 2];

        let mut restored = vec![0.; o_signal.len()];

        i16_cdf53
            .execute_forward(&o_signal, &mut approx, &mut details)
            .unwrap();

        i16_cdf53
            .execute_inverse(&approx, &details, &mut restored)
            .unwrap();
        o_signal.iter().zip(restored.iter()).enumerate().for_each(|(idx, (o, re))| {
            assert!((o - re).abs() < 1e-3, "Reconstruction difference should be less than 1e-3, but it's not for original o {o}, restored {re} at idx {idx}");
        });
    }

    #[test]
    fn test_cdf53_1() {
        let i16_cdf53 = Cdf53::<f32> {
            phantom0: Default::default(),
        };
        let o_signal = vec![
            10i16, 20, 30, 40, 50, 60, 70, 80, 52, 63, 13, 255, 63, 42, 32, 12, 52, 54, 23, 125,
            23, 255, 43, 23, 123, 54, 34, 255, 255, 23, 255, 32, 13, 15, 65, 23, 5, 7, 7, 3, 9, 1,
        ]
        .iter()
        .map(|&x| x as f32)
        .collect::<Vec<_>>();

        let mut approx: Vec<f32> = vec![0.; (o_signal.len() + 1) / 2];
        let mut details: Vec<f32> = vec![0.; o_signal.len() / 2];

        let mut restored = vec![0.; o_signal.len()];

        i16_cdf53
            .execute_forward(&o_signal, &mut approx, &mut details)
            .unwrap();

        i16_cdf53
            .execute_inverse(&approx, &details, &mut restored)
            .unwrap();
        o_signal.iter().zip(restored.iter()).enumerate().for_each(|(idx, (o, re))| {
            assert!((o - re).abs() < 1e-3, "Reconstruction difference should be less than 1e-3, but it's not for original o {o}, restored {re} at idx {idx}");
        });
    }

    #[test]
    fn test_cdf53_2() {
        let i16_cdf53 = Cdf53::<f32> {
            phantom0: Default::default(),
        };
        let o_signal = vec![
            1, 55, 523, 40, 8, 32, 45, 166, 52, 63, 13, 255, 63, 42, 32, 12, 52, 54, 23, 125, 23,
            255, 43, 23, 123, 54, 34, 255, 255, 23, 255, 32, 13, 15, 65, 23, 5, 7, 7, 3, 9, 1,
        ]
        .iter()
        .map(|&x| x as f32)
        .collect::<Vec<_>>();

        let mut approx: Vec<f32> = vec![0.; (o_signal.len() + 1) / 2];
        let mut details: Vec<f32> = vec![0.; o_signal.len() / 2];

        let mut restored = vec![0.; o_signal.len()];

        i16_cdf53
            .execute_forward(&o_signal, &mut approx, &mut details)
            .unwrap();

        i16_cdf53
            .execute_inverse(&approx, &details, &mut restored)
            .unwrap();
        o_signal.iter().zip(restored.iter()).enumerate().for_each(|(idx, (o, re))| {
            assert!((o - re).abs() < 1e-3, "Reconstruction difference should be less than 1e-3, but it's not for original o {o}, restored {re} at idx {idx}");
        });
    }
}
