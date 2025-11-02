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
use crate::cdf53f::define_dwt_cdf_float;
use crate::err::{OscletError, try_vec};
use crate::{
    Dwt, DwtExecutor, DwtForwardExecutor, DwtInverseExecutor, IncompleteDwtExecutor, MultiDwt,
};
use num_traits::{AsPrimitive, MulAdd};
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};

const ALPHA: f64 = -1.5861343420693648;
const BETA: f64 = -0.0529801185718856;
const GAMMA: f64 = 0.8829110755411875;
const DELTA: f64 = 0.4435068520511142;

#[derive(Default)]
pub(crate) struct Cdf97<T> {
    phantom0: PhantomData<T>,
}

fn dwt97_forward_update_even<
    T: Copy
        + 'static
        + MulAdd<T, Output = T>
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Default
        + Sub<T, Output = T>,
>(
    approx: &mut [T],
    details: &mut [T],
    c: T,
) where
    f64: AsPrimitive<T>,
{
    let u = (details[0] + details[0]) * c;
    approx[0] = approx[0] + u;

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
        let update = (d_left + d_right) * c;
        *dst = *dst + update;
    }

    if approx.len() > 1 {
        let i = approx.len() - 1;
        let d_left = details[i - 1];
        let d_right = *details.last().unwrap();
        let update = (d_left + d_right) * c;
        approx[i] = approx[i] + update;
    }
}

fn dwt97_forward_update_odd<
    T: Copy
        + 'static
        + MulAdd<T, Output = T>
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Default
        + Sub<T, Output = T>,
>(
    approx: &mut [T],
    details: &mut [T],
    c: T,
) where
    f64: AsPrimitive<T>,
{
    let approx_next = &approx[1..];
    for ((dst, &src_left), &src_right) in details
        .iter_mut()
        .zip(approx.iter())
        .zip(approx_next.iter())
    {
        let update = (src_left + src_right) * c;
        *dst = *dst + update;
    }

    if approx.len() == details.len() {
        let i = details.len();
        let src_left = approx[i - 1];
        // at boundary, mirror last sample
        let src_right = src_left;
        let update = (src_left + src_right) * c;
        details[i - 1] = details[i - 1] + update;
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
> DwtForwardExecutor<T> for Cdf97<T>
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
        if n < 4 {
            return Err(OscletError::MinFilterSize(n, 4));
        }
        if approx.len() != n.div_ceil(2) {
            return Err(OscletError::ApproxSizeNotMatches(approx.len(), n));
        }
        if details.len() != n / 2 {
            return Err(OscletError::ApproxSizeNotMatches(details.len(), n));
        }

        let signal_n = &input[1..];
        let signal_n2 = &input[2..];

        // First lifting step
        details[0] = input[0] + (input[0] + input[1]) * ALPHA.as_();

        for (((dst, &s_previous), &s_current), &s_next) in details
            .iter_mut()
            .zip(input.iter().step_by(2))
            .zip(signal_n.iter().step_by(2))
            .zip(signal_n2.iter().step_by(2))
        {
            let left = s_previous;
            let right = s_next;
            let predicted = (left + right) * ALPHA.as_();
            *dst = s_current + predicted;
        }

        // Handle last odd index if n is even (boundary)
        if n.is_multiple_of(2) {
            let i = n - 1;
            let left = input[i - 1];
            let predicted = (left + left) * ALPHA.as_();
            *details.last_mut().unwrap() = input[i] + predicted;
        }

        for (dst, src) in approx.iter_mut().zip(input.iter().step_by(2)) {
            *dst = *src;
        }

        // Update 1: even += beta * (odd_left + odd_right)
        dwt97_forward_update_even(approx, details, BETA.as_());

        // Predict 2: odd += gamma * (even_left + even_right)
        dwt97_forward_update_odd(approx, details, GAMMA.as_());

        // Update 2: even += delta * (odd_left + odd_right)
        dwt97_forward_update_even(approx, details, DELTA.as_());

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
> DwtInverseExecutor<T> for Cdf97<T>
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
        if n < 4 {
            return Err(OscletError::MinFilterSize(n, 4));
        }

        let mut approx_inv = approx.to_vec();
        let mut detail_inv = details.to_vec();

        // Inverse update 2: even -= delta * (odd_left + odd_right)
        dwt97_forward_update_even(&mut approx_inv, &mut detail_inv, (-DELTA).as_());

        // Inverse predict 2: odd -= gamma * (even_left + even_right)
        dwt97_forward_update_odd(&mut approx_inv, &mut detail_inv, (-GAMMA).as_());

        // Inverse update 1: even -= beta * (odd_left + odd_right) >> 14
        dwt97_forward_update_even(&mut approx_inv, &mut detail_inv, (-BETA).as_());

        // Inverse predict 1: odd -= alpha * (even_left + even_right) >> 14
        dwt97_forward_update_odd(&mut approx_inv, &mut detail_inv, (-ALPHA).as_());

        // Interleave approx and detail to reconstruct signal
        for ((dst, &src_even), &src_odd) in output
            .chunks_exact_mut(2)
            .zip(approx_inv.iter())
            .zip(detail_inv.iter())
        {
            dst[0] = src_even;
            dst[1] = src_odd;
        }
        if !output.len().is_multiple_of(2) {
            *output.last_mut().unwrap() = *approx_inv.last().unwrap();
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
> IncompleteDwtExecutor<T> for Cdf97<T>
where
    f64: AsPrimitive<T>,
{
    fn filter_length(&self) -> usize {
        8
    }
}

define_dwt_cdf_float!(Cdf97, 4);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdf97() {
        let m_cdf97 = Cdf97::<f32> {
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

        m_cdf97
            .execute_forward(&o_signal, &mut approx, &mut details)
            .unwrap();

        m_cdf97
            .execute_inverse(&approx, &details, &mut restored)
            .unwrap();
        o_signal.iter().zip(restored.iter()).enumerate().for_each(|(idx, (o, re))| {
            assert!((o - re).abs() < 1e-3, "Reconstruction difference should be less than 1e-3, but it's not for original o {o}, restored {re} at idx {idx}");
        });
    }

    #[test]
    fn test_cdf97_1() {
        let m_cdf97 = Cdf97::<f32> {
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

        m_cdf97
            .execute_forward(&o_signal, &mut approx, &mut details)
            .unwrap();

        m_cdf97
            .execute_inverse(&approx, &details, &mut restored)
            .unwrap();
        o_signal.iter().zip(restored.iter()).enumerate().for_each(|(idx, (o, re))| {
            assert!((o - re).abs() < 1e-3, "Reconstruction difference should be less than 1e-3, but it's not for original o {o}, restored {re} at idx {idx}");
        });
    }

    #[test]
    fn test_cdf97_2() {
        let m_cdf97 = Cdf97::<f32> {
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

        m_cdf97
            .execute_forward(&o_signal, &mut approx, &mut details)
            .unwrap();

        m_cdf97
            .execute_inverse(&approx, &details, &mut restored)
            .unwrap();
        o_signal.iter().zip(restored.iter()).enumerate().for_each(|(idx, (o, re))| {
            assert!((o - re).abs() < 1e-3, "Reconstruction difference should be less than 1e-3, but it's not for original o {o}, restored {re} at idx {idx}");
        });
    }
}
