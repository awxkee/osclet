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
use crate::cdf53i::define_integer_cdf;
use crate::err::{OscletError, try_vec};
use crate::{
    Dwt, DwtExecutor, DwtForwardExecutor, DwtInverseExecutor, IncompleteDwtExecutor, MultiDwt,
};
use num_traits::{AsPrimitive, WrappingAdd, WrappingSub};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Shr, Sub, SubAssign};

#[derive(Default)]
pub(crate) struct AvxCdf53Integer<T, V> {
    phantom0: PhantomData<T>,
    phantom1: PhantomData<V>,
}

impl<
    T: Copy + AsPrimitive<V> + 'static + AddAssign + WrappingAdd<Output = T>,
    V: Copy
        + 'static
        + Add<V, Output = V>
        + Shr<u32, Output = V>
        + Copy
        + Sub<V, Output = V>
        + AsPrimitive<T>,
> DwtForwardExecutor<T> for AvxCdf53Integer<T, V>
where
    i32: AsPrimitive<V>,
{
    /// Perform the forward CDF 5/3 DWT (lifting scheme) on a 1D signal.
    /// Splits the signal into approximation (low-pass) and detail (high-pass) coefficients.
    fn execute_forward(
        &self,
        input: &[T],
        approx: &mut [T],
        details: &mut [T],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_forward_impl(input, approx, details) }
    }
}

impl<
    T: Copy + AsPrimitive<V> + 'static + AddAssign + WrappingAdd<Output = T>,
    V: Copy
        + 'static
        + Add<V, Output = V>
        + Shr<u32, Output = V>
        + Copy
        + Sub<V, Output = V>
        + AsPrimitive<T>,
> AvxCdf53Integer<T, V>
where
    i32: AsPrimitive<V>,
{
    /// Perform the forward CDF 5/3 DWT (lifting scheme) on a 1D signal.
    /// Splits the signal into approximation (low-pass) and detail (high-pass) coefficients.
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_forward_impl(
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

        // Split into even (approx) and odd (detail)
        // Predict step: d[i] = odd - floor((even_left + even_right)/2)
        let signal_n = &input[1..];
        let signal_n2 = &input[2..];

        let r: V = (input[0].as_() + input[1].as_()) >> 1;
        details[0] = (input[0].as_() - r).as_();

        for (((dst, &s_previous), &s_current), &s_next) in details
            .iter_mut()
            .zip(input.iter().step_by(2))
            .zip(signal_n.iter().step_by(2))
            .zip(signal_n2.iter().step_by(2))
        {
            let left: V = s_previous.as_();
            let right: V = s_next.as_();
            let predicted = (left + right) >> 1;
            *dst = (s_current.as_() - predicted).as_();
        }

        // Handle last odd index if n is even (boundary)
        if n.is_multiple_of(2) {
            let i = n - 1;
            let left: V = input[i - 1].as_();
            let predicted = (left + left) >> 1; // right = left at boundary
            *details.last_mut().unwrap() = (input[i].as_() - predicted).as_();
        }

        for (dst, src) in approx.iter_mut().zip(input.iter().step_by(2)) {
            *dst = *src;
        }

        // Update step: s[i] = even + floor((d_left + d_right + 2)/4)
        let d_prod = (details[0].as_() + details[0].as_() + 2.as_()) >> 2;
        approx[0] = approx[0].wrapping_add(&d_prod.as_());

        let details_next = &details[1..];
        let approx_len = approx.len();
        let approx_k = &mut approx[1..approx_len - 1];

        for ((dst, detail), detail_next) in approx_k
            .iter_mut()
            .zip(details.iter())
            .zip(details_next.iter())
        {
            let d_left = detail.as_();
            let d_right = detail_next.as_();
            let update = ((d_left + d_right + 2.as_()) >> 2).as_();
            *dst = dst.wrapping_add(&update);
        }

        if approx.len() > 1 {
            let i = approx.len() - 1;
            let d_left = details[i - 1].as_();
            let d_right = details.last().unwrap().as_();
            let update = ((d_left + d_right + 2.as_()) >> 2).as_();
            approx[i] = approx[i].wrapping_add(&update);
        }
        Ok(())
    }
}

impl<
    T: Copy + AsPrimitive<V> + 'static + SubAssign + Default + WrappingSub<Output = T>,
    V: Copy
        + 'static
        + Add<V, Output = V>
        + Shr<u32, Output = V>
        + Copy
        + Sub<V, Output = V>
        + AsPrimitive<T>,
> DwtInverseExecutor<T> for AvxCdf53Integer<T, V>
where
    i32: AsPrimitive<V> + AsPrimitive<T>,
{
    /// Perform the inverse CDF 5/3 DWT (lifting scheme) to reconstruct the original signal.
    fn execute_inverse(
        &self,
        approx: &[T],
        details: &[T],
        output: &mut [T],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_inverse_impl(approx, details, output) }
    }
}

impl<
    T: Copy + AsPrimitive<V> + 'static + SubAssign + Default + WrappingSub<Output = T>,
    V: Copy
        + 'static
        + Add<V, Output = V>
        + Shr<u32, Output = V>
        + Copy
        + Sub<V, Output = V>
        + AsPrimitive<T>,
> AvxCdf53Integer<T, V>
where
    i32: AsPrimitive<V> + AsPrimitive<T>,
{
    /// Perform the inverse CDF 5/3 DWT (lifting scheme) to reconstruct the original signal.
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_inverse_impl(
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

        let r: V = (details[0].as_() + details[0].as_() + 2.as_()) >> 2;
        approx_inv[0] = approx_inv[0].wrapping_sub(&r.as_());

        let approx_k = &mut approx_inv[1..];
        let detail_next = &details[1..];

        let mut iters = 1usize;

        for ((dst, detail), detail_next) in approx_k
            .iter_mut()
            .zip(details.iter())
            .zip(detail_next.iter())
        {
            let d_left = detail.as_();
            let d_right = detail_next.as_();
            let update: T = ((d_left + d_right + 2i32.as_()) >> 2).as_();
            *dst = dst.wrapping_sub(&update);
            iters += 1;
        }

        if iters < approx.len() {
            let i = approx.len() - 1;
            let d_left = details[i - 1].as_();
            let d_right = if i < details.len() {
                details[i].as_()
            } else {
                details[details.len() - 1].as_()
            };
            let update = ((d_left + d_right + 2i32.as_()) >> 2).as_();
            approx_inv[i] = approx_inv[i].wrapping_sub(&update);
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
            let left = current.as_();
            let right = next.as_();
            let predicted = (left + right) >> 1;
            *dst = (detail.as_() + predicted).as_();
        }

        if !details.is_empty() {
            let i = details.len() - 1;
            let left = approx_inv[i].as_();
            let right = if i + 1 < approx_inv.len() {
                approx_inv[i + 1].as_()
            } else {
                left
            };
            let predicted = (left + right) >> 1;
            odd_values[i] = (details[i].as_() + predicted).as_();
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
        + AsPrimitive<V>
        + 'static
        + SubAssign
        + Default
        + Send
        + Sync
        + AddAssign
        + WrappingSub<Output = T>
        + WrappingAdd<Output = T>,
    V: Copy
        + 'static
        + Add<V, Output = V>
        + Shr<u32, Output = V>
        + Copy
        + Sub<V, Output = V>
        + AsPrimitive<T>
        + Send
        + Sync,
> IncompleteDwtExecutor<T> for AvxCdf53Integer<T, V>
where
    i32: AsPrimitive<V> + AsPrimitive<T>,
{
    fn filter_length(&self) -> usize {
        6
    }
}

define_integer_cdf!(AvxCdf53Integer, 3);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cdf53() {
        let i16_cdf53 = AvxCdf53Integer::<i16, i32> {
            phantom0: Default::default(),
            phantom1: Default::default(),
        };
        let o_signal = vec![
            10i16, 20, 30, 40, 50, 60, 70, 80, 52, 63, 13, 255, 63, 42, 32, 12, 52, 54, 23, 125,
            23, 255, 43, 23, 123, 54, 34, 255, 255, 23, 255, 32, 13, 15, 65, 23, 5, 7, 7, 3, 9,
        ];

        let mut approx: Vec<i16> = vec![0; (o_signal.len() + 1) / 2];
        let mut details: Vec<i16> = vec![0; o_signal.len() / 2];

        let mut restored = vec![0i16; o_signal.len()];

        i16_cdf53
            .execute_forward(&o_signal, &mut approx, &mut details)
            .unwrap();

        i16_cdf53
            .execute_inverse(&approx, &details, &mut restored)
            .unwrap();
        assert_eq!(&o_signal, &restored);
    }

    #[test]
    fn test_cdf53_1() {
        let i16_cdf53 = AvxCdf53Integer::<i16, i32> {
            phantom0: Default::default(),
            phantom1: Default::default(),
        };
        let o_signal = vec![
            10i16, 20, 30, 40, 50, 60, 70, 80, 52, 63, 13, 255, 63, 42, 32, 12, 52, 54, 23, 125,
            23, 255, 43, 23, 123, 54, 34, 255, 255, 23, 255, 32, 13, 15, 65, 23, 5, 7, 7, 3, 9, 1,
        ];

        let mut approx: Vec<i16> = vec![0; (o_signal.len() + 1) / 2];
        let mut details: Vec<i16> = vec![0; o_signal.len() / 2];

        let mut restored = vec![0i16; o_signal.len()];

        i16_cdf53
            .execute_forward(&o_signal, &mut approx, &mut details)
            .unwrap();

        i16_cdf53
            .execute_inverse(&approx, &details, &mut restored)
            .unwrap();
        assert_eq!(&o_signal, &restored);
    }

    #[test]
    fn test_cdf53_2() {
        let i16_cdf53 = AvxCdf53Integer::<i16, i32> {
            phantom0: Default::default(),
            phantom1: Default::default(),
        };
        let o_signal = vec![
            1, 55, 523, 40, 8, 32, 45, 166, 52, 63, 13, 255, 63, 42, 32, 12, 52, 54, 23, 125, 23,
            255, 43, 23, 123, 54, 34, 255, 255, 23, 255, 32, 13, 15, 65, 23, 5, 7, 7, 3, 9, 1,
        ];

        let mut approx: Vec<i16> = vec![0; (o_signal.len() + 1) / 2];
        let mut details: Vec<i16> = vec![0; o_signal.len() / 2];

        let mut restored = vec![0i16; o_signal.len()];

        i16_cdf53
            .execute_forward(&o_signal, &mut approx, &mut details)
            .unwrap();

        i16_cdf53
            .execute_inverse(&approx, &details, &mut restored)
            .unwrap();
        assert_eq!(&o_signal, &restored);
    }
}
