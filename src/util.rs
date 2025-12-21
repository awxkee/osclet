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
use crate::WaveletSample;
use crate::err::{OscletError, try_vec};
use num_traits::AsPrimitive;
use std::fmt::Debug;
use std::ops::Mul;

/// Computes the length of the **approximation/detail coefficients** after a single-level
/// discrete wavelet transform (DWT) on a 1D signal.
///
/// # Parameters
/// - `len`: Length of the input signal.
/// - `filter_length`: Length of the wavelet filter (number of taps).
///
/// # Returns
/// The number of coefficients in the resulting approximation or detail vector.
#[inline]
pub fn dwt_length(len: usize, filter_length: usize) -> usize {
    (len + filter_length - 1) / 2
}

/// Computes the length of the **reconstructed signal** from approximation coefficients
/// during the inverse discrete wavelet transform (IDWT).
///
/// # Parameters
/// - `approx_length`: Length of the approximation coefficients (after DWT).
/// - `filter_length`: Length of the wavelet filter (number of taps).
#[inline]
pub fn idwt_length(approx_length: usize, filter_length: usize) -> usize {
    2 * approx_length - (filter_length - 2)
}

pub(crate) fn low_pass_to_high<T: WaveletSample>(pass: &[T]) -> Vec<T>
where
    f64: AsPrimitive<T>,
{
    let low_pass_length = pass.len();
    let mut g = vec![T::default(); pass.len()];

    for (n, dst) in g.iter_mut().enumerate() {
        *dst = if n % 2 == 0 {
            1.0f64.as_()
        } else {
            (-1.0f64).as_()
        } * pass[low_pass_length - 1 - n];
    }
    g
}

pub(crate) fn low_pass_to_high_from_arr<
    T: Copy + 'static + Debug + Default + Mul<T, Output = T>,
    const N: usize,
>(
    pass: &[T; N],
) -> [T; N]
where
    f64: AsPrimitive<T>,
{
    let mut g = [T::default(); N];

    for (n, dst) in g.iter_mut().enumerate() {
        *dst = if n % 2 == 0 {
            1.0f64.as_()
        } else {
            (-1.0f64).as_()
        } * pass[N - 1 - n];
    }
    g
}

pub(crate) struct Wavelet<T> {
    pub(crate) dec_lo: Vec<T>,
    pub(crate) dec_hi: Vec<T>,
    #[allow(unused)]
    pub(crate) rec_lo: Vec<T>,
    #[allow(unused)]
    pub(crate) rec_hi: Vec<T>,
}

pub(crate) fn fill_wavelet<T: WaveletSample>(src: &[T]) -> Result<Wavelet<T>, OscletError> {
    let rec_len = src.len();
    let dec_len = src.len();

    let mut w_rec_lo = try_vec![T::default(); rec_len];
    let mut w_rec_hi = try_vec![T::default(); rec_len];
    let mut w_dec_lo = try_vec![T::default(); dec_len];
    let mut w_dec_hi = try_vec![T::default(); dec_len];

    for (i, (((rec_lo, rec_hi), dec_lo), dec_hi)) in w_rec_lo
        .iter_mut()
        .zip(w_rec_hi.iter_mut())
        .zip(w_dec_lo.iter_mut())
        .zip(w_dec_hi.iter_mut())
        .enumerate()
    {
        let rev_i = dec_len - 1 - i;

        *rec_lo = src[i];
        *dec_lo = src[rev_i];

        *rec_hi = if i % 2 == 1 { -src[rev_i] } else { src[rev_i] };
        *dec_hi = if rev_i % 2 == 1 { -src[i] } else { src[i] };
    }

    Ok(Wavelet {
        dec_lo: w_dec_lo,
        dec_hi: w_dec_hi,
        rec_hi: w_rec_hi,
        rec_lo: w_rec_lo,
    })
}

#[inline]
pub(crate) fn two_taps_size_for_input(input: usize, size: usize) -> usize {
    if !input.is_multiple_of(2) {
        size - 1
    } else {
        size
    }
}

#[inline]
pub(crate) fn four_taps_size_for_input(input: usize, size: usize) -> usize {
    if !input.is_multiple_of(2) {
        size - 3
    } else {
        size - 2
    }
}

#[inline]
pub(crate) fn sixth_taps_size_for_input(input: usize, size: usize) -> usize {
    const FILTER_SIZE: usize = 6;

    let whole_pad_size = (2 * size + FILTER_SIZE - 2) - input;
    let left_pad = whole_pad_size / 2;
    let right_pad = whole_pad_size - left_pad;
    size - right_pad
}

#[inline]
pub(crate) fn eight_taps_size_for_input(input: usize, size: usize) -> usize {
    const FILTER_SIZE: usize = 8;

    let whole_size = (2 * size + FILTER_SIZE - 2) - input;
    let left_pad = whole_size / 2;
    let right_pad = whole_size - left_pad;
    size - right_pad
}

#[inline]
pub(crate) fn ten_taps_size_for_input(input: usize, size: usize) -> usize {
    const FILTER_SIZE: usize = 10;

    let whole_size = (2 * size + FILTER_SIZE - 2) - input;
    let left_pad = whole_size / 2;
    let right_pad = whole_size - left_pad;
    size - right_pad
}

#[inline]
pub(crate) fn twelve_taps_size_for_input(input: usize, size: usize) -> usize {
    const FILTER_SIZE: usize = 12;

    let whole_size = (2 * size + FILTER_SIZE - 2) - input;
    let left_pad = whole_size / 2;
    let right_pad = whole_size - left_pad;

    size - right_pad
}

#[inline]
pub(crate) fn sixteen_taps_size_for_input(input: usize, size: usize) -> usize {
    const FILTER_SIZE: usize = 16;

    let whole_size = (2 * size + FILTER_SIZE - 2) - input;
    let left_pad = whole_size / 2;
    let right_pad = whole_size - left_pad;

    size - right_pad
}
