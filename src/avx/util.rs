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
use num_traits::MulAdd;
use std::arch::x86_64::*;
use std::ops::{Add, Mul};

#[inline(always)]
pub(crate) const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline]
#[target_feature(enable = "sse3")]
pub(crate) fn _mm_hsum_ps(v: __m128) -> __m128 {
    let mut shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
    let mut sums = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
    sums = _mm_add_ss(sums, shuf);
    sums
}

#[inline]
#[target_feature(enable = "sse2")]
pub(crate) fn _mm_hsum_pd(v: __m128d) -> __m128d {
    let undef = _mm_undefined_ps();
    let shuftmp = _mm_movehl_ps(undef, _mm_castpd_ps(v));
    let shuf = _mm_castps_pd(shuftmp);
    _mm_add_sd(v, shuf)
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) fn _mm256_hsum_pd(v: __m256d) -> __m128d {
    _mm_hsum_pd(_mm_add_pd(
        _mm256_castpd256_pd128(v),
        _mm256_extractf128_pd::<1>(v),
    ))
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) fn _mm256_hsum_ps(v: __m256) -> __m128 {
    _mm_hsum_ps(_mm_add_ps(
        _mm256_castps256_ps128(v),
        _mm256_extractf128_ps::<1>(v),
    ))
}

// #[inline]
// #[target_feature(enable = "avx2")]
// pub(crate) fn _mm256_hpadd_ps(v: __m256) -> __m256 {
//     let wa0 = _mm256_hadd_ps(v, v);
//     _mm256_castpd_ps(_mm256_permute4x64_pd::<{ shuffle(3, 1, 2, 0) }>(
//         _mm256_castps_pd(wa0),
//     ))
// }

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn _mm256_hpadd2_ps(v0: __m256, v1: __m256) -> __m256 {
    let wa0 = _mm256_hadd_ps(v0, v1);
    _mm256_castpd_ps(_mm256_permute4x64_pd::<{ shuffle(3, 1, 2, 0) }>(
        _mm256_castps_pd(wa0),
    ))
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn _mm256_permute_ps64<const MASK: i32>(v: __m256) -> __m256 {
    _mm256_castpd_ps(_mm256_permute4x64_pd::<MASK>(_mm256_castps_pd(v)))
}

#[inline]
#[target_feature(enable = "avx2")]
pub(crate) fn _mm256_hpadd2_pd(v0: __m256d, v1: __m256d) -> __m256d {
    let wa0 = _mm256_hadd_pd(v0, v1);
    _mm256_permute4x64_pd::<{ shuffle(3, 1, 2, 0) }>(wa0)
}

#[inline]
pub(crate) fn afmla<T: Copy + Mul<T, Output = T> + Add<T, Output = T> + MulAdd<T, Output = T>>(
    a: T,
    b: T,
    c: T,
) -> T {
    MulAdd::mul_add(a, b, c)
}

#[inline]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn _mm_unpackhilo_ps64(a: __m128, b: __m128) -> __m128 {
    _mm_shuffle_ps::<{ shuffle(1, 0, 3, 2) }>(a, b)
}

#[inline]
#[target_feature(enable = "sse2")]
pub(crate) fn _mm_unpack2lo_ps(a: __m128, b: __m128) -> __m128 {
    _mm_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(a, b)
}

#[inline]
#[target_feature(enable = "sse2")]
pub(crate) fn _mm_unpack2hi_ps(a: __m128, b: __m128) -> __m128 {
    _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(a, b)
}
