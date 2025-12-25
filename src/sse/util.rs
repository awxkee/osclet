/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
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
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
#[target_feature(enable = "sse2")]
pub(crate) fn _mm_fma_pd(a: __m128d, b: __m128d, c: __m128d) -> __m128d {
    _mm_add_pd(_mm_mul_pd(a, b), c)
}

#[inline]
#[target_feature(enable = "sse2")]
pub(crate) fn _mm_fma_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    _mm_add_ps(_mm_mul_ps(a, b), c)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn _mm_load2_ps(a: *const f32) -> __m128 {
    unsafe { _mm_castsi128_ps(_mm_loadu_si64(a.cast())) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub(crate) fn _mm_store2_ps(a: *mut f32, x: __m128) {
    unsafe { _mm_storeu_si64(a.cast(), _mm_castps_si128(x)) }
}

#[inline]
#[target_feature(enable = "sse2")]
pub(crate) fn _mm_unpack2hi_ps(a: __m128, b: __m128) -> __m128 {
    _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(a, b)
}

#[inline]
#[target_feature(enable = "sse2")]
pub(crate) fn _mm_unpack2lo_ps(a: __m128, b: __m128) -> __m128 {
    _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(a), _mm_castps_pd(b)))
}

#[inline]
#[target_feature(enable = "sse2")]
pub(crate) fn _mm_swap_hilo(a: __m128, b: __m128) -> __m128 {
    _mm_shuffle_ps::<{ shuffle(1, 0, 3, 2) }>(a, b)
}
