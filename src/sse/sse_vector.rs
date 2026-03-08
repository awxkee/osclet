/*
 * // Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use crate::sse::util::{
    _mm_fma_ps, _mm_hsum_ps, _mm_load2_ps, _mm_store2_ps, _mm_swap_hilo, _mm_unpack2hi_ps,
    _mm_unpack2lo_ps, shuffle,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{Add, Mul};

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct SseVector {
    pub(crate) v: __m128,
}

impl SseVector {
    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn load(v: &[f32]) -> SseVector {
        unsafe { Self::raw(_mm_loadu_ps(v.as_ptr())) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn load2(v: &[f32]) -> SseVector {
        Self::raw(_mm_load2_ps(v.as_ptr()))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn load1(v: &[f32]) -> SseVector {
        unsafe { Self::raw(_mm_load1_ps(v.as_ptr())) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn load1_lane(v: &f32) -> SseVector {
        unsafe { Self::raw(_mm_load_ss(v)) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn distribute_00_11(self) -> SseVector {
        Self::raw(_mm_shuffle_ps::<{ shuffle(1, 1, 0, 0) }>(self.v, self.v))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn distribute_element<const IDX: u32>(self) -> SseVector {
        match IDX {
            1 => Self::raw(_mm_shuffle_ps::<{ shuffle(1, 1, 1, 1) }>(self.v, self.v)),
            2 => Self::raw(_mm_shuffle_ps::<{ shuffle(2, 2, 2, 2) }>(self.v, self.v)),
            3 => Self::raw(_mm_shuffle_ps::<{ shuffle(3, 3, 3, 3) }>(self.v, self.v)),
            _ => Self::raw(_mm_shuffle_ps::<{ shuffle(0, 0, 0, 0) }>(self.v, self.v)),
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub(crate) fn raw(v: __m128) -> Self {
        Self { v }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn mul_add(self, b: Self, c: Self) -> SseVector {
        Self::raw(_mm_fma_ps(self.v, b.v, c.v))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn write(self, output: &mut [f32]) {
        unsafe { _mm_storeu_ps(output.as_mut_ptr(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn write2(self, output: &mut [f32]) {
        _mm_store2_ps(output.as_mut_ptr(), self.v)
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn write1(self, output: &mut f32) {
        unsafe { _mm_store_ss(output, self.v) }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub(crate) fn from_elements(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self {
            v: _mm_setr_ps(a, b, c, d),
        }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn zero() -> Self {
        Self::raw(_mm_setzero_ps())
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn unpack2_hi(self, other: Self) -> SseVector {
        Self::raw(_mm_unpack2hi_ps(self.v, other.v))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn unpack2_lo(self, other: Self) -> SseVector {
        Self::raw(_mm_unpack2lo_ps(self.v, other.v))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn hsum(self) -> SseVector {
        Self::raw(_mm_hsum_ps(self.v))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn hadd(self, other: Self) -> SseVector {
        Self::raw(_mm_hadd_ps(self.v, other.v))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn pack_evens_in_lo(self) -> SseVector {
        Self::raw(_mm_shuffle_ps::<{ shuffle(0, 0, 2, 0) }>(self.v, self.v))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn swap_hilo(self, other: Self) -> SseVector {
        Self::raw(_mm_swap_hilo(self.v, other.v))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn dup(v: f32) -> SseVector {
        Self::raw(_mm_set1_ps(v))
    }
}

impl Add for SseVector {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { Self::raw(_mm_add_ps(self.v, rhs.v)) }
    }
}

impl Mul for SseVector {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { Self::raw(_mm_mul_ps(self.v, rhs.v)) }
    }
}
