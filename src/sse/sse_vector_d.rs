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
use crate::sse::util::{_mm_fma_pd, _mm_hsum_pd};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::{Add, Mul};

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct SseVectorD {
    pub(crate) v: __m128d,
}

impl SseVectorD {
    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn load(v: &[f64]) -> Self {
        unsafe { Self::raw(_mm_loadu_pd(v.as_ptr())) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn load1(v: &[f64]) -> Self {
        unsafe { Self::raw(_mm_load1_pd(v.as_ptr())) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn load1_ref(v: &f64) -> Self {
        unsafe { Self::raw(_mm_load_sd(v)) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn write(self, output: &mut [f64]) {
        unsafe { _mm_storeu_pd(output.as_mut_ptr(), self.v) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn write1(self, output: &mut f64) {
        unsafe { _mm_store_sd(output, self.v) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn duplicate_lo(self) -> Self {
        Self::raw(_mm_shuffle_pd::<0>(self.v, self.v))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn duplicate_element<const IDX: u32>(self) -> Self {
        match IDX {
            1 => Self::duplicate_hi(self),
            _ => Self::duplicate_lo(self),
        }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn duplicate_hi(self) -> Self {
        Self::raw(_mm_shuffle_pd::<0b11>(self.v, self.v))
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub(crate) fn raw(v: __m128d) -> Self {
        Self { v }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn hsum(self) -> Self {
        Self::raw(_mm_hsum_pd(self.v))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn hadd(self, other: Self) -> Self {
        Self::raw(_mm_hadd_pd(self.v, other.v))
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn mul_add(self, b: Self, c: Self) -> Self {
        Self::raw(_mm_fma_pd(self.v, b.v, c.v))
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub(crate) fn from_elements(a: f64, b: f64) -> Self {
        Self {
            v: _mm_setr_pd(a, b),
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    pub(crate) fn dup(a: f64) -> Self {
        Self { v: _mm_set1_pd(a) }
    }

    #[inline]
    #[target_feature(enable = "sse4.1")]
    pub(crate) fn zero() -> Self {
        Self::raw(_mm_setzero_pd())
    }
}

impl Add for SseVectorD {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        unsafe { Self::raw(_mm_add_pd(self.v, rhs.v)) }
    }
}

impl Mul for SseVectorD {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        unsafe { Self::raw(_mm_mul_pd(self.v, rhs.v)) }
    }
}
