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

// wasm_vector_d.rs — WASM SIMD f64x2 implementation replacing x86 SSE2/4.1 __m128d

use std::arch::wasm32::*;
use std::ops::{Add, Mul};

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct WasmVectorD {
    pub(crate) v: v128,
}

impl WasmVectorD {
    /// Load 2 f64s from slice (replaces _mm_loadu_pd)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn load(v: &[f64]) -> Self {
        debug_assert!(v.len() >= 2);
        unsafe { Self::raw(v128_load(v.as_ptr() as *const v128)) }
    }

    /// Broadcast 1 f64 into both lanes (replaces _mm_load1_pd)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn load1(v: &[f64]) -> Self {
        debug_assert!(!v.is_empty());
        unsafe { Self::raw(v128_load64_splat(v.as_ptr().cast())) }
    }

    /// Load single f64 into lane 0, zero lane 1 (replaces _mm_load_sd)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn load1_ref(v: &f64) -> Self {
        Self::raw(f64x2(*v, 0.0))
    }

    /// Store 2 f64s to slice (replaces _mm_storeu_pd)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn write(self, output: &mut [f64]) {
        debug_assert!(output.len() >= 2);
        unsafe { v128_store(output.as_mut_ptr() as *mut v128, self.v) }
    }

    /// Store lane 0 to scalar (replaces _mm_store_sd)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn write1(self, output: &mut f64) {
        unsafe { v128_store64_lane::<0>(self.v, (output as *mut f64).cast()) }
    }

    /// Broadcast lane 0 to both lanes (replaces _mm_shuffle_pd::<0>)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn duplicate_lo(self) -> Self {
        Self::raw(i64x2_shuffle::<0, 0>(self.v, self.v))
    }

    /// Broadcast lane 1 to both lanes (replaces _mm_shuffle_pd::<0b11>)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn duplicate_hi(self) -> Self {
        Self::raw(i64x2_shuffle::<1, 1>(self.v, self.v))
    }

    /// Broadcast lane IDX to both lanes (replaces duplicate_element)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn duplicate_element<const IDX: u32>(self) -> Self {
        match IDX {
            1 => self.duplicate_hi(),
            _ => self.duplicate_lo(),
        }
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn raw(v: v128) -> Self {
        Self { v }
    }

    /// Horizontal sum: both lanes = lane0 + lane1 (replaces _mm_hsum_pd)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn hsum(self) -> Self {
        // [a, b] -> [a+b, a+b]
        let swapped = i64x2_shuffle::<1, 0>(self.v, self.v);
        Self::raw(f64x2_add(self.v, swapped))
    }

    /// Horizontal add of adjacent pairs (replaces _mm_hadd_pd)
    /// self=[a,b], other=[c,d] -> [a+b, c+d]
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn hadd(self, other: Self) -> Self {
        // lo = [a, c], hi = [b, d]
        let lo = i64x2_shuffle::<0, 2>(self.v, other.v);
        let hi = i64x2_shuffle::<1, 3>(self.v, other.v);
        Self::raw(f64x2_add(lo, hi))
    }

    /// Fused multiply-add: self * b + c (replaces _mm_fma_pd)
    /// WASM has no native f64 FMA in standard simd128 — emulated with mul+add.
    /// Enable relaxed-simd and use f64x2_relaxed_madd for true FMA semantics.
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn mul_add(self, b: Self, c: Self) -> Self {
        let mul = f64x2_mul(self.v, b.v);
        Self::raw(f64x2_add(mul, c.v))

        // Uncomment for relaxed-simd targets (Chrome 114+, Firefox 113+, wasmtime 9+):
        // Self::raw(f64x2_relaxed_madd(self.v, b.v, c.v))
    }

    /// Construct from 2 individual f64s (replaces _mm_setr_pd)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn from_elements(a: f64, b: f64) -> Self {
        Self::raw(f64x2(a, b))
    }

    /// Broadcast scalar to both lanes (replaces _mm_set1_pd)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn dup(a: f64) -> Self {
        Self::raw(f64x2_splat(a))
    }

    /// All-zero vector (replaces _mm_setzero_pd)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn zero() -> Self {
        Self::raw(f64x2_splat(0.0))
    }
}

impl Add for WasmVectorD {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::raw(f64x2_add(self.v, rhs.v))
    }
}

impl Mul for WasmVectorD {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::raw(f64x2_mul(self.v, rhs.v))
    }
}

// ── tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(target_arch = "wasm32")]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_node_experimental);

    // ── helpers ────────────────────────────────────────────────────────────────

    #[target_feature(enable = "simd128")]
    unsafe fn to_array(v: WasmVectorD) -> [f64; 2] {
        [f64x2_extract_lane::<0>(v.v), f64x2_extract_lane::<1>(v.v)]
    }

    fn assert_f64x2_eq(got: [f64; 2], expected: [f64; 2]) {
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-12, "lane {i}: got {g}, expected {e}");
        }
    }

    fn assert_v_eq(v: WasmVectorD, expected: [f64; 2]) {
        assert_f64x2_eq(unsafe { to_array(v) }, expected);
    }

    // ── load ───────────────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_load() {
        let data = [1.0f64, 2.0];
        let v = WasmVectorD::load(&data);
        assert_v_eq(v, [1.0, 2.0]);
    }

    #[wasm_bindgen_test]
    fn test_load_negative() {
        let data = [-3.5f64, -7.25];
        let v = WasmVectorD::load(&data);
        assert_v_eq(v, [-3.5, -7.25]);
    }

    #[wasm_bindgen_test]
    fn test_load1_broadcasts_both_lanes() {
        let data = [5.0f64, 99.0]; // only first element should be read
        let v = WasmVectorD::load1(&data);
        assert_v_eq(v, [5.0, 5.0]);
    }

    #[wasm_bindgen_test]
    fn test_load1_ref_zeroes_lane1() {
        let v = WasmVectorD::load1_ref(&3.14);
        assert_v_eq(v, [3.14, 0.0]);
    }

    // ── store ──────────────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_write_roundtrip() {
        let src = [1.0f64, 2.0];
        let v = WasmVectorD::load(&src);
        let mut dst = [0.0f64; 2];
        v.write(&mut dst);
        assert_eq!(dst, src);
    }

    #[wasm_bindgen_test]
    fn test_write1_only_lane0() {
        let v = WasmVectorD::from_elements(42.0, 99.0);
        let mut out = 0.0f64;
        v.write1(&mut out);
        assert!((out - 42.0).abs() < 1e-12, "expected 42.0, got {out}");
    }

    #[wasm_bindgen_test]
    fn test_write1_does_not_touch_adjacent_memory() {
        let v = WasmVectorD::from_elements(1.0, 2.0);
        let mut buf = [0.0f64; 2];
        v.write1(&mut buf[0]);
        assert!((buf[0] - 1.0).abs() < 1e-12);
        assert!((buf[1] - 0.0).abs() < 1e-12, "lane1 should be untouched");
    }

    // ── constructors ───────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_from_elements() {
        let v = WasmVectorD::from_elements(3.0, 7.0);
        assert_v_eq(v, [3.0, 7.0]);
    }

    #[wasm_bindgen_test]
    fn test_dup() {
        let v = WasmVectorD::dup(9.0);
        assert_v_eq(v, [9.0, 9.0]);
    }

    #[wasm_bindgen_test]
    fn test_zero() {
        let v = WasmVectorD::zero();
        assert_v_eq(v, [0.0, 0.0]);
    }

    // ── duplicate / shuffle ────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_duplicate_lo() {
        // [3, 7] -> [3, 3]
        let v = WasmVectorD::from_elements(3.0, 7.0);
        assert_v_eq(v.duplicate_lo(), [3.0, 3.0]);
    }

    #[wasm_bindgen_test]
    fn test_duplicate_hi() {
        // [3, 7] -> [7, 7]
        let v = WasmVectorD::from_elements(3.0, 7.0);
        assert_v_eq(v.duplicate_hi(), [7.0, 7.0]);
    }

    #[wasm_bindgen_test]
    fn test_duplicate_element_lane0() {
        let v = WasmVectorD::from_elements(3.0, 7.0);
        assert_v_eq(v.duplicate_element::<0>(), [3.0, 3.0]);
    }

    #[wasm_bindgen_test]
    fn test_duplicate_element_lane1() {
        let v = WasmVectorD::from_elements(3.0, 7.0);
        assert_v_eq(v.duplicate_element::<1>(), [7.0, 7.0]);
    }

    #[wasm_bindgen_test]
    fn test_duplicate_element_oob_defaults_to_lo() {
        // Any IDX other than 1 should behave like duplicate_lo
        let v = WasmVectorD::from_elements(3.0, 7.0);
        assert_v_eq(v.duplicate_element::<99>(), [3.0, 3.0]);
    }

    // ── horizontal ops ─────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_hsum() {
        // [3, 7] -> both lanes = 10
        let v = WasmVectorD::from_elements(3.0, 7.0);
        let r = v.hsum();
        assert_v_eq(r, [10.0, 10.0]);
    }

    #[wasm_bindgen_test]
    fn test_hsum_zeros() {
        let v = WasmVectorD::zero();
        assert_v_eq(v.hsum(), [0.0, 0.0]);
    }

    #[wasm_bindgen_test]
    fn test_hsum_negative() {
        let v = WasmVectorD::from_elements(-4.0, 4.0);
        assert_v_eq(v.hsum(), [0.0, 0.0]);
    }

    #[wasm_bindgen_test]
    fn test_hadd() {
        // [1,2], [3,4] -> [1+2, 3+4] = [3, 7]
        let a = WasmVectorD::from_elements(1.0, 2.0);
        let b = WasmVectorD::from_elements(3.0, 4.0);
        assert_v_eq(a.hadd(b), [3.0, 7.0]);
    }

    #[wasm_bindgen_test]
    fn test_hadd_same_vector() {
        // [5, 9] hadd itself -> [14, 14]
        let a = WasmVectorD::from_elements(5.0, 9.0);
        assert_v_eq(a.hadd(a), [14.0, 14.0]);
    }

    #[wasm_bindgen_test]
    fn test_hadd_with_zero() {
        let a = WasmVectorD::from_elements(3.0, 5.0);
        let z = WasmVectorD::zero();
        assert_v_eq(a.hadd(z), [8.0, 0.0]);
    }

    // ── arithmetic ─────────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_add() {
        let a = WasmVectorD::from_elements(1.0, 2.0);
        let b = WasmVectorD::from_elements(10.0, 20.0);
        assert_v_eq(a + b, [11.0, 22.0]);
    }

    #[wasm_bindgen_test]
    fn test_add_zero_identity() {
        let a = WasmVectorD::from_elements(1.0, 2.0);
        let z = WasmVectorD::zero();
        assert_v_eq(a + z, [1.0, 2.0]);
    }

    #[wasm_bindgen_test]
    fn test_add_negative() {
        let a = WasmVectorD::from_elements(5.0, -3.0);
        let b = WasmVectorD::from_elements(-5.0, 3.0);
        assert_v_eq(a + b, [0.0, 0.0]);
    }

    #[wasm_bindgen_test]
    fn test_mul() {
        let a = WasmVectorD::from_elements(3.0, 4.0);
        let b = WasmVectorD::from_elements(2.0, 0.5);
        assert_v_eq(a * b, [6.0, 2.0]);
    }

    #[wasm_bindgen_test]
    fn test_mul_by_zero() {
        let a = WasmVectorD::from_elements(99.0, 99.0);
        let z = WasmVectorD::zero();
        assert_v_eq(a * z, [0.0, 0.0]);
    }

    #[wasm_bindgen_test]
    fn test_mul_by_one() {
        let a = WasmVectorD::from_elements(3.0, 7.0);
        let one = WasmVectorD::dup(1.0);
        assert_v_eq(a * one, [3.0, 7.0]);
    }

    #[wasm_bindgen_test]
    fn test_mul_add() {
        // (2 * 3) + 1 = 7 per lane
        let a = WasmVectorD::dup(2.0);
        let b = WasmVectorD::dup(3.0);
        let c = WasmVectorD::dup(1.0);
        assert_v_eq(a.mul_add(b, c), [7.0, 7.0]);
    }

    #[wasm_bindgen_test]
    fn test_mul_add_mixed_lanes() {
        // [1,2] * [4,3] + [1,1] = [5, 7]
        let a = WasmVectorD::from_elements(1.0, 2.0);
        let b = WasmVectorD::from_elements(4.0, 3.0);
        let c = WasmVectorD::dup(1.0);
        assert_v_eq(a.mul_add(b, c), [5.0, 7.0]);
    }

    #[wasm_bindgen_test]
    fn test_mul_add_zero_accumulator() {
        let a = WasmVectorD::from_elements(3.0, 4.0);
        let b = WasmVectorD::from_elements(2.0, 2.0);
        let c = WasmVectorD::zero();
        assert_v_eq(a.mul_add(b, c), [6.0, 8.0]);
    }

    #[wasm_bindgen_test]
    fn test_mul_add_consistency_with_add_mul() {
        // mul_add(a, b, c) should equal (a * b) + c
        let a = WasmVectorD::from_elements(1.5, 2.5);
        let b = WasmVectorD::from_elements(4.0, 2.0);
        let c = WasmVectorD::from_elements(0.5, 0.5);
        let fma = unsafe { to_array(a.mul_add(b, c)) };
        let manual = unsafe { to_array((a * b) + c) };
        assert_f64x2_eq(fma, manual);
    }
}
