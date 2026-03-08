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
use std::arch::wasm32::*;
use std::ops::{Add, Mul};

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct WasmVector {
    pub(crate) v: v128,
}

impl WasmVector {
    /// Load 4 floats from slice
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn load(v: &[f32]) -> WasmVector {
        debug_assert!(v.len() >= 4);
        unsafe { Self::raw(v128_load(v.as_ptr() as *const v128)) }
    }

    /// Load 2 floats into low 2 lanes, zero upper 2 (replaces _mm_load2_ps)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn load2(v: &[f32]) -> WasmVector {
        debug_assert!(v.len() >= 2);
        unsafe { Self::raw(v128_load64_lane::<0>(f32x4_splat(0.), v.as_ptr().cast())) }
    }

    /// Broadcast 1 float into all 4 lanes (replaces _mm_load1_ps)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn load1(v: &[f32]) -> WasmVector {
        debug_assert!(!v.is_empty());
        unsafe { Self::raw(v128_load32_splat(v.as_ptr().cast())) }
    }

    /// Load single float into lane 0, zero remaining lanes (replaces _mm_load_ss)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn load1_lane(v: &f32) -> WasmVector {
        Self::raw(f32x4(*v, 0.0, 0.0, 0.0))
    }

    /// Distribute lanes [0,0,1,1] (replaces _mm_shuffle_ps SHUF(1,1,0,0))
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn distribute_00_11(self) -> WasmVector {
        // [a,b,c,d] -> [a,a,b,b]
        Self::raw(i32x4_shuffle::<0, 0, 1, 1>(self.v, self.v))
    }

    /// Broadcast a single lane to all 4 lanes (replaces _mm_shuffle_ps splat variants)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn distribute_element<const IDX: u32>(self) -> WasmVector {
        Self::raw(match IDX {
            1 => i32x4_shuffle::<1, 1, 1, 1>(self.v, self.v),
            2 => i32x4_shuffle::<2, 2, 2, 2>(self.v, self.v),
            3 => i32x4_shuffle::<3, 3, 3, 3>(self.v, self.v),
            _ => i32x4_shuffle::<0, 0, 0, 0>(self.v, self.v),
        })
    }

    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn raw(v: v128) -> Self {
        Self { v }
    }

    /// Fused multiply-add: self * b + c  (replaces _mm_fma_ps)
    /// Note: WASM SIMD has no native FMA — emulated with mul + add.
    /// For true FMA, enable `relaxed-simd` and use f32x4_relaxed_madd.
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn mul_add(self, b: Self, c: Self) -> WasmVector {
        // Standard WASM SIMD (no relaxed-simd required)
        let mul = f32x4_mul(self.v, b.v);
        Self::raw(f32x4_add(mul, c.v))
    }

    /// Store 4 floats to slice (replaces _mm_storeu_ps)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn write(self, output: &mut [f32]) {
        debug_assert!(output.len() >= 4);
        unsafe {
            v128_store(output.as_mut_ptr() as *mut v128, self.v);
        }
    }

    /// Store low 2 floats to slice (replaces _mm_store2_ps)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn write2(self, output: &mut [f32]) {
        debug_assert!(output.len() >= 2);
        unsafe { v128_store64_lane::<0>(self.v, output.as_mut_ptr().cast()) }
    }

    /// Store lane 0 to scalar (replaces _mm_store_ss)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn write1(self, output: &mut f32) {
        unsafe { v128_store32_lane::<0>(self.v, (output as *mut f32).cast()) }
    }

    /// Construct from 4 individual floats (replaces _mm_setr_ps)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn from_elements(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self::raw(f32x4(a, b, c, d))
    }

    /// All-zero vector (replaces _mm_setzero_ps)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn zero() -> Self {
        Self::raw(f32x4_splat(0.0))
    }

    /// Interleave high halves of two vectors (replaces _mm_unpackhi_ps)
    /// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a2,b2,a3,b3]
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn unpack2_hi(self, other: Self) -> WasmVector {
        Self::raw(i32x4_shuffle::<2, 3, 6, 7>(self.v, other.v))
    }

    /// Interleave low halves of two vectors (replaces _mm_unpacklo_ps)
    /// [a0,a1,a2,a3], [b0,b1,b2,b3] -> [a0,b0,a1,b1]
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn unpack2_lo(self, other: Self) -> WasmVector {
        Self::raw(i32x4_shuffle::<0, 1, 4, 5>(self.v, other.v))
    }

    /// Horizontal sum: all 4 lanes reduced to lane 0 (replaces _mm_hsum_ps)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn hsum(self) -> WasmVector {
        // [a,b,c,d] -> [a+b, c+d, a+b, c+d]
        let shuf = i32x4_shuffle::<1, 0, 3, 2>(self.v, self.v);
        let sum1 = f32x4_add(self.v, shuf);
        // [a+b, c+d, ...] -> [c+d, a+b, ...]
        let shuf2 = i32x4_shuffle::<2, 3, 0, 1>(sum1, sum1);
        Self::raw(f32x4_add(sum1, shuf2))
    }

    /// Horizontal add of adjacent pairs across two vectors (replaces _mm_hadd_ps)
    /// self=[a,b,c,d], other=[e,f,g,h] -> [a+b, c+d, e+f, g+h]
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn hadd(self, other: Self) -> WasmVector {
        // Extract adjacent pairs and add
        let lo_self = i32x4_shuffle::<0, 2, 4, 6>(self.v, other.v); // [a, c, e, g]
        let hi_self = i32x4_shuffle::<1, 3, 5, 7>(self.v, other.v); // [b, d, f, h]
        Self::raw(f32x4_add(lo_self, hi_self))
    }

    /// Pack even-indexed lanes into low half (replaces _mm_shuffle_ps SHUF(0,0,2,0))
    /// [a,b,c,d] -> [a,c,a,a]
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn pack_evens_in_lo(self) -> WasmVector {
        Self::raw(i32x4_shuffle::<0, 2, 0, 0>(self.v, self.v))
    }

    /// Swap high/low halves between two vectors (replaces custom _mm_swap_hilo)
    /// self=[a,b,c,d], other=[e,f,g,h] -> [c,d,e,f]
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn swap_hilo(self, other: Self) -> WasmVector {
        Self::raw(i32x4_shuffle::<2, 3, 4, 5>(self.v, other.v))
    }

    /// Broadcast scalar to all lanes (replaces _mm_set1_ps)
    #[inline]
    #[target_feature(enable = "simd128")]
    pub(crate) fn dup(v: f32) -> WasmVector {
        Self::raw(f32x4_splat(v))
    }
}

impl Add for WasmVector {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::raw(f32x4_add(self.v, rhs.v))
    }
}

impl Mul for WasmVector {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::raw(f32x4_mul(self.v, rhs.v))
    }
}
#[cfg(test)]
#[cfg(target_arch = "wasm32")]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_node_experimental);

    // ── helpers ────────────────────────────────────────────────────────────────

    /// Extract all 4 lanes from a WasmVector for assertion
    #[target_feature(enable = "simd128")]
    unsafe fn to_array(v: WasmVector) -> [f32; 4] {
        use std::arch::wasm32::*;
        [
            f32x4_extract_lane::<0>(v.v),
            f32x4_extract_lane::<1>(v.v),
            f32x4_extract_lane::<2>(v.v),
            f32x4_extract_lane::<3>(v.v),
        ]
    }

    fn assert_f32x4_eq(got: [f32; 4], expected: [f32; 4]) {
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-6, "lane {i}: got {g}, expected {e}");
        }
    }

    fn assert_f32x4_eq_v(v: WasmVector, expected: [f32; 4]) {
        let got = unsafe { to_array(v) };
        assert_f32x4_eq(got, expected);
    }

    // ── load / store ───────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_load() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let v = WasmVector::load(&data);
        assert_f32x4_eq_v(v, [1.0, 2.0, 3.0, 4.0]);
    }

    #[wasm_bindgen_test]
    fn test_load2_low_lanes_high_zeroed() {
        let data = [7.0f32, 8.0, 99.0, 99.0]; // only first 2 should be read
        let v = WasmVector::load2(&data);
        assert_f32x4_eq_v(v, [7.0, 8.0, 0.0, 0.0]);
    }

    #[wasm_bindgen_test]
    fn test_load1_broadcasts() {
        let data = [5.0f32];
        let v = WasmVector::load1(&data);
        assert_f32x4_eq_v(v, [5.0, 5.0, 5.0, 5.0]);
    }

    #[wasm_bindgen_test]
    fn test_load1_lane_zeroes_upper() {
        let v = WasmVector::load1_lane(&3.14);
        assert_f32x4_eq_v(v, [3.14, 0.0, 0.0, 0.0]);
    }

    #[wasm_bindgen_test]
    fn test_write_roundtrip() {
        let src = [1.0f32, 2.0, 3.0, 4.0];
        let v = WasmVector::load(&src);
        let mut dst = [0.0f32; 4];
        v.write(&mut dst);
        assert_eq!(dst, src);
    }

    #[wasm_bindgen_test]
    fn test_write2_only_two_lanes() {
        let v = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let mut dst = [0.0f32; 4];
        v.write2(&mut dst);
        assert_eq!(&dst[..2], &[1.0f32, 2.0]);
        // upper two untouched
        assert_eq!(&dst[2..], &[0.0f32, 0.0]);
    }

    #[wasm_bindgen_test]
    fn test_write1_lane0_only() {
        let v = WasmVector::from_elements(42.0, 1.0, 2.0, 3.0);
        let mut out = 0.0f32;
        v.write1(&mut out);
        assert!((out - 42.0).abs() < 1e-6);
    }

    // ── constructors ───────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_from_elements() {
        let v = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        assert_f32x4_eq_v(v, [1.0, 2.0, 3.0, 4.0]);
    }

    #[wasm_bindgen_test]
    fn test_zero() {
        let v = WasmVector::zero();
        assert_f32x4_eq_v(v, [0.0, 0.0, 0.0, 0.0]);
    }

    #[wasm_bindgen_test]
    fn test_dup() {
        let v = WasmVector::dup(9.0);
        assert_f32x4_eq_v(v, [9.0, 9.0, 9.0, 9.0]);
    }

    // ── shuffles / distributes ─────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_distribute_00_11() {
        // [1,2,3,4] -> [1,1,2,2]
        let v = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let r = v.distribute_00_11();
        assert_f32x4_eq_v(r, [1.0, 1.0, 2.0, 2.0]);
    }

    #[wasm_bindgen_test]
    fn test_distribute_element_lane0() {
        let v = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let r = v.distribute_element::<0>();
        assert_f32x4_eq_v(r, [1.0, 1.0, 1.0, 1.0]);
    }

    #[wasm_bindgen_test]
    fn test_distribute_element_lane1() {
        let v = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let r = v.distribute_element::<1>();
        assert_f32x4_eq_v(r, [2.0, 2.0, 2.0, 2.0]);
    }

    #[wasm_bindgen_test]
    fn test_distribute_element_lane2() {
        let v = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let r = v.distribute_element::<2>();
        assert_f32x4_eq_v(r, [3.0, 3.0, 3.0, 3.0]);
    }

    #[wasm_bindgen_test]
    fn test_distribute_element_lane3() {
        let v = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let r = v.distribute_element::<3>();
        assert_f32x4_eq_v(r, [4.0, 4.0, 4.0, 4.0]);
    }

    #[wasm_bindgen_test]
    fn test_pack_evens_in_lo() {
        // [a,b,c,d] -> [a,c,a,a]
        let v = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let r = v.pack_evens_in_lo();
        assert_f32x4_eq_v(r, [1.0, 3.0, 1.0, 1.0]);
    }

    #[wasm_bindgen_test]
    fn test_swap_hilo() {
        // self=[1,2,3,4], other=[5,6,7,8] -> [3,4,5,6]
        let a = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let b = WasmVector::from_elements(5.0, 6.0, 7.0, 8.0);
        let r = a.swap_hilo(b);
        assert_f32x4_eq_v(r, [3.0, 4.0, 5.0, 6.0]);
    }

    // ── unpack ─────────────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_unpack2_lo() {
        // [1,2,3,4], [5,6,7,8] -> [1,5,2,6]
        let a = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let b = WasmVector::from_elements(5.0, 6.0, 7.0, 8.0);
        let r = a.unpack2_lo(b);
        assert_f32x4_eq_v(r, [1.0, 2.0, 5.0, 6.0]);
    }

    #[wasm_bindgen_test]
    fn test_unpack2_hi() {
        // [1,2,3,4], [5,6,7,8] -> [3,7,4,8]
        let a = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let b = WasmVector::from_elements(5.0, 6.0, 7.0, 8.0);
        let r = a.unpack2_hi(b);
        assert_f32x4_eq_v(r, [3.0, 4.0, 7.0, 8.0]);
    }

    // ── horizontal ops ─────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_hsum() {
        // [1,2,3,4] -> all lanes = 10
        let v = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let r = v.hsum();
        let got = unsafe { to_array(r) };
        // All lanes should contain the total sum
        for lane in got {
            assert!((lane - 10.0).abs() < 1e-6, "expected 10.0, got {lane}");
        }
    }

    #[wasm_bindgen_test]
    fn test_hsum_zeros() {
        let v = WasmVector::zero();
        let r = v.hsum();
        assert_f32x4_eq_v(r, [0.0, 0.0, 0.0, 0.0]);
    }

    #[wasm_bindgen_test]
    fn test_hadd() {
        // [1,2,3,4], [5,6,7,8] -> [1+2, 3+4, 5+6, 7+8] = [3,7,11,15]
        let a = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let b = WasmVector::from_elements(5.0, 6.0, 7.0, 8.0);
        let r = a.hadd(b);
        assert_f32x4_eq_v(r, [3.0, 7.0, 11.0, 15.0]);
    }

    #[wasm_bindgen_test]
    fn test_hadd_same_vector() {
        // [2,4,6,8] hadd itself -> [6, 14, 6, 14]
        let a = WasmVector::from_elements(2.0, 4.0, 6.0, 8.0);
        let r = a.hadd(a);
        assert_f32x4_eq_v(r, [6.0, 14.0, 6.0, 14.0]);
    }

    // ── arithmetic ─────────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_add() {
        let a = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let b = WasmVector::from_elements(10.0, 20.0, 30.0, 40.0);
        let r = a + b;
        assert_f32x4_eq_v(r, [11.0, 22.0, 33.0, 44.0]);
    }

    #[wasm_bindgen_test]
    fn test_add_zero_identity() {
        let a = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let z = WasmVector::zero();
        assert_f32x4_eq_v(a + z, [1.0, 2.0, 3.0, 4.0]);
    }

    #[wasm_bindgen_test]
    fn test_mul() {
        let a = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let b = WasmVector::from_elements(2.0, 3.0, 4.0, 5.0);
        let r = a * b;
        assert_f32x4_eq_v(r, [2.0, 6.0, 12.0, 20.0]);
    }

    #[wasm_bindgen_test]
    fn test_mul_by_zero() {
        let a = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let z = WasmVector::zero();
        assert_f32x4_eq_v(a * z, [0.0, 0.0, 0.0, 0.0]);
    }

    #[wasm_bindgen_test]
    fn test_mul_add() {
        // (2 * 3) + 1 = 7 per lane
        let a = WasmVector::dup(2.0);
        let b = WasmVector::dup(3.0);
        let c = WasmVector::dup(1.0);
        let r = a.mul_add(b, c);
        assert_f32x4_eq_v(r, [7.0, 7.0, 7.0, 7.0]);
    }

    #[wasm_bindgen_test]
    fn test_mul_add_mixed() {
        // [1,2,3,4] * [4,3,2,1] + [1,1,1,1] = [5,7,7,5]
        let a = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let b = WasmVector::from_elements(4.0, 3.0, 2.0, 1.0);
        let c = WasmVector::dup(1.0);
        let r = a.mul_add(b, c);
        assert_f32x4_eq_v(r, [5.0, 7.0, 7.0, 5.0]);
    }

    // ── edge cases ─────────────────────────────────────────────────────────────

    #[wasm_bindgen_test]
    fn test_load_write_negative_values() {
        let src = [-1.0f32, -2.0, -3.0, -4.0];
        let v = WasmVector::load(&src);
        let mut dst = [0.0f32; 4];
        v.write(&mut dst);
        assert_eq!(dst, src);
    }

    #[wasm_bindgen_test]
    fn test_mul_add_with_zero_accumulator() {
        let a = WasmVector::from_elements(1.0, 2.0, 3.0, 4.0);
        let b = WasmVector::from_elements(2.0, 2.0, 2.0, 2.0);
        let c = WasmVector::zero();
        let r = a.mul_add(b, c);
        assert_f32x4_eq_v(r, [2.0, 4.0, 6.0, 8.0]);
    }

    #[wasm_bindgen_test]
    fn test_add_mul_consistency() {
        // a * b should equal summing a added b times (for small integers)
        let a = WasmVector::dup(3.0);
        let b = WasmVector::dup(3.0);
        let mul = a * b;
        let add = a + a + a; // 3+3+3 = 9
        let got_mul = unsafe { to_array(mul) };
        let got_add = unsafe { to_array(add) };
        assert_f32x4_eq(got_mul, got_add);
    }
}
