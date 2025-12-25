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
use crate::BorderMode;
use crate::avx::util::{_mm_hsum_ps, _mm256_hsum_ps, shuffle};
use crate::border_mode::BorderInterpolation;
use crate::convolve1d::{Convolve1d, ConvolvePaddings};
use crate::err::OscletError;
use crate::filter_padding::write_arena_1d;
use std::arch::x86_64::*;

pub(crate) struct AvxConvolution1dF32 {
    pub(crate) border_mode: BorderMode,
}

impl AvxConvolution1dF32 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn convolve_2taps(
        &self,
        arena: &[f32],
        output: &mut [f32],
        kernel: &[f32; 2],
        filter_offset: isize,
    ) {
        unsafe {
            const FILTER_LENGTH: usize = 2;
            let paddings = ConvolvePaddings::from_filter(FILTER_LENGTH, filter_offset);
            let padding_right = paddings.padding_right.min(arena.len());
            let padding_left = paddings.padding_left.min(arena.len());
            let offset = paddings.padding_left as isize;

            let interpolation = BorderInterpolation::new(self.border_mode, 0, arena.len() as isize);

            let coeffs = _mm_setr_ps(kernel[0], kernel[1], 0., 0.);

            let mut x = 0usize;

            while x < padding_left {
                let src0 = interpolation.interpolate(arena, x as isize - offset);
                let src1 = interpolation.interpolate(arena, x as isize - offset + 1);

                let vals = _mm_setr_ps(src0, src1, 0., 0.);
                let a = _mm_mul_ps(vals, coeffs);
                let h = _mm_hsum_ps(a);

                _mm_store_ss(output.get_unchecked_mut(x), h);
                x += 1;
            }

            let c0 = _mm_permute_ps::<{ shuffle(0, 0, 0, 0) }>(coeffs);
            let c1 = _mm_permute_ps::<{ shuffle(1, 1, 1, 1) }>(coeffs);

            let c0 = _mm256_setr_m128(c0, c0);
            let c1 = _mm256_setr_m128(c1, c1);

            let max_safe_end = arena.len().saturating_sub(padding_right);

            while x + 16 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k0 = _mm256_mul_ps(_mm256_loadu_ps(shifted_src.as_ptr()), c0);
                let mut k1 =
                    _mm256_mul_ps(_mm256_loadu_ps(shifted_src.get_unchecked(8..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr) => {
                        k0 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                            k0,
                        );
                        k1 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(shifted_src.get_unchecked($i + 8..).as_ptr()),
                            $c,
                            k1,
                        );
                    };
                }

                step!(1, c1);

                _mm256_storeu_ps(output.get_unchecked_mut(x..).as_mut_ptr(), k0);
                _mm256_storeu_ps(output.get_unchecked_mut(x + 8..).as_mut_ptr(), k1);
                x += 16;
            }

            while x + 4 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k = _mm_mul_ps(
                    _mm_loadu_ps(shifted_src.as_ptr()),
                    _mm256_castps256_ps128(c0),
                );

                macro_rules! step {
                    ($i: expr, $c: expr) => {
                        k = _mm_fmadd_ps(
                            _mm_loadu_ps(shifted_src.get_unchecked($i..).as_ptr()),
                            _mm256_castps256_ps128($c),
                            k,
                        );
                    };
                }

                step!(1, c1);

                _mm_storeu_ps(output.get_unchecked_mut(x..).as_mut_ptr(), k);
                x += 4;
            }

            while x < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let q0 = _mm_castsi128_ps(_mm_loadu_si64(shifted_src.as_ptr().cast()));
                let w0 = _mm_mul_ps(q0, coeffs);

                let f = _mm_hsum_ps(w0);
                _mm_store_ss(output.get_unchecked_mut(x), f);
                x += 1;
            }

            while x < output.len() {
                let src0 = interpolation.interpolate(arena, x as isize - offset);
                let src1 = interpolation.interpolate(arena, x as isize - offset + 1);

                let vals = _mm_setr_ps(src0, src1, 0., 0.);
                let a = _mm_mul_ps(vals, coeffs);
                let h = _mm_hsum_ps(a);

                _mm_store_ss(output.get_unchecked_mut(x), h);
                x += 1;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn convolve_4taps(
        &self,
        arena: &[f32],
        output: &mut [f32],
        kernel: &[f32; 4],
        filter_offset: isize,
    ) {
        unsafe {
            const FILTER_LENGTH: usize = 4;
            let paddings = ConvolvePaddings::from_filter(FILTER_LENGTH, filter_offset);
            let padding_right = paddings.padding_right.min(arena.len());
            let padding_left = paddings.padding_left.min(arena.len());
            let offset = paddings.padding_left as isize;

            let interpolation = BorderInterpolation::new(self.border_mode, 0, arena.len() as isize);

            let coeffs = _mm_loadu_ps(kernel.as_ptr().cast());

            let mut x = 0usize;

            while x < padding_left {
                let src0 = interpolation.interpolate(arena, x as isize - offset);
                let src1 = interpolation.interpolate(arena, x as isize - offset + 1);
                let src2 = interpolation.interpolate(arena, x as isize - offset + 2);
                let src3 = interpolation.interpolate(arena, x as isize - offset + 3);

                let vals = _mm_setr_ps(src0, src1, src2, src3);
                let a = _mm_mul_ps(vals, coeffs);
                let h = _mm_hsum_ps(a);

                _mm_store_ss(output.get_unchecked_mut(x), h);
                x += 1;
            }

            let c0 = _mm_permute_ps::<{ shuffle(0, 0, 0, 0) }>(coeffs);
            let c1 = _mm_permute_ps::<{ shuffle(1, 1, 1, 1) }>(coeffs);
            let c2 = _mm_permute_ps::<{ shuffle(2, 2, 2, 2) }>(coeffs);
            let c3 = _mm_permute_ps::<{ shuffle(3, 3, 3, 3) }>(coeffs);

            let c0 = _mm256_setr_m128(c0, c0);
            let c1 = _mm256_setr_m128(c1, c1);
            let c2 = _mm256_setr_m128(c2, c2);
            let c3 = _mm256_setr_m128(c3, c3);

            let max_safe_end = arena.len().saturating_sub(padding_right);

            while x + 16 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k0 = _mm256_mul_ps(_mm256_loadu_ps(shifted_src.as_ptr()), c0);
                let mut k1 =
                    _mm256_mul_ps(_mm256_loadu_ps(shifted_src.get_unchecked(8..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr) => {
                        k0 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                            k0,
                        );
                        k1 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(shifted_src.get_unchecked($i + 8..).as_ptr()),
                            $c,
                            k1,
                        );
                    };
                }

                step!(1, c1);
                step!(2, c2);
                step!(3, c3);

                _mm256_storeu_ps(output.get_unchecked_mut(x..).as_mut_ptr(), k0);
                _mm256_storeu_ps(output.get_unchecked_mut(x + 8..).as_mut_ptr(), k1);
                x += 16;
            }

            while x + 4 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k = _mm_mul_ps(
                    _mm_loadu_ps(shifted_src.as_ptr()),
                    _mm256_castps256_ps128(c0),
                );

                macro_rules! step {
                    ($i: expr, $c: expr) => {
                        k = _mm_fmadd_ps(
                            _mm_loadu_ps(shifted_src.get_unchecked($i..).as_ptr()),
                            _mm256_castps256_ps128($c),
                            k,
                        );
                    };
                }

                step!(1, c1);
                step!(2, c2);
                step!(3, c3);

                _mm_storeu_ps(output.get_unchecked_mut(x..).as_mut_ptr(), k);
                x += 4;
            }

            while x < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let q0 = _mm_loadu_ps(shifted_src.as_ptr());
                let w0 = _mm_mul_ps(q0, coeffs);

                let f = _mm_hsum_ps(w0);
                _mm_store_ss(output.get_unchecked_mut(x), f);
                x += 1;
            }

            while x < output.len() {
                let src0 = interpolation.interpolate(arena, x as isize - offset);
                let src1 = interpolation.interpolate(arena, x as isize - offset + 1);
                let src2 = interpolation.interpolate(arena, x as isize - offset + 2);
                let src3 = interpolation.interpolate(arena, x as isize - offset + 3);

                let vals = _mm_setr_ps(src0, src1, src2, src3);
                let a = _mm_mul_ps(vals, coeffs);
                let h = _mm_hsum_ps(a);

                _mm_store_ss(output.get_unchecked_mut(x), h);
                x += 1;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn convolve_6taps(
        &self,
        arena: &[f32],
        output: &mut [f32],
        kernel: &[f32; 6],
        filter_offset: isize,
    ) {
        unsafe {
            const FILTER_LENGTH: usize = 6;
            let paddings = ConvolvePaddings::from_filter(FILTER_LENGTH, filter_offset);
            let padding_right = paddings.padding_right.min(arena.len());
            let padding_left = paddings.padding_left.min(arena.len());
            let offset = paddings.padding_left as isize;

            let interpolation = BorderInterpolation::new(self.border_mode, 0, arena.len() as isize);

            let full_coeffs = _mm256_setr_ps(
                kernel[0], kernel[1], kernel[2], kernel[3], kernel[4], kernel[5], 0., 0.,
            );

            let mut x = 0usize;

            while x < padding_left {
                let src0 = interpolation.interpolate(arena, x as isize - offset);
                let src1 = interpolation.interpolate(arena, x as isize - offset + 1);
                let src2 = interpolation.interpolate(arena, x as isize - offset + 2);
                let src3 = interpolation.interpolate(arena, x as isize - offset + 3);
                let src4 = interpolation.interpolate(arena, x as isize - offset + 4);
                let src5 = interpolation.interpolate(arena, x as isize - offset + 5);

                let vals = _mm256_setr_ps(src0, src1, src2, src3, src4, src5, 0., 0.);
                let a = _mm256_mul_ps(vals, full_coeffs);
                let h = _mm256_hsum_ps(a);

                _mm_store_ss(output.get_unchecked_mut(x), h);
                x += 1;
            }

            let max_safe_end = arena.len().saturating_sub(padding_right);

            let coeffs = _mm_loadu_ps(kernel.as_ptr().cast());
            let c0 = _mm_permute_ps::<{ shuffle(0, 0, 0, 0) }>(coeffs);
            let c1 = _mm_permute_ps::<{ shuffle(1, 1, 1, 1) }>(coeffs);
            let c2 = _mm_permute_ps::<{ shuffle(2, 2, 2, 2) }>(coeffs);
            let c3 = _mm_permute_ps::<{ shuffle(3, 3, 3, 3) }>(coeffs);

            let c0 = _mm256_setr_m128(c0, c0);
            let c1 = _mm256_setr_m128(c1, c1);
            let c2 = _mm256_setr_m128(c2, c2);
            let c3 = _mm256_setr_m128(c3, c3);

            let coeffs2 =
                _mm_castsi128_ps(_mm_loadu_si64(kernel.get_unchecked(4..).as_ptr().cast()));

            let c4 = _mm_permute_ps::<{ shuffle(0, 0, 0, 0) }>(coeffs2);
            let c5 = _mm_permute_ps::<{ shuffle(1, 1, 1, 1) }>(coeffs2);

            let c4 = _mm256_setr_m128(c4, c4);
            let c5 = _mm256_setr_m128(c5, c5);

            while x + 16 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k0 = _mm256_mul_ps(_mm256_loadu_ps(shifted_src.as_ptr()), c0);
                let mut k1 =
                    _mm256_mul_ps(_mm256_loadu_ps(shifted_src.get_unchecked(8..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr) => {
                        k0 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                            k0,
                        );
                        k1 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(shifted_src.get_unchecked($i + 8..).as_ptr()),
                            $c,
                            k1,
                        );
                    };
                }

                step!(1, c1);
                step!(2, c2);
                step!(3, c3);
                step!(4, c4);
                step!(5, c5);

                _mm256_storeu_ps(output.get_unchecked_mut(x..).as_mut_ptr(), k0);
                _mm256_storeu_ps(output.get_unchecked_mut(x + 8..).as_mut_ptr(), k1);
                x += 16;
            }

            while x + 4 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k = _mm_mul_ps(
                    _mm_loadu_ps(shifted_src.as_ptr()),
                    _mm256_castps256_ps128(c0),
                );

                macro_rules! step {
                    ($i: expr, $c: expr) => {
                        k = _mm_fmadd_ps(
                            _mm_loadu_ps(shifted_src.get_unchecked($i..).as_ptr()),
                            _mm256_castps256_ps128($c),
                            k,
                        );
                    };
                }

                step!(1, c1);
                step!(2, c2);
                step!(3, c3);
                step!(4, c4);
                step!(5, c5);

                _mm_storeu_ps(output.get_unchecked_mut(x..).as_mut_ptr(), k);
                x += 4;
            }

            while x < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let q0 = _mm_loadu_ps(shifted_src.as_ptr());
                let q1 = _mm_castsi128_ps(_mm_loadu_si64(
                    shifted_src.get_unchecked(4..).as_ptr().cast(),
                ));

                let b = _mm_mul_ps(q0, coeffs);
                let w0 = _mm_fmadd_ps(q1, coeffs2, b);

                let f = _mm_hsum_ps(w0);
                _mm_store_ss(output.get_unchecked_mut(x..).as_mut_ptr(), f);
                x += 1;
            }

            while x < output.len() {
                let src0 = interpolation.interpolate(arena, x as isize - offset);
                let src1 = interpolation.interpolate(arena, x as isize - offset + 1);
                let src2 = interpolation.interpolate(arena, x as isize - offset + 2);
                let src3 = interpolation.interpolate(arena, x as isize - offset + 3);
                let src4 = interpolation.interpolate(arena, x as isize - offset + 4);
                let src5 = interpolation.interpolate(arena, x as isize - offset + 5);

                let vals = _mm256_setr_ps(src0, src1, src2, src3, src4, src5, 0., 0.);
                let a = _mm256_mul_ps(vals, full_coeffs);
                let h = _mm256_hsum_ps(a);

                _mm_store_ss(output.get_unchecked_mut(x), h);
                x += 1;
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn convolve_8taps(
        &self,
        arena: &[f32],
        output: &mut [f32],
        kernel: &[f32; 8],
        filter_offset: isize,
    ) {
        unsafe {
            const FILTER_LENGTH: usize = 8;
            let paddings = ConvolvePaddings::from_filter(FILTER_LENGTH, filter_offset);
            let padding_right = paddings.padding_right.min(arena.len());
            let padding_left = paddings.padding_left.min(arena.len());
            let offset = paddings.padding_left as isize;

            let interpolation = BorderInterpolation::new(self.border_mode, 0, arena.len() as isize);

            let coeffs = _mm256_loadu_ps(kernel.as_ptr());

            let mut x = 0usize;

            while x < padding_left {
                let src0 = interpolation.interpolate(arena, x as isize - offset);
                let src1 = interpolation.interpolate(arena, x as isize - offset + 1);
                let src2 = interpolation.interpolate(arena, x as isize - offset + 2);
                let src3 = interpolation.interpolate(arena, x as isize - offset + 3);
                let src4 = interpolation.interpolate(arena, x as isize - offset + 4);
                let src5 = interpolation.interpolate(arena, x as isize - offset + 5);
                let src6 = interpolation.interpolate(arena, x as isize - offset + 6);
                let src7 = interpolation.interpolate(arena, x as isize - offset + 7);

                let vals = _mm256_setr_ps(src0, src1, src2, src3, src4, src5, src6, src7);
                let a = _mm256_mul_ps(vals, coeffs);
                let h = _mm256_hsum_ps(a);

                _mm_store_ss(output.get_unchecked_mut(x), h);
                x += 1;
            }

            let max_safe_end = arena.len().saturating_sub(padding_right);

            let c0 = _mm_permute_ps::<{ shuffle(0, 0, 0, 0) }>(_mm256_castps256_ps128(coeffs));
            let c1 = _mm_permute_ps::<{ shuffle(1, 1, 1, 1) }>(_mm256_castps256_ps128(coeffs));
            let c2 = _mm_permute_ps::<{ shuffle(2, 2, 2, 2) }>(_mm256_castps256_ps128(coeffs));
            let c3 = _mm_permute_ps::<{ shuffle(3, 3, 3, 3) }>(_mm256_castps256_ps128(coeffs));

            let c0 = _mm256_setr_m128(c0, c0);
            let c1 = _mm256_setr_m128(c1, c1);
            let c2 = _mm256_setr_m128(c2, c2);
            let c3 = _mm256_setr_m128(c3, c3);

            let coeffs2 = _mm256_extractf128_ps::<1>(coeffs);

            let c4 = _mm_permute_ps::<{ shuffle(0, 0, 0, 0) }>(coeffs2);
            let c5 = _mm_permute_ps::<{ shuffle(1, 1, 1, 1) }>(coeffs2);
            let c6 = _mm_permute_ps::<{ shuffle(2, 2, 2, 2) }>(coeffs2);
            let c7 = _mm_permute_ps::<{ shuffle(3, 3, 3, 3) }>(coeffs2);

            let c4 = _mm256_setr_m128(c4, c4);
            let c5 = _mm256_setr_m128(c5, c5);
            let c6 = _mm256_setr_m128(c6, c6);
            let c7 = _mm256_setr_m128(c7, c7);

            while x + 16 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k0 = _mm256_mul_ps(_mm256_loadu_ps(shifted_src.as_ptr()), c0);
                let mut k1 =
                    _mm256_mul_ps(_mm256_loadu_ps(shifted_src.get_unchecked(8..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr) => {
                        k0 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                            k0,
                        );
                        k1 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(shifted_src.get_unchecked($i + 8..).as_ptr()),
                            $c,
                            k1,
                        );
                    };
                }

                step!(1, c1);
                step!(2, c2);
                step!(3, c3);
                step!(4, c4);
                step!(5, c5);
                step!(6, c6);
                step!(7, c7);

                _mm256_storeu_ps(output.get_unchecked_mut(x..).as_mut_ptr(), k0);
                _mm256_storeu_ps(output.get_unchecked_mut(x + 8..).as_mut_ptr(), k1);
                x += 16;
            }

            while x + 4 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k = _mm_mul_ps(
                    _mm_loadu_ps(shifted_src.as_ptr()),
                    _mm256_castps256_ps128(c0),
                );

                macro_rules! step {
                    ($i: expr, $c: expr) => {
                        k = _mm_fmadd_ps(
                            _mm_loadu_ps(shifted_src.get_unchecked($i..).as_ptr()),
                            _mm256_castps256_ps128($c),
                            k,
                        );
                    };
                }

                step!(1, c1);
                step!(2, c2);
                step!(3, c3);
                step!(4, c4);
                step!(5, c5);
                step!(6, c6);
                step!(7, c7);

                _mm_storeu_ps(output.get_unchecked_mut(x..).as_mut_ptr(), k);
                x += 4;
            }

            while x < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let q0 = _mm256_loadu_ps(shifted_src.as_ptr());

                let w0 = _mm256_mul_ps(q0, coeffs);

                let f = _mm256_hsum_ps(w0);
                _mm_store_ss(output.get_unchecked_mut(x), f);
                x += 1;
            }

            while x < output.len() {
                let src0 = interpolation.interpolate(arena, x as isize - offset);
                let src1 = interpolation.interpolate(arena, x as isize - offset + 1);
                let src2 = interpolation.interpolate(arena, x as isize - offset + 2);
                let src3 = interpolation.interpolate(arena, x as isize - offset + 3);
                let src4 = interpolation.interpolate(arena, x as isize - offset + 4);
                let src5 = interpolation.interpolate(arena, x as isize - offset + 5);
                let src6 = interpolation.interpolate(arena, x as isize - offset + 6);
                let src7 = interpolation.interpolate(arena, x as isize - offset + 7);

                let vals = _mm256_setr_ps(src0, src1, src2, src3, src4, src5, src6, src7);
                let a = _mm256_mul_ps(vals, coeffs);
                let h = _mm256_hsum_ps(a);

                _mm_store_ss(output.get_unchecked_mut(x), h);
                x += 1;
            }
        }
    }
}

impl Convolve1d<f32> for AvxConvolution1dF32 {
    fn convolve(
        &self,
        input: &[f32],
        output: &mut [f32],
        scratch: &mut [f32],
        kernel: &[f32],
        filter_center: isize,
    ) -> Result<(), OscletError> {
        unsafe { self.convolve_impl(input, output, scratch, kernel, filter_center) }
    }

    fn scratch_size(&self, input_size: usize, filter_size: usize, filter_center: isize) -> usize {
        if filter_size == 2 || filter_size == 4 || filter_size == 6 || filter_size == 8 {
            return 0;
        }
        let paddings = ConvolvePaddings::from_filter(filter_size, filter_center);
        input_size + paddings.padding_right + paddings.padding_left
    }
}

impl AvxConvolution1dF32 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn convolve_impl(
        &self,
        input: &[f32],
        output: &mut [f32],
        scratch: &mut [f32],
        kernel: &[f32],
        filter_center: isize,
    ) -> Result<(), OscletError> {
        if input.len() != output.len() {
            return Err(OscletError::InOutSizesMismatch(input.len(), output.len()));
        }

        if input.is_empty() {
            return Err(OscletError::ZeroedBaseSize);
        }

        let filter_size = kernel.len();

        if kernel.is_empty() {
            output.copy_from_slice(input);
            return Ok(());
        }

        if filter_center.unsigned_abs() >= filter_size {
            return Err(OscletError::MisconfiguredFilterCenter(
                filter_center.unsigned_abs(),
                kernel.len(),
            ));
        }

        let required_scratch_size = self.scratch_size(input.len(), filter_size, filter_center);

        if scratch.len() < required_scratch_size {
            return Err(OscletError::ScratchSize(
                required_scratch_size,
                scratch.len(),
            ));
        }

        if filter_size == 2 {
            self.convolve_2taps(input, output, kernel.try_into().unwrap(), filter_center);
            return Ok(());
        } else if filter_size == 4 {
            self.convolve_4taps(input, output, kernel.try_into().unwrap(), filter_center);
            return Ok(());
        } else if filter_size == 6 {
            self.convolve_6taps(input, output, kernel.try_into().unwrap(), filter_center);
            return Ok(());
        } else if filter_size == 8 {
            self.convolve_8taps(input, output, kernel.try_into().unwrap(), filter_center);
            return Ok(());
        }

        let (arena, _) = scratch.split_at_mut(required_scratch_size);

        let paddings = ConvolvePaddings::from_filter(filter_size, filter_center);

        write_arena_1d(
            input,
            arena,
            paddings.padding_left,
            paddings.padding_right,
            self.border_mode,
        )?;

        unsafe {
            let c0 = _mm256_set1_ps(*kernel.get_unchecked(0));

            let mut p = output.chunks_exact_mut(16).len() * 16;

            for (x, dst) in output.chunks_exact_mut(16).enumerate() {
                let zx = x * 16;
                let shifted_src = arena.get_unchecked(zx..);

                let mut k0 = _mm256_mul_ps(_mm256_loadu_ps(shifted_src.as_ptr()), c0);
                let mut k1 =
                    _mm256_mul_ps(_mm256_loadu_ps(shifted_src.get_unchecked(8..).as_ptr()), c0);

                let mut f = 1usize;

                while f + 4 < filter_size {
                    let coeff = _mm_loadu_ps(kernel.get_unchecked(f..).as_ptr());

                    macro_rules! step {
                        ($i: expr, $coeff: expr, $k: expr) => {
                            let cc = _mm_permute_ps::<{ shuffle($k, $k, $k, $k) }>($coeff);
                            let c256 = _mm256_setr_m128(cc, cc);
                            k0 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(shifted_src.get_unchecked($i..).as_ptr()),
                                c256,
                                k0,
                            );
                            k1 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(shifted_src.get_unchecked($i + 8..).as_ptr()),
                                c256,
                                k1,
                            );
                        };
                    }
                    step!(f, coeff, 0);
                    step!(f + 1, coeff, 1);
                    step!(f + 2, coeff, 2);
                    step!(f + 3, coeff, 3);
                    f += 4;
                }

                for i in f..filter_size {
                    let coeff = _mm256_set1_ps(*kernel.get_unchecked(i));
                    k0 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(shifted_src.get_unchecked(i..).as_ptr()),
                        coeff,
                        k0,
                    );
                    k1 = _mm256_fmadd_ps(
                        _mm256_loadu_ps(shifted_src.get_unchecked(i + 8..).as_ptr()),
                        coeff,
                        k1,
                    );
                }

                _mm256_storeu_ps(dst.as_mut_ptr(), k0);
                _mm256_storeu_ps(dst.get_unchecked_mut(8..).as_mut_ptr(), k1);
            }

            let output = output.chunks_exact_mut(16).into_remainder();

            for (x, dst) in output.chunks_exact_mut(4).enumerate() {
                let zx = x * 4;
                let shifted_src = arena.get_unchecked(p + zx..);

                let mut k = _mm_mul_ps(
                    _mm_loadu_ps(shifted_src.as_ptr()),
                    _mm256_castps256_ps128(c0),
                );

                for i in 1..filter_size {
                    let coeff = _mm_set1_ps(*kernel.get_unchecked(i));
                    k = _mm_fmadd_ps(
                        _mm_loadu_ps(shifted_src.get_unchecked(i..).as_ptr()),
                        coeff,
                        k,
                    );
                }

                _mm_storeu_ps(dst.as_mut_ptr(), k);
            }

            p += output.chunks_exact_mut(4).len() * 4;
            let output = output.chunks_exact_mut(4).into_remainder();

            for (x, dst) in output.iter_mut().enumerate() {
                let shifted_src = arena.get_unchecked(p + x..);

                let mut k0 = _mm_mul_ss(
                    _mm_load_ss(shifted_src.get_unchecked(0)),
                    _mm256_castps256_ps128(c0),
                );

                for i in 1..filter_size {
                    let coeff = _mm_load_ss(kernel.get_unchecked(i));
                    k0 = _mm_fmadd_ss(_mm_load_ss(shifted_src.get_unchecked(i)), coeff, k0);
                }
                _mm_store_ss(dst as *mut f32, k0);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convolve1d::ScalarConvolution1d;
    use std::marker::PhantomData;

    #[test]
    fn test_2_taps() {
        let filter: Vec<f32> = vec![1. / 4., 1. / 4.];
        for i in 1..50 {
            let mut arr: Vec<f32> = vec![0.; i];
            for i in 0..i {
                arr[i] = i as f32;
            }
            for i in 0..2 {
                let v = arr.to_vec();
                let convolve = AvxConvolution1dF32 {
                    border_mode: BorderMode::Wrap,
                };
                let scalar_convolve = ScalarConvolution1d::<f32> {
                    phantom_data: PhantomData,
                    border_mode: BorderMode::Wrap,
                };
                let mut output = vec![0.; v.len()];
                let filter_offset = -i;
                let mut scratch =
                    vec![0.; convolve.scratch_size(arr.len(), filter.len(), filter_offset)];
                convolve
                    .convolve(&arr, &mut output, &mut scratch, &filter, filter_offset)
                    .unwrap();

                let mut output2 = vec![0.; v.len()];
                let mut scratch =
                    vec![0.; scalar_convolve.scratch_size(arr.len(), filter.len(), filter_offset)];
                scalar_convolve
                    .convolve(&arr, &mut output2, &mut scratch, &filter, filter_offset)
                    .unwrap();

                assert_eq!(output, output2, "failed at offset {}", -i);
            }
        }
    }

    #[test]
    fn test_4_taps() {
        let filter: Vec<f32> = vec![1. / 4., 1. / 4., 1. / 4., 1. / 4.];
        for i in 1..35 {
            let mut arr: Vec<f32> = vec![0.; i];
            for i in 0..i {
                arr[i] = i as f32;
            }
            for i in 0..4 {
                let v = arr.to_vec();
                let convolve = AvxConvolution1dF32 {
                    border_mode: BorderMode::Wrap,
                };
                let scalar_convolve = ScalarConvolution1d::<f32> {
                    phantom_data: PhantomData,
                    border_mode: BorderMode::Wrap,
                };
                let mut output = vec![0.; v.len()];
                let filter_offset = -i;
                let mut scratch =
                    vec![0.; convolve.scratch_size(arr.len(), filter.len(), filter_offset)];
                convolve
                    .convolve(&arr, &mut output, &mut scratch, &filter, filter_offset)
                    .unwrap();

                let mut output2 = vec![0.; v.len()];
                let mut scratch =
                    vec![0.; scalar_convolve.scratch_size(arr.len(), filter.len(), filter_offset)];
                scalar_convolve
                    .convolve(&arr, &mut output2, &mut scratch, &filter, filter_offset)
                    .unwrap();

                assert_eq!(output, output2, "failed at offset {}", -i);
            }
        }
    }

    #[test]
    fn test_6_taps() {
        let filter: Vec<f32> = vec![1. / 4., 1. / 4., 1. / 4., 1. / 4., 1. / 4., 1. / 4.];
        for i in 3..35 {
            let mut arr: Vec<f32> = vec![0.; i];
            for i in 0..i {
                arr[i] = i as f32;
            }
            for i in 0..6 {
                let v = arr.to_vec();
                let convolve = AvxConvolution1dF32 {
                    border_mode: BorderMode::Wrap,
                };
                let scalar_convolve = ScalarConvolution1d::<f32> {
                    phantom_data: PhantomData,
                    border_mode: BorderMode::Wrap,
                };
                let mut output = vec![0.; v.len()];
                let filter_offset = -i;
                let mut scratch =
                    vec![0.; convolve.scratch_size(arr.len(), filter.len(), filter_offset)];
                convolve
                    .convolve(&arr, &mut output, &mut scratch, &filter, filter_offset)
                    .unwrap();

                let mut output2 = vec![0.; v.len()];
                let mut scratch =
                    vec![0.; scalar_convolve.scratch_size(arr.len(), filter.len(), filter_offset)];
                scalar_convolve
                    .convolve(&arr, &mut output2, &mut scratch, &filter, filter_offset)
                    .unwrap();

                assert_eq!(output, output2, "failed at offset {}", -i);
            }
        }
    }

    #[test]
    fn test_8_taps() {
        let filter: Vec<f32> = vec![
            1. / 4.,
            1. / 4.,
            1. / 4.,
            1. / 4.,
            1. / 4.,
            1. / 4.,
            1. / 4.,
            1. / 4.,
        ];
        for i in 1..35 {
            let mut arr: Vec<f32> = vec![0.; i];
            for i in 0..i {
                arr[i] = i as f32;
            }
            for i in 0..8 {
                let v = arr.to_vec();
                let convolve = AvxConvolution1dF32 {
                    border_mode: BorderMode::Wrap,
                };
                let scalar_convolve = ScalarConvolution1d::<f32> {
                    phantom_data: PhantomData,
                    border_mode: BorderMode::Wrap,
                };
                let mut output = vec![0.; v.len()];
                let filter_offset = -i;
                let mut scratch =
                    vec![0.; convolve.scratch_size(arr.len(), filter.len(), filter_offset)];
                convolve
                    .convolve(&arr, &mut output, &mut scratch, &filter, filter_offset)
                    .unwrap();

                let mut output2 = vec![0.; v.len()];
                let mut scratch =
                    vec![0.; scalar_convolve.scratch_size(arr.len(), filter.len(), filter_offset)];
                scalar_convolve
                    .convolve(&arr, &mut output2, &mut scratch, &filter, filter_offset)
                    .unwrap();

                assert_eq!(output, output2, "failed at offset {}", -i);
            }
        }
    }
}
