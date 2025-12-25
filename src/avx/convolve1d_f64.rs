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
use crate::BorderMode;
use crate::avx::util::{_mm256_hsum_pd, shuffle};
use crate::convolve1d::{Convolve1d, ConvolvePaddings};
use crate::err::OscletError;
use crate::filter_padding::write_arena_1d;
use std::arch::x86_64::*;

pub(crate) struct AvxConvolution1dF64 {
    pub(crate) border_mode: BorderMode,
}

impl AvxConvolution1dF64 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn convolve_4taps(&self, arena: &[f64], output: &mut [f64], kernel: &[f64; 4]) {
        unsafe {
            let coeffs = _mm256_loadu_pd(kernel.as_ptr().cast());

            let c0 = _mm256_permute4x64_pd::<{ shuffle(0, 0, 0, 0) }>(coeffs);
            let c1 = _mm256_permute4x64_pd::<{ shuffle(1, 1, 1, 1) }>(coeffs);
            let c2 = _mm256_permute4x64_pd::<{ shuffle(2, 2, 2, 2) }>(coeffs);
            let c3 = _mm256_permute4x64_pd::<{ shuffle(3, 3, 3, 3) }>(coeffs);

            let mut p = output.chunks_exact_mut(8).len() * 8;

            for (x, dst) in output.chunks_exact_mut(8).enumerate() {
                let zx = x * 8;
                let shifted_src = arena.get_unchecked(zx..);

                let mut k0 = _mm256_mul_pd(_mm256_loadu_pd(shifted_src.as_ptr()), c0);
                let mut k1 =
                    _mm256_mul_pd(_mm256_loadu_pd(shifted_src.get_unchecked(4..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr) => {
                        k0 = _mm256_fmadd_pd(
                            _mm256_loadu_pd(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                            k0,
                        );
                        k1 = _mm256_fmadd_pd(
                            _mm256_loadu_pd(shifted_src.get_unchecked($i + 4..).as_ptr()),
                            $c,
                            k1,
                        );
                    };
                }

                step!(1, c1);
                step!(2, c2);
                step!(3, c3);

                _mm256_storeu_pd(dst.as_mut_ptr(), k0);
                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr(), k1);
            }

            let output = output.chunks_exact_mut(8).into_remainder();

            for (x, dst) in output.chunks_exact_mut(4).enumerate() {
                let zx = x * 4;
                let shifted_src = arena.get_unchecked(p + zx..);

                let mut k0 = _mm256_mul_pd(_mm256_loadu_pd(shifted_src.as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr) => {
                        k0 = _mm256_fmadd_pd(
                            _mm256_loadu_pd(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                            k0,
                        );
                    };
                }

                step!(1, c1);
                step!(2, c2);
                step!(3, c3);

                _mm256_storeu_pd(dst.as_mut_ptr(), k0);
            }

            p += output.chunks_exact_mut(4).len() * 4;
            let output = output.chunks_exact_mut(4).into_remainder();

            for (x, dst) in output.iter_mut().enumerate() {
                let shifted_src = arena.get_unchecked(p + x..);

                let q0 = _mm256_loadu_pd(shifted_src.as_ptr());
                let w0 = _mm256_mul_pd(q0, coeffs);

                let w1 = _mm256_hsum_pd(w0);
                _mm_store_sd(dst, w1);
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn convolve_6taps(&self, arena: &[f64], output: &mut [f64], kernel: &[f64; 6]) {
        assert_eq!(kernel.len(), 6);
        unsafe {
            let coeffs = _mm256_loadu_pd(kernel.as_ptr());

            let c0 = _mm256_permute4x64_pd::<{ shuffle(0, 0, 0, 0) }>(coeffs);
            let c1 = _mm256_permute4x64_pd::<{ shuffle(1, 1, 1, 1) }>(coeffs);
            let c2 = _mm256_permute4x64_pd::<{ shuffle(2, 2, 2, 2) }>(coeffs);
            let c3 = _mm256_permute4x64_pd::<{ shuffle(3, 3, 3, 3) }>(coeffs);

            let coeffs2x = _mm_loadu_pd(kernel.get_unchecked(4..).as_ptr());
            let coeffs2 = _mm256_setr_m128d(coeffs2x, coeffs2x);

            let c4 = _mm256_permute4x64_pd::<{ shuffle(0, 0, 0, 0) }>(coeffs2);
            let c5 = _mm256_permute4x64_pd::<{ shuffle(1, 1, 1, 1) }>(coeffs2);

            let mut p = output.chunks_exact_mut(8).len() * 8;

            for (x, dst) in output.chunks_exact_mut(8).enumerate() {
                let zx = x * 8;
                let shifted_src = arena.get_unchecked(zx..);

                let mut k0 = _mm256_mul_pd(_mm256_loadu_pd(shifted_src.as_ptr()), c0);
                let mut k1 =
                    _mm256_mul_pd(_mm256_loadu_pd(shifted_src.get_unchecked(4..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr) => {
                        k0 = _mm256_fmadd_pd(
                            _mm256_loadu_pd(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                            k0,
                        );
                        k1 = _mm256_fmadd_pd(
                            _mm256_loadu_pd(shifted_src.get_unchecked($i + 4..).as_ptr()),
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

                _mm256_storeu_pd(dst.as_mut_ptr(), k0);
                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr(), k1);
            }

            let output = output.chunks_exact_mut(8).into_remainder();

            for (x, dst) in output.chunks_exact_mut(4).enumerate() {
                let zx = x * 4;
                let shifted_src = arena.get_unchecked(p + zx..);

                let mut k0 = _mm256_mul_pd(_mm256_loadu_pd(shifted_src.as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr) => {
                        k0 = _mm256_fmadd_pd(
                            _mm256_loadu_pd(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                            k0,
                        );
                    };
                }

                step!(1, c1);
                step!(2, c2);
                step!(3, c3);
                step!(4, c4);
                step!(5, c5);

                _mm256_storeu_pd(dst.as_mut_ptr(), k0);
            }

            p += output.chunks_exact_mut(4).len() * 4;
            let output = output.chunks_exact_mut(4).into_remainder();

            for (x, dst) in output.iter_mut().enumerate() {
                let shifted_src = arena.get_unchecked(p + x..);

                let q0 = _mm256_loadu_pd(shifted_src.as_ptr());
                let q2 = _mm256_setr_m128d(
                    _mm_loadu_pd(shifted_src.get_unchecked(4..).as_ptr()),
                    _mm_setzero_pd(),
                );

                let b = _mm256_mul_pd(q0, coeffs);
                let w1 = _mm256_fmadd_pd(q2, _mm256_setr_m128d(coeffs2x, _mm_setzero_pd()), b);

                let f = _mm256_hsum_pd(w1);
                _mm_store_sd(dst, f);
            }
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn convolve_8taps(&self, arena: &[f64], output: &mut [f64], kernel: &[f64; 8]) {
        unsafe {
            let coeffs = _mm256_loadu_pd(kernel.as_ptr());
            let coeffs2 = _mm256_loadu_pd(kernel.get_unchecked(4..).as_ptr());

            let c0 = _mm256_permute4x64_pd::<{ shuffle(0, 0, 0, 0) }>(coeffs);
            let c1 = _mm256_permute4x64_pd::<{ shuffle(1, 1, 1, 1) }>(coeffs);
            let c2 = _mm256_permute4x64_pd::<{ shuffle(2, 2, 2, 2) }>(coeffs);
            let c3 = _mm256_permute4x64_pd::<{ shuffle(3, 3, 3, 3) }>(coeffs);

            let c4 = _mm256_permute4x64_pd::<{ shuffle(0, 0, 0, 0) }>(coeffs2);
            let c5 = _mm256_permute4x64_pd::<{ shuffle(1, 1, 1, 1) }>(coeffs2);
            let c6 = _mm256_permute4x64_pd::<{ shuffle(2, 2, 2, 2) }>(coeffs2);
            let c7 = _mm256_permute4x64_pd::<{ shuffle(3, 3, 3, 3) }>(coeffs2);

            let mut p = output.chunks_exact_mut(8).len() * 8;

            for (x, dst) in output.chunks_exact_mut(8).enumerate() {
                let zx = x * 8;
                let shifted_src = arena.get_unchecked(zx..);

                let mut k0 = _mm256_mul_pd(_mm256_loadu_pd(shifted_src.as_ptr()), c0);
                let mut k1 =
                    _mm256_mul_pd(_mm256_loadu_pd(shifted_src.get_unchecked(4..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr) => {
                        k0 = _mm256_fmadd_pd(
                            _mm256_loadu_pd(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                            k0,
                        );
                        k1 = _mm256_fmadd_pd(
                            _mm256_loadu_pd(shifted_src.get_unchecked($i + 4..).as_ptr()),
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

                _mm256_storeu_pd(dst.as_mut_ptr(), k0);
                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr(), k1);
            }

            let output = output.chunks_exact_mut(8).into_remainder();

            for (x, dst) in output.chunks_exact_mut(4).enumerate() {
                let zx = x * 4;
                let shifted_src = arena.get_unchecked(p + zx..);

                let mut k0 = _mm256_mul_pd(_mm256_loadu_pd(shifted_src.as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr) => {
                        k0 = _mm256_fmadd_pd(
                            _mm256_loadu_pd(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                            k0,
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

                _mm256_storeu_pd(dst.as_mut_ptr(), k0);
            }

            p += output.chunks_exact_mut(4).len() * 4;
            let output = output.chunks_exact_mut(4).into_remainder();

            for (x, dst) in output.iter_mut().enumerate() {
                let shifted_src = arena.get_unchecked(p + x..);

                let q0 = _mm256_loadu_pd(shifted_src.as_ptr());
                let q1 = _mm256_loadu_pd(shifted_src.get_unchecked(4..).as_ptr());

                let w0 = _mm256_fmadd_pd(q1, coeffs2, _mm256_mul_pd(q0, coeffs));

                let f = _mm256_hsum_pd(w0);
                _mm_store_sd(dst, f);
            }
        }
    }
}

impl Convolve1d<f64> for AvxConvolution1dF64 {
    fn convolve(
        &self,
        input: &[f64],
        output: &mut [f64],
        scratch: &mut [f64],
        kernel: &[f64],
        filter_center: isize,
    ) -> Result<(), OscletError> {
        unsafe { self.convolve_impl(input, output, scratch, kernel, filter_center) }
    }

    fn scratch_size(&self, input_size: usize, filter_size: usize, filter_center: isize) -> usize {
        let paddings = ConvolvePaddings::from_filter(filter_size, filter_center);
        input_size + paddings.padding_right + paddings.padding_left
    }
}

impl AvxConvolution1dF64 {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn convolve_impl(
        &self,
        input: &[f64],
        output: &mut [f64],
        scratch: &mut [f64],
        kernel: &[f64],
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

        let (arena, _) = scratch.split_at_mut(required_scratch_size);

        let paddings = ConvolvePaddings::from_filter(filter_size, filter_center);

        write_arena_1d(
            input,
            arena,
            paddings.padding_left,
            paddings.padding_right,
            self.border_mode,
        )?;

        if filter_size == 4 {
            self.convolve_4taps(arena, output, kernel.try_into().unwrap());
            return Ok(());
        } else if filter_size == 6 {
            self.convolve_6taps(arena, output, kernel.try_into().unwrap());
            return Ok(());
        } else if filter_size == 8 {
            self.convolve_8taps(arena, output, kernel.try_into().unwrap());
            return Ok(());
        }

        unsafe {
            let c0 = _mm256_set1_pd(*kernel.get_unchecked(0));

            let mut p = output.chunks_exact_mut(8).len() * 8;

            for (x, dst) in output.chunks_exact_mut(8).enumerate() {
                let zx = x * 8;
                let shifted_src = arena.get_unchecked(zx..);

                let mut k0 = _mm256_mul_pd(_mm256_loadu_pd(shifted_src.as_ptr()), c0);
                let mut k1 =
                    _mm256_mul_pd(_mm256_loadu_pd(shifted_src.get_unchecked(4..).as_ptr()), c0);

                for i in 1..filter_size {
                    let coeff = _mm256_set1_pd(*kernel.get_unchecked(i));
                    k0 = _mm256_fmadd_pd(
                        _mm256_loadu_pd(shifted_src.get_unchecked(i..).as_ptr()),
                        coeff,
                        k0,
                    );
                    k1 = _mm256_fmadd_pd(
                        _mm256_loadu_pd(shifted_src.get_unchecked(i + 4..).as_ptr()),
                        coeff,
                        k1,
                    );
                }

                _mm256_storeu_pd(dst.as_mut_ptr(), k0);
                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr(), k1);
            }

            let output = output.chunks_exact_mut(8).into_remainder();

            for (x, dst) in output.chunks_exact_mut(4).enumerate() {
                let zx = x * 4;
                let shifted_src = arena.get_unchecked(p + zx..);

                let mut k0 = _mm256_mul_pd(_mm256_loadu_pd(shifted_src.as_ptr()), c0);

                for i in 1..filter_size {
                    let coeff = _mm256_set1_pd(*kernel.get_unchecked(i));
                    k0 = _mm256_fmadd_pd(
                        _mm256_loadu_pd(shifted_src.get_unchecked(i..).as_ptr()),
                        coeff,
                        k0,
                    );
                }

                _mm256_storeu_pd(dst.as_mut_ptr(), k0);
            }

            p += output.chunks_exact_mut(4).len() * 4;
            let output = output.chunks_exact_mut(4).into_remainder();

            for (x, dst) in output.iter_mut().enumerate() {
                let shifted_src = arena.get_unchecked(p + x..);

                let mut k0 = _mm_mul_sd(
                    _mm_load_sd(shifted_src.get_unchecked(0)),
                    _mm256_castpd256_pd128(c0),
                );

                for i in 1..filter_size {
                    let coeff = _mm_load_sd(kernel.get_unchecked(i));
                    k0 = _mm_fmadd_sd(_mm_load_sd(shifted_src.get_unchecked(i)), coeff, k0);
                }
                _mm_store_sd(dst, k0);
            }
        }

        Ok(())
    }
}
