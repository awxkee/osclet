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
use crate::BorderMode;
use crate::convolve1d::{Convolve1d, ConvolvePaddings};
use crate::err::OscletError;
use crate::filter_padding::write_arena_1d;
use crate::mla::fmla;
use crate::sse::util::_mm_fma_pd;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Mul;

pub(crate) struct SseConvolution1dF64 {
    pub(crate) border_mode: BorderMode,
}

impl Convolve1d<f64> for SseConvolution1dF64 {
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

impl SseConvolution1dF64 {
    #[target_feature(enable = "sse4.2")]
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

        unsafe {
            let c0 = _mm_set1_pd(*kernel.get_unchecked(0));

            let mut p = output.chunks_exact_mut(8).len() * 8;

            for (x, dst) in output.chunks_exact_mut(8).enumerate() {
                let zx = x * 8;
                let shifted_src = arena.get_unchecked(zx..);

                let mut k0 = _mm_mul_pd(_mm_loadu_pd(shifted_src.as_ptr()), c0);
                let mut k1 = _mm_mul_pd(_mm_loadu_pd(shifted_src.get_unchecked(2..).as_ptr()), c0);
                let mut k2 = _mm_mul_pd(_mm_loadu_pd(shifted_src.get_unchecked(4..).as_ptr()), c0);
                let mut k3 = _mm_mul_pd(_mm_loadu_pd(shifted_src.get_unchecked(6..).as_ptr()), c0);

                let mut f = 1usize;

                while f + 4 < filter_size {
                    let c0 = _mm_loadu_pd(kernel.get_unchecked(f..).as_ptr());
                    let c1 = _mm_loadu_pd(kernel.get_unchecked(f + 2..).as_ptr());
                    macro_rules! step {
                        ($i: expr, $c: expr, $k: expr) => {
                            let c = _mm_shuffle_pd::<$k>($c, $c);
                            k0 = _mm_fma_pd(
                                _mm_loadu_pd(shifted_src.get_unchecked($i..).as_ptr()),
                                c,
                                k0,
                            );
                            k1 = _mm_fma_pd(
                                _mm_loadu_pd(shifted_src.get_unchecked($i + 2..).as_ptr()),
                                c,
                                k1,
                            );
                            k2 = _mm_fma_pd(
                                _mm_loadu_pd(shifted_src.get_unchecked($i + 4..).as_ptr()),
                                c,
                                k2,
                            );
                            k3 = _mm_fma_pd(
                                _mm_loadu_pd(shifted_src.get_unchecked($i + 6..).as_ptr()),
                                c,
                                k3,
                            );
                        };
                    }
                    step!(f, c0, 0);
                    step!(f + 1, c0, 0b11);
                    step!(f + 2, c1, 0);
                    step!(f + 3, c1, 0b11);
                    f += 4;
                }

                for i in f..filter_size {
                    let coeff = _mm_load1_pd(kernel.get_unchecked(i));
                    k0 = _mm_fma_pd(
                        _mm_loadu_pd(shifted_src.get_unchecked(i..).as_ptr()),
                        coeff,
                        k0,
                    );
                    k1 = _mm_fma_pd(
                        _mm_loadu_pd(shifted_src.get_unchecked(i + 2..).as_ptr()),
                        coeff,
                        k1,
                    );
                    k2 = _mm_fma_pd(
                        _mm_loadu_pd(shifted_src.get_unchecked(i + 4..).as_ptr()),
                        coeff,
                        k2,
                    );
                    k3 = _mm_fma_pd(
                        _mm_loadu_pd(shifted_src.get_unchecked(i + 6..).as_ptr()),
                        coeff,
                        k3,
                    );
                }

                _mm_storeu_pd(dst.as_mut_ptr(), k0);
                _mm_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr(), k1);
                _mm_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr(), k2);
                _mm_storeu_pd(dst.get_unchecked_mut(6..).as_mut_ptr(), k3);
            }

            let output = output.chunks_exact_mut(8).into_remainder();

            for (x, dst) in output.chunks_exact_mut(4).enumerate() {
                let zx = x * 4;
                let shifted_src = arena.get_unchecked(p + zx..);

                let mut k0 = _mm_mul_pd(_mm_loadu_pd(shifted_src.as_ptr()), c0);
                let mut k1 = _mm_mul_pd(_mm_loadu_pd(shifted_src.get_unchecked(2..).as_ptr()), c0);

                for i in 1..filter_size {
                    let coeff = _mm_load1_pd(kernel.get_unchecked(i));
                    k0 = _mm_fma_pd(
                        _mm_loadu_pd(shifted_src.get_unchecked(i..).as_ptr()),
                        coeff,
                        k0,
                    );
                    k1 = _mm_fma_pd(
                        _mm_loadu_pd(shifted_src.get_unchecked(i + 2..).as_ptr()),
                        coeff,
                        k1,
                    );
                }

                _mm_storeu_pd(dst.as_mut_ptr(), k0);
                _mm_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr(), k1);
            }

            p += output.chunks_exact_mut(4).len() * 4;
            let output = output.chunks_exact_mut(4).into_remainder();

            let c0 = *kernel.get_unchecked(0);

            for (x, dst) in output.iter_mut().enumerate() {
                let shifted_src = arena.get_unchecked(p + x..);

                let mut k0 = (*shifted_src.get_unchecked(0)).mul(c0);

                for i in 1..filter_size {
                    let coeff = *kernel.get_unchecked(i);
                    k0 = fmla(*shifted_src.get_unchecked(i), coeff, k0);
                }
                *dst = k0;
            }
        }

        Ok(())
    }
}
