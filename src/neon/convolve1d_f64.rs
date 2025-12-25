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
use crate::convolve1d::{Convolve1d, ConvolvePaddings};
use crate::err::OscletError;
use crate::filter_padding::write_arena_1d;
use crate::mla::fmla;
use std::arch::aarch64::*;
use std::ops::Mul;

pub(crate) struct NeonConvolution1dF64 {
    pub(crate) border_mode: BorderMode,
}

impl NeonConvolution1dF64 {
    fn convolve_4taps(&self, arena: &[f64], output: &mut [f64], kernel: &[f64; 4]) {
        unsafe {
            let c0 = vld1q_f64(kernel.as_ptr().cast());
            let c1 = vld1q_f64(kernel.get_unchecked(2..).as_ptr().cast());

            let mut p = output.chunks_exact_mut(8).len() * 8;

            for (x, dst) in output.chunks_exact_mut(8).enumerate() {
                let zx = x * 8;
                let shifted_src = arena.get_unchecked(zx..);

                let mut k0 = vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.as_ptr()), c0);
                let mut k1 =
                    vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.get_unchecked(2..).as_ptr()), c0);
                let mut k2 =
                    vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.get_unchecked(4..).as_ptr()), c0);
                let mut k3 =
                    vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.get_unchecked(6..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k0 = vfmaq_laneq_f64::<$l>(
                            k0,
                            vld1q_f64(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                        k1 = vfmaq_laneq_f64::<$l>(
                            k1,
                            vld1q_f64(shifted_src.get_unchecked($i + 2..).as_ptr()),
                            $c,
                        );
                        k2 = vfmaq_laneq_f64::<$l>(
                            k2,
                            vld1q_f64(shifted_src.get_unchecked($i + 4..).as_ptr()),
                            $c,
                        );
                        k3 = vfmaq_laneq_f64::<$l>(
                            k3,
                            vld1q_f64(shifted_src.get_unchecked($i + 6..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);
                step!(2, c1, 0);
                step!(3, c1, 1);

                vst1q_f64(dst.as_mut_ptr(), k0);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr(), k1);
                vst1q_f64(dst.get_unchecked_mut(4..).as_mut_ptr(), k2);
                vst1q_f64(dst.get_unchecked_mut(6..).as_mut_ptr(), k3);
            }

            let output = output.chunks_exact_mut(8).into_remainder();

            for (x, dst) in output.chunks_exact_mut(4).enumerate() {
                let zx = x * 4;
                let shifted_src = arena.get_unchecked(p + zx..);

                let mut k0 = vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.as_ptr()), c0);
                let mut k1 =
                    vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.get_unchecked(2..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k0 = vfmaq_laneq_f64::<$l>(
                            k0,
                            vld1q_f64(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                        k1 = vfmaq_laneq_f64::<$l>(
                            k1,
                            vld1q_f64(shifted_src.get_unchecked($i + 2..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);
                step!(2, c1, 0);
                step!(3, c0, 1);

                vst1q_f64(dst.as_mut_ptr(), k0);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr(), k1);
            }

            p += output.chunks_exact_mut(4).len() * 4;
            let output = output.chunks_exact_mut(4).into_remainder();

            for (x, dst) in output.iter_mut().enumerate() {
                let shifted_src = arena.get_unchecked(p + x..);

                let q0 = vld1q_f64(shifted_src.as_ptr());
                let q1 = vld1q_f64(shifted_src.get_unchecked(2..).as_ptr());
                let w0 = vmulq_f64(q0, c0);
                let w1 = vfmaq_f64(w0, q1, c1);

                *dst = vpaddd_f64(w1);
            }
        }
    }

    fn convolve_6taps(&self, arena: &[f64], output: &mut [f64], kernel: &[f64; 6]) {
        unsafe {
            let c0 = vld1q_f64(kernel.as_ptr());
            let c2 = vld1q_f64(kernel.get_unchecked(2..).as_ptr());
            let c4 = vld1q_f64(kernel.get_unchecked(4..).as_ptr());

            let mut p = output.chunks_exact_mut(8).len() * 8;

            for (x, dst) in output.chunks_exact_mut(8).enumerate() {
                let zx = x * 8;
                let shifted_src = arena.get_unchecked(zx..);

                let mut k0 = vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.as_ptr()), c0);
                let mut k1 =
                    vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.get_unchecked(2..).as_ptr()), c0);
                let mut k2 =
                    vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.get_unchecked(4..).as_ptr()), c0);
                let mut k3 =
                    vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.get_unchecked(6..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k0 = vfmaq_laneq_f64::<$l>(
                            k0,
                            vld1q_f64(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                        k1 = vfmaq_laneq_f64::<$l>(
                            k1,
                            vld1q_f64(shifted_src.get_unchecked($i + 2..).as_ptr()),
                            $c,
                        );
                        k2 = vfmaq_laneq_f64::<$l>(
                            k2,
                            vld1q_f64(shifted_src.get_unchecked($i + 4..).as_ptr()),
                            $c,
                        );
                        k3 = vfmaq_laneq_f64::<$l>(
                            k3,
                            vld1q_f64(shifted_src.get_unchecked($i + 6..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);
                step!(2, c2, 0);
                step!(3, c2, 1);
                step!(4, c4, 0);
                step!(5, c4, 1);

                vst1q_f64(dst.as_mut_ptr(), k0);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr(), k1);
                vst1q_f64(dst.get_unchecked_mut(4..).as_mut_ptr(), k2);
                vst1q_f64(dst.get_unchecked_mut(6..).as_mut_ptr(), k3);
            }

            let output = output.chunks_exact_mut(8).into_remainder();

            for (x, dst) in output.chunks_exact_mut(4).enumerate() {
                let zx = x * 4;
                let shifted_src = arena.get_unchecked(p + zx..);

                let mut k0 = vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.as_ptr()), c0);
                let mut k1 =
                    vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.get_unchecked(2..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k0 = vfmaq_laneq_f64::<$l>(
                            k0,
                            vld1q_f64(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                        k1 = vfmaq_laneq_f64::<$l>(
                            k1,
                            vld1q_f64(shifted_src.get_unchecked($i + 2..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);
                step!(2, c2, 0);
                step!(3, c2, 1);
                step!(4, c4, 0);
                step!(5, c4, 1);

                vst1q_f64(dst.as_mut_ptr(), k0);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr(), k1);
            }

            p += output.chunks_exact_mut(4).len() * 4;
            let output = output.chunks_exact_mut(4).into_remainder();

            for (x, dst) in output.iter_mut().enumerate() {
                let shifted_src = arena.get_unchecked(p + x..);

                let q0 = vld1q_f64(shifted_src.as_ptr());
                let q1 = vld1q_f64(shifted_src.get_unchecked(2..).as_ptr());
                let q2 = vld1q_f64(shifted_src.get_unchecked(4..).as_ptr());

                let b = vmulq_f64(q0, c0);
                let w0 = vfmaq_f64(b, q1, c2);
                let w1 = vfmaq_f64(w0, q2, c4);

                let f = vpaddd_f64(w1);
                *dst = f;
            }
        }
    }

    fn convolve_8taps(&self, arena: &[f64], output: &mut [f64], kernel: &[f64; 8]) {
        unsafe {
            let c0 = vld1q_f64(kernel.as_ptr());
            let c2 = vld1q_f64(kernel.get_unchecked(2..).as_ptr());
            let c4 = vld1q_f64(kernel.get_unchecked(4..).as_ptr());
            let c6 = vld1q_f64(kernel.get_unchecked(6..).as_ptr());

            let mut p = output.chunks_exact_mut(8).len() * 8;

            for (x, dst) in output.chunks_exact_mut(8).enumerate() {
                let zx = x * 8;
                let shifted_src = arena.get_unchecked(zx..);

                let mut k0 = vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.as_ptr()), c0);
                let mut k1 =
                    vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.get_unchecked(2..).as_ptr()), c0);
                let mut k2 =
                    vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.get_unchecked(4..).as_ptr()), c0);
                let mut k3 =
                    vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.get_unchecked(6..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k0 = vfmaq_laneq_f64::<$l>(
                            k0,
                            vld1q_f64(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                        k1 = vfmaq_laneq_f64::<$l>(
                            k1,
                            vld1q_f64(shifted_src.get_unchecked($i + 2..).as_ptr()),
                            $c,
                        );
                        k2 = vfmaq_laneq_f64::<$l>(
                            k2,
                            vld1q_f64(shifted_src.get_unchecked($i + 4..).as_ptr()),
                            $c,
                        );
                        k3 = vfmaq_laneq_f64::<$l>(
                            k3,
                            vld1q_f64(shifted_src.get_unchecked($i + 6..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);
                step!(2, c2, 0);
                step!(3, c2, 1);
                step!(4, c4, 0);
                step!(5, c4, 1);
                step!(6, c6, 0);
                step!(7, c6, 1);

                vst1q_f64(dst.as_mut_ptr(), k0);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr(), k1);
                vst1q_f64(dst.get_unchecked_mut(4..).as_mut_ptr(), k2);
                vst1q_f64(dst.get_unchecked_mut(6..).as_mut_ptr(), k3);
            }

            let output = output.chunks_exact_mut(8).into_remainder();

            for (x, dst) in output.chunks_exact_mut(4).enumerate() {
                let zx = x * 4;
                let shifted_src = arena.get_unchecked(p + zx..);

                let mut k0 = vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.as_ptr()), c0);
                let mut k1 =
                    vmulq_laneq_f64::<0>(vld1q_f64(shifted_src.get_unchecked(2..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k0 = vfmaq_laneq_f64::<$l>(
                            k0,
                            vld1q_f64(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                        k1 = vfmaq_laneq_f64::<$l>(
                            k1,
                            vld1q_f64(shifted_src.get_unchecked($i + 2..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);
                step!(2, c2, 0);
                step!(3, c2, 1);
                step!(4, c4, 0);
                step!(5, c4, 1);
                step!(6, c6, 0);
                step!(7, c6, 1);

                vst1q_f64(dst.as_mut_ptr(), k0);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr(), k1);
            }

            p += output.chunks_exact_mut(4).len() * 4;
            let output = output.chunks_exact_mut(4).into_remainder();

            for (x, dst) in output.iter_mut().enumerate() {
                let shifted_src = arena.get_unchecked(p + x..);

                let q0 = vld1q_f64(shifted_src.as_ptr());
                let q1 = vld1q_f64(shifted_src.get_unchecked(2..).as_ptr());
                let q2 = vld1q_f64(shifted_src.get_unchecked(4..).as_ptr());
                let q3 = vld1q_f64(shifted_src.get_unchecked(6..).as_ptr());

                let w0 = vfmaq_f64(vmulq_f64(q0, c0), q1, c2);
                let w1 = vfmaq_f64(w0, q2, c4);
                let w2 = vfmaq_f64(w1, q3, c6);

                let f = vpaddd_f64(w2);
                *dst = f;
            }
        }
    }
}

impl Convolve1d<f64> for NeonConvolution1dF64 {
    fn convolve(
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
            let c0 = vdupq_n_f64(*kernel.get_unchecked(0));

            let mut p = output.chunks_exact_mut(8).len() * 8;

            for (x, dst) in output.chunks_exact_mut(8).enumerate() {
                let zx = x * 8;
                let shifted_src = arena.get_unchecked(zx..);

                let mut k0 = vmulq_f64(vld1q_f64(shifted_src.as_ptr()), c0);
                let mut k1 = vmulq_f64(vld1q_f64(shifted_src.get_unchecked(2..).as_ptr()), c0);
                let mut k2 = vmulq_f64(vld1q_f64(shifted_src.get_unchecked(4..).as_ptr()), c0);
                let mut k3 = vmulq_f64(vld1q_f64(shifted_src.get_unchecked(6..).as_ptr()), c0);

                let mut f = 1usize;

                while f + 4 < filter_size {
                    let c0 = vld1q_f64(kernel.get_unchecked(f..).as_ptr());
                    let c1 = vld1q_f64(kernel.get_unchecked(f + 2..).as_ptr());
                    macro_rules! step {
                        ($i: expr, $c: expr, $k: expr) => {
                            k0 = vfmaq_laneq_f64::<$k>(
                                k0,
                                vld1q_f64(shifted_src.get_unchecked($i..).as_ptr()),
                                $c,
                            );
                            k1 = vfmaq_laneq_f64::<$k>(
                                k1,
                                vld1q_f64(shifted_src.get_unchecked($i + 2..).as_ptr()),
                                $c,
                            );
                            k2 = vfmaq_laneq_f64::<$k>(
                                k2,
                                vld1q_f64(shifted_src.get_unchecked($i + 4..).as_ptr()),
                                $c,
                            );
                            k3 = vfmaq_laneq_f64::<$k>(
                                k3,
                                vld1q_f64(shifted_src.get_unchecked($i + 6..).as_ptr()),
                                $c,
                            );
                        };
                    }
                    step!(f, c0, 0);
                    step!(f + 1, c0, 1);
                    step!(f + 2, c1, 0);
                    step!(f + 3, c1, 1);
                    f += 4;
                }

                for i in f..filter_size {
                    let coeff = *kernel.get_unchecked(i);
                    k0 = vfmaq_n_f64(
                        k0,
                        vld1q_f64(shifted_src.get_unchecked(i..).as_ptr()),
                        coeff,
                    );
                    k1 = vfmaq_n_f64(
                        k1,
                        vld1q_f64(shifted_src.get_unchecked(i + 2..).as_ptr()),
                        coeff,
                    );
                    k2 = vfmaq_n_f64(
                        k2,
                        vld1q_f64(shifted_src.get_unchecked(i + 4..).as_ptr()),
                        coeff,
                    );
                    k3 = vfmaq_n_f64(
                        k3,
                        vld1q_f64(shifted_src.get_unchecked(i + 6..).as_ptr()),
                        coeff,
                    );
                }

                vst1q_f64(dst.as_mut_ptr(), k0);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr(), k1);
                vst1q_f64(dst.get_unchecked_mut(4..).as_mut_ptr(), k2);
                vst1q_f64(dst.get_unchecked_mut(6..).as_mut_ptr(), k3);
            }

            let output = output.chunks_exact_mut(8).into_remainder();

            for (x, dst) in output.chunks_exact_mut(4).enumerate() {
                let zx = x * 4;
                let shifted_src = arena.get_unchecked(p + zx..);

                let mut k0 = vmulq_f64(vld1q_f64(shifted_src.as_ptr()), c0);
                let mut k1 = vmulq_f64(vld1q_f64(shifted_src.get_unchecked(2..).as_ptr()), c0);

                for i in 1..filter_size {
                    let coeff = *kernel.get_unchecked(i);
                    k0 = vfmaq_n_f64(
                        k0,
                        vld1q_f64(shifted_src.get_unchecked(i..).as_ptr()),
                        coeff,
                    );
                    k1 = vfmaq_n_f64(
                        k1,
                        vld1q_f64(shifted_src.get_unchecked(i + 2..).as_ptr()),
                        coeff,
                    );
                }

                vst1q_f64(dst.as_mut_ptr(), k0);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr(), k1);
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

    fn scratch_size(&self, input_size: usize, filter_size: usize, filter_center: isize) -> usize {
        let paddings = ConvolvePaddings::from_filter(filter_size, filter_center);
        input_size + paddings.padding_right + paddings.padding_left
    }
}
