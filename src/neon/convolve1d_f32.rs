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
use crate::border_mode::BorderInterpolation;
use crate::convolve1d::{Convolve1d, ConvolvePaddings};
use crate::err::OscletError;
use crate::filter_padding::write_arena_1d;
use crate::mla::fmla;
use std::arch::aarch64::*;
use std::ops::Mul;

pub(crate) struct NeonConvolution1dF32 {
    pub(crate) border_mode: BorderMode,
}

impl NeonConvolution1dF32 {
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

            let c0 = kernel[0];
            let c1 = kernel[1];

            let mut x = 0usize;

            let interpolation = BorderInterpolation::new(self.border_mode, 0, arena.len() as isize);

            while x < padding_left {
                let src0 = interpolation.interpolate(arena, x as isize - offset);
                let src1 = interpolation.interpolate(arena, x as isize - offset + 1);

                let mut k0 = src0.mul(c0);
                k0 = fmla(src1, c1, k0);

                *output.get_unchecked_mut(x) = k0;
                x += 1;
            }

            let max_safe_end = arena.len().saturating_sub(padding_right);

            let c0 = vld1_f32(kernel.as_ptr().cast());

            while x + 16 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k0 = vmulq_lane_f32::<0>(vld1q_f32(shifted_src.as_ptr()), c0);
                let mut k1 =
                    vmulq_lane_f32::<0>(vld1q_f32(shifted_src.get_unchecked(4..).as_ptr()), c0);
                let mut k2 =
                    vmulq_lane_f32::<0>(vld1q_f32(shifted_src.get_unchecked(8..).as_ptr()), c0);
                let mut k3 =
                    vmulq_lane_f32::<0>(vld1q_f32(shifted_src.get_unchecked(12..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k0 = vfmaq_lane_f32::<$l>(
                            k0,
                            vld1q_f32(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                        k1 = vfmaq_lane_f32::<$l>(
                            k1,
                            vld1q_f32(shifted_src.get_unchecked($i + 4..).as_ptr()),
                            $c,
                        );
                        k2 = vfmaq_lane_f32::<$l>(
                            k2,
                            vld1q_f32(shifted_src.get_unchecked($i + 8..).as_ptr()),
                            $c,
                        );
                        k3 = vfmaq_lane_f32::<$l>(
                            k3,
                            vld1q_f32(shifted_src.get_unchecked($i + 12..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);

                vst1q_f32(output.get_unchecked_mut(x..).as_mut_ptr(), k0);
                vst1q_f32(output.get_unchecked_mut(x + 4..).as_mut_ptr(), k1);
                vst1q_f32(output.get_unchecked_mut(x + 8..).as_mut_ptr(), k2);
                vst1q_f32(output.get_unchecked_mut(x + 12..).as_mut_ptr(), k3);
                x += 16;
            }

            while x + 4 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k = vmulq_lane_f32::<0>(vld1q_f32(shifted_src.as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k = vfmaq_lane_f32::<$l>(
                            k,
                            vld1q_f32(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);

                vst1q_f32(output.get_unchecked_mut(x..).as_mut_ptr(), k);
                x += 4;
            }

            while x < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let q0 = vld1_f32(shifted_src.as_ptr());
                let w0 = vmul_f32(q0, c0);

                let f = vpadds_f32(w0);
                *output.get_unchecked_mut(x) = f;
                x += 1;
            }

            let c0 = kernel[0];

            while x < output.len() {
                let src0 = interpolation.interpolate(arena, x as isize - offset);
                let src1 = interpolation.interpolate(arena, x as isize - offset + 1);
                let mut k0 = src0.mul(c0);
                k0 = fmla(src1, c1, k0);

                *output.get_unchecked_mut(x) = k0;

                x += 1;
            }
        }
    }

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

            let c0 = vld1q_f32(kernel.as_ptr().cast());

            let mut x = 0usize;

            while x < padding_left {
                let src0 = interpolation.interpolate(arena, x as isize - offset);
                let src1 = interpolation.interpolate(arena, x as isize - offset + 1);
                let src2 = interpolation.interpolate(arena, x as isize - offset + 2);
                let src3 = interpolation.interpolate(arena, x as isize - offset + 3);

                let val0 = vld1q_f32([src0, src1, src2, src3].as_ptr());

                let a = vmulq_f32(val0, c0);
                let h = vadd_f32(vget_high_f32(a), vget_low_f32(a));
                let q = vpadd_f32(h, h);

                vst1_lane_f32::<0>(output.get_unchecked_mut(x), q);

                x += 1;
            }

            let max_safe_end = arena.len().saturating_sub(padding_right);

            while x + 16 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k0 = vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.as_ptr()), c0);
                let mut k1 =
                    vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.get_unchecked(4..).as_ptr()), c0);
                let mut k2 =
                    vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.get_unchecked(8..).as_ptr()), c0);
                let mut k3 =
                    vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.get_unchecked(12..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k0 = vfmaq_laneq_f32::<$l>(
                            k0,
                            vld1q_f32(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                        k1 = vfmaq_laneq_f32::<$l>(
                            k1,
                            vld1q_f32(shifted_src.get_unchecked($i + 4..).as_ptr()),
                            $c,
                        );
                        k2 = vfmaq_laneq_f32::<$l>(
                            k2,
                            vld1q_f32(shifted_src.get_unchecked($i + 8..).as_ptr()),
                            $c,
                        );
                        k3 = vfmaq_laneq_f32::<$l>(
                            k3,
                            vld1q_f32(shifted_src.get_unchecked($i + 12..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);
                step!(2, c0, 2);
                step!(3, c0, 3);

                vst1q_f32(output.get_unchecked_mut(x..).as_mut_ptr(), k0);
                vst1q_f32(output.get_unchecked_mut(x + 4..).as_mut_ptr(), k1);
                vst1q_f32(output.get_unchecked_mut(x + 8..).as_mut_ptr(), k2);
                vst1q_f32(output.get_unchecked_mut(x + 12..).as_mut_ptr(), k3);
                x += 16;
            }

            while x + 4 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k = vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k = vfmaq_laneq_f32::<$l>(
                            k,
                            vld1q_f32(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);
                step!(2, c0, 2);
                step!(3, c0, 3);

                vst1q_f32(output.get_unchecked_mut(x..).as_mut_ptr(), k);
                x += 4;
            }

            while x < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let q0 = vld1q_f32(shifted_src.as_ptr());
                let w0 = vmulq_f32(q0, c0);

                let f = vpadds_f32(vpadd_f32(vget_low_f32(w0), vget_high_f32(w0)));
                *output.get_unchecked_mut(x) = f;
                x += 1;
            }

            while x < output.len() {
                let src0 = interpolation.interpolate(arena, x as isize - offset);
                let src1 = interpolation.interpolate(arena, x as isize - offset + 1);
                let src2 = interpolation.interpolate(arena, x as isize - offset + 2);
                let src3 = interpolation.interpolate(arena, x as isize - offset + 3);

                let val0 = vld1q_f32([src0, src1, src2, src3].as_ptr());

                let a = vmulq_f32(val0, c0);
                let h = vadd_f32(vget_high_f32(a), vget_low_f32(a));
                let q = vpadd_f32(h, h);

                vst1_lane_f32::<0>(output.get_unchecked_mut(x), q);

                x += 1;
            }
        }
    }

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

            let c0 = vld1q_f32(kernel.get_unchecked(0..).as_ptr());
            let c4 = vld1q_f32([kernel[4], kernel[5], 0., 0.].as_ptr());

            let mut x = 0usize;

            while x < padding_left {
                let src0 = interpolation.interpolate(arena, x as isize - offset);
                let src1 = interpolation.interpolate(arena, x as isize - offset + 1);
                let src2 = interpolation.interpolate(arena, x as isize - offset + 2);
                let src3 = interpolation.interpolate(arena, x as isize - offset + 3);
                let src4 = interpolation.interpolate(arena, x as isize - offset + 4);
                let src5 = interpolation.interpolate(arena, x as isize - offset + 5);

                let val0 = vld1q_f32([src0, src1, src2, src3].as_ptr());
                let val1 = vld1q_f32([src4, src5, 0., 0.].as_ptr());

                let a = vfmaq_f32(vmulq_f32(val0, c0), val1, c4);
                let h = vadd_f32(vget_high_f32(a), vget_low_f32(a));
                let q = vpadd_f32(h, h);

                vst1_lane_f32::<0>(output.get_unchecked_mut(x), q);
                x += 1;
            }

            let c4 = vld1_f32(kernel.get_unchecked(4..).as_ptr());

            let max_safe_end = arena.len().saturating_sub(padding_right);

            while x + 16 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k0 = vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.as_ptr()), c0);
                let mut k1 =
                    vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.get_unchecked(4..).as_ptr()), c0);
                let mut k2 =
                    vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.get_unchecked(8..).as_ptr()), c0);
                let mut k3 =
                    vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.get_unchecked(12..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k0 = vfmaq_laneq_f32::<$l>(
                            k0,
                            vld1q_f32(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                        k1 = vfmaq_laneq_f32::<$l>(
                            k1,
                            vld1q_f32(shifted_src.get_unchecked($i + 4..).as_ptr()),
                            $c,
                        );
                        k2 = vfmaq_laneq_f32::<$l>(
                            k2,
                            vld1q_f32(shifted_src.get_unchecked($i + 8..).as_ptr()),
                            $c,
                        );
                        k3 = vfmaq_laneq_f32::<$l>(
                            k3,
                            vld1q_f32(shifted_src.get_unchecked($i + 12..).as_ptr()),
                            $c,
                        );
                    };
                }

                macro_rules! steph {
                    ($i: expr, $c: expr, $l: expr) => {
                        k0 = vfmaq_lane_f32::<$l>(
                            k0,
                            vld1q_f32(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                        k1 = vfmaq_lane_f32::<$l>(
                            k1,
                            vld1q_f32(shifted_src.get_unchecked($i + 4..).as_ptr()),
                            $c,
                        );
                        k2 = vfmaq_lane_f32::<$l>(
                            k2,
                            vld1q_f32(shifted_src.get_unchecked($i + 8..).as_ptr()),
                            $c,
                        );
                        k3 = vfmaq_lane_f32::<$l>(
                            k3,
                            vld1q_f32(shifted_src.get_unchecked($i + 12..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);
                step!(2, c0, 2);
                step!(3, c0, 3);
                steph!(4, c4, 0);
                steph!(5, c4, 1);

                vst1q_f32(output.get_unchecked_mut(x..).as_mut_ptr(), k0);
                vst1q_f32(output.get_unchecked_mut(x + 4..).as_mut_ptr(), k1);
                vst1q_f32(output.get_unchecked_mut(x + 8..).as_mut_ptr(), k2);
                vst1q_f32(output.get_unchecked_mut(x + 12..).as_mut_ptr(), k3);

                x += 16;
            }

            while x + 4 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k = vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k = vfmaq_laneq_f32::<$l>(
                            k,
                            vld1q_f32(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                    };
                }

                macro_rules! steph {
                    ($i: expr, $c: expr, $l: expr) => {
                        k = vfmaq_lane_f32::<$l>(
                            k,
                            vld1q_f32(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);
                step!(2, c0, 2);
                step!(3, c0, 3);
                steph!(4, c4, 0);
                steph!(5, c4, 1);

                vst1q_f32(output.get_unchecked_mut(x..).as_mut_ptr(), k);
                x += 4;
            }

            while x < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let q0 = vld1q_f32(shifted_src.as_ptr());
                let q1 = vld1_f32(shifted_src.get_unchecked(4..).as_ptr());

                let b = vmulq_f32(q0, c0);
                let w0 = vfma_f32(vpadd_f32(vget_low_f32(b), vget_high_f32(b)), q1, c4);

                let f = vpadds_f32(w0);
                *output.get_unchecked_mut(x) = f;

                x += 1;
            }

            let c4 = vld1q_f32([kernel[4], kernel[5], 0., 0.].as_ptr());

            while x < output.len() {
                let src0 = interpolation.interpolate(arena, x as isize - offset);
                let src1 = interpolation.interpolate(arena, x as isize - offset + 1);
                let src2 = interpolation.interpolate(arena, x as isize - offset + 2);
                let src3 = interpolation.interpolate(arena, x as isize - offset + 3);
                let src4 = interpolation.interpolate(arena, x as isize - offset + 4);
                let src5 = interpolation.interpolate(arena, x as isize - offset + 5);

                let val0 = vld1q_f32([src0, src1, src2, src3].as_ptr());
                let val1 = vld1q_f32([src4, src5, 0., 0.].as_ptr());

                let a = vfmaq_f32(vmulq_f32(val0, c0), val1, c4);
                let h = vadd_f32(vget_high_f32(a), vget_low_f32(a));
                let q = vpadd_f32(h, h);

                vst1_lane_f32::<0>(output.get_unchecked_mut(x), q);
                x += 1;
            }
        }
    }

    fn convolve_8taps(
        &self,
        arena: &[f32],
        output: &mut [f32],
        kernel: &[f32; 8],
        filter_offset: isize,
    ) {
        const FILTER_LENGTH: usize = 8;
        let paddings = ConvolvePaddings::from_filter(FILTER_LENGTH, filter_offset);
        let padding_right = paddings.padding_right.min(arena.len());
        let padding_left = paddings.padding_left.min(arena.len());
        let offset = paddings.padding_left as isize;

        unsafe {
            let interpolation = BorderInterpolation::new(self.border_mode, 0, arena.len() as isize);

            let c0 = vld1q_f32(kernel.get_unchecked(0..).as_ptr());
            let c4 = vld1q_f32(kernel.get_unchecked(4..).as_ptr());

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

                let val0 = vld1q_f32([src0, src1, src2, src3].as_ptr());
                let val1 = vld1q_f32([src4, src5, src6, src7].as_ptr());

                let a = vfmaq_f32(vmulq_f32(val0, c0), val1, c4);
                let h = vadd_f32(vget_high_f32(a), vget_low_f32(a));
                let q = vpadd_f32(h, h);

                vst1_lane_f32::<0>(output.get_unchecked_mut(x), q);
                x += 1;
            }

            let max_safe_end = arena.len().saturating_sub(padding_right);

            while x + 16 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k0 = vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.as_ptr()), c0);
                let mut k1 =
                    vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.get_unchecked(4..).as_ptr()), c0);
                let mut k2 =
                    vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.get_unchecked(8..).as_ptr()), c0);
                let mut k3 =
                    vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.get_unchecked(12..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k0 = vfmaq_laneq_f32::<$l>(
                            k0,
                            vld1q_f32(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                        k1 = vfmaq_laneq_f32::<$l>(
                            k1,
                            vld1q_f32(shifted_src.get_unchecked($i + 4..).as_ptr()),
                            $c,
                        );
                        k2 = vfmaq_laneq_f32::<$l>(
                            k2,
                            vld1q_f32(shifted_src.get_unchecked($i + 8..).as_ptr()),
                            $c,
                        );
                        k3 = vfmaq_laneq_f32::<$l>(
                            k3,
                            vld1q_f32(shifted_src.get_unchecked($i + 12..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);
                step!(2, c0, 2);
                step!(3, c0, 3);
                step!(4, c4, 0);
                step!(5, c4, 1);
                step!(6, c4, 2);
                step!(7, c4, 3);

                vst1q_f32(output.get_unchecked_mut(x..).as_mut_ptr(), k0);
                vst1q_f32(output.get_unchecked_mut(x + 4..).as_mut_ptr(), k1);
                vst1q_f32(output.get_unchecked_mut(x + 8..).as_mut_ptr(), k2);
                vst1q_f32(output.get_unchecked_mut(x + 12..).as_mut_ptr(), k3);

                x += 16;
            }

            while x + 4 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k = vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k = vfmaq_laneq_f32::<$l>(
                            k,
                            vld1q_f32(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);
                step!(2, c0, 2);
                step!(3, c0, 3);
                step!(4, c4, 0);
                step!(5, c4, 1);
                step!(6, c4, 2);
                step!(7, c4, 3);

                vst1q_f32(output.get_unchecked_mut(x..).as_mut_ptr(), k);
                x += 4;
            }

            while x < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let q0 = vld1q_f32(shifted_src.as_ptr());
                let q1 = vld1q_f32(shifted_src.get_unchecked(4..).as_ptr());

                let w0 = vfmaq_f32(vmulq_f32(q0, c0), q1, c4);

                let f = vpadds_f32(vpadd_f32(vget_low_f32(w0), vget_high_f32(w0)));
                *output.get_unchecked_mut(x) = f;
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

                let val0 = vld1q_f32([src0, src1, src2, src3].as_ptr());
                let val1 = vld1q_f32([src4, src5, src6, src7].as_ptr());

                let a = vfmaq_f32(vmulq_f32(val0, c0), val1, c4);
                let h = vadd_f32(vget_high_f32(a), vget_low_f32(a));
                let q = vpadd_f32(h, h);

                vst1_lane_f32::<0>(output.get_unchecked_mut(x), q);
                x += 1;
            }
        }
    }

    fn convolve_10taps(
        &self,
        arena: &[f32],
        output: &mut [f32],
        kernel: &[f32; 10],
        filter_offset: isize,
    ) {
        const FILTER_LENGTH: usize = 10;
        let paddings = ConvolvePaddings::from_filter(FILTER_LENGTH, filter_offset);
        let padding_right = paddings.padding_right.min(arena.len());
        let padding_left = paddings.padding_left.min(arena.len());
        let offset = paddings.padding_left as isize;

        unsafe {
            let interpolation = BorderInterpolation::new(self.border_mode, 0, arena.len() as isize);

            let c0 = vld1q_f32(kernel.get_unchecked(0..).as_ptr());
            let c4 = vld1q_f32(kernel.get_unchecked(4..).as_ptr());
            let c8 = vld1q_f32([kernel[8], kernel[9], 0., 0.].as_ptr());

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
                let src8 = interpolation.interpolate(arena, x as isize - offset + 8);
                let src9 = interpolation.interpolate(arena, x as isize - offset + 9);

                let val0 = vld1q_f32([src0, src1, src2, src3].as_ptr());
                let val1 = vld1q_f32([src4, src5, src6, src7].as_ptr());
                let val2 = vld1q_f32([src8, src9, 0., 0.].as_ptr());

                let a = vfmaq_f32(vfmaq_f32(vmulq_f32(val0, c0), val1, c4), val2, c8);
                let h = vadd_f32(vget_high_f32(a), vget_low_f32(a));
                let q = vpadd_f32(h, h);

                vst1_lane_f32::<0>(output.get_unchecked_mut(x), q);
                x += 1;
            }

            let max_safe_end = arena.len().saturating_sub(padding_right);

            while x + 16 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k0 = vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.as_ptr()), c0);
                let mut k1 =
                    vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.get_unchecked(4..).as_ptr()), c0);
                let mut k2 =
                    vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.get_unchecked(8..).as_ptr()), c0);
                let mut k3 =
                    vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.get_unchecked(12..).as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k0 = vfmaq_laneq_f32::<$l>(
                            k0,
                            vld1q_f32(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                        k1 = vfmaq_laneq_f32::<$l>(
                            k1,
                            vld1q_f32(shifted_src.get_unchecked($i + 4..).as_ptr()),
                            $c,
                        );
                        k2 = vfmaq_laneq_f32::<$l>(
                            k2,
                            vld1q_f32(shifted_src.get_unchecked($i + 8..).as_ptr()),
                            $c,
                        );
                        k3 = vfmaq_laneq_f32::<$l>(
                            k3,
                            vld1q_f32(shifted_src.get_unchecked($i + 12..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);
                step!(2, c0, 2);
                step!(3, c0, 3);
                step!(4, c4, 0);
                step!(5, c4, 1);
                step!(6, c4, 2);
                step!(7, c4, 3);
                step!(8, c8, 0);
                step!(9, c8, 1);

                vst1q_f32(output.get_unchecked_mut(x..).as_mut_ptr(), k0);
                vst1q_f32(output.get_unchecked_mut(x + 4..).as_mut_ptr(), k1);
                vst1q_f32(output.get_unchecked_mut(x + 8..).as_mut_ptr(), k2);
                vst1q_f32(output.get_unchecked_mut(x + 12..).as_mut_ptr(), k3);

                x += 16;
            }

            while x + 4 < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let mut k = vmulq_laneq_f32::<0>(vld1q_f32(shifted_src.as_ptr()), c0);

                macro_rules! step {
                    ($i: expr, $c: expr, $l: expr) => {
                        k = vfmaq_laneq_f32::<$l>(
                            k,
                            vld1q_f32(shifted_src.get_unchecked($i..).as_ptr()),
                            $c,
                        );
                    };
                }

                step!(1, c0, 1);
                step!(2, c0, 2);
                step!(3, c0, 3);
                step!(4, c4, 0);
                step!(5, c4, 1);
                step!(6, c4, 2);
                step!(7, c4, 3);
                step!(8, c8, 0);
                step!(9, c8, 1);

                vst1q_f32(output.get_unchecked_mut(x..).as_mut_ptr(), k);
                x += 4;
            }

            while x < max_safe_end {
                let shifted_src = arena.get_unchecked(x - padding_left..);

                let q0 = vld1q_f32(shifted_src.as_ptr());
                let q1 = vld1q_f32(shifted_src.get_unchecked(4..).as_ptr());
                let q2 = vcombine_f32(
                    vld1_f32(shifted_src.get_unchecked(8..).as_ptr()),
                    vdup_n_f32(0.),
                );

                let w0 = vfmaq_f32(vfmaq_f32(vmulq_f32(q0, c0), q1, c4), q2, c8);

                let f = vpadds_f32(vpadd_f32(vget_low_f32(w0), vget_high_f32(w0)));
                *output.get_unchecked_mut(x) = f;
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
                let src8 = interpolation.interpolate(arena, x as isize - offset + 8);
                let src9 = interpolation.interpolate(arena, x as isize - offset + 9);

                let val0 = vld1q_f32([src0, src1, src2, src3].as_ptr());
                let val1 = vld1q_f32([src4, src5, src6, src7].as_ptr());
                let val2 = vld1q_f32([src8, src9, 0., 0.].as_ptr());

                let a = vfmaq_f32(vfmaq_f32(vmulq_f32(val0, c0), val1, c4), val2, c8);
                let h = vadd_f32(vget_high_f32(a), vget_low_f32(a));
                let q = vpadd_f32(h, h);

                vst1_lane_f32::<0>(output.get_unchecked_mut(x), q);
                x += 1;
            }
        }
    }
}

impl Convolve1d<f32> for NeonConvolution1dF32 {
    fn convolve(
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
        } else if filter_size == 10 {
            self.convolve_10taps(input, output, kernel.try_into().unwrap(), filter_center);
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
            let c0 = vdupq_n_f32(*kernel.get_unchecked(0));

            let mut p = output.chunks_exact_mut(16).len() * 16;

            for (x, dst) in output.chunks_exact_mut(16).enumerate() {
                let zx = x * 16;
                let shifted_src = arena.get_unchecked(zx..);

                let mut k0 = vmulq_f32(vld1q_f32(shifted_src.as_ptr()), c0);
                let mut k1 = vmulq_f32(vld1q_f32(shifted_src.get_unchecked(4..).as_ptr()), c0);
                let mut k2 = vmulq_f32(vld1q_f32(shifted_src.get_unchecked(8..).as_ptr()), c0);
                let mut k3 = vmulq_f32(vld1q_f32(shifted_src.get_unchecked(12..).as_ptr()), c0);

                let mut f = 1usize;

                while f + 4 < filter_size {
                    let coeff = vld1q_f32(kernel.get_unchecked(f..).as_ptr());
                    macro_rules! step {
                        ($i: expr, $coeff: expr, $k: expr) => {
                            k0 = vfmaq_laneq_f32::<$k>(
                                k0,
                                vld1q_f32(shifted_src.get_unchecked($i..).as_ptr()),
                                $coeff,
                            );
                            k1 = vfmaq_laneq_f32::<$k>(
                                k1,
                                vld1q_f32(shifted_src.get_unchecked($i + 4..).as_ptr()),
                                $coeff,
                            );
                            k2 = vfmaq_laneq_f32::<$k>(
                                k2,
                                vld1q_f32(shifted_src.get_unchecked($i + 8..).as_ptr()),
                                $coeff,
                            );
                            k3 = vfmaq_laneq_f32::<$k>(
                                k3,
                                vld1q_f32(shifted_src.get_unchecked($i + 12..).as_ptr()),
                                $coeff,
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
                    let coeff = *kernel.get_unchecked(i);
                    k0 = vfmaq_n_f32(
                        k0,
                        vld1q_f32(shifted_src.get_unchecked(i..).as_ptr()),
                        coeff,
                    );
                    k1 = vfmaq_n_f32(
                        k1,
                        vld1q_f32(shifted_src.get_unchecked(i + 4..).as_ptr()),
                        coeff,
                    );
                    k2 = vfmaq_n_f32(
                        k2,
                        vld1q_f32(shifted_src.get_unchecked(i + 8..).as_ptr()),
                        coeff,
                    );
                    k3 = vfmaq_n_f32(
                        k3,
                        vld1q_f32(shifted_src.get_unchecked(i + 12..).as_ptr()),
                        coeff,
                    );
                }

                vst1q_f32(dst.as_mut_ptr(), k0);
                vst1q_f32(dst.get_unchecked_mut(4..).as_mut_ptr(), k1);
                vst1q_f32(dst.get_unchecked_mut(8..).as_mut_ptr(), k2);
                vst1q_f32(dst.get_unchecked_mut(12..).as_mut_ptr(), k3);
            }

            let output = output.chunks_exact_mut(16).into_remainder();

            for (x, dst) in output.chunks_exact_mut(4).enumerate() {
                let zx = x * 4;
                let shifted_src = arena.get_unchecked(p + zx..);

                let mut k = vmulq_f32(vld1q_f32(shifted_src.as_ptr()), c0);

                for i in 1..filter_size {
                    let coeff = *kernel.get_unchecked(i);
                    k = vfmaq_n_f32(k, vld1q_f32(shifted_src.get_unchecked(i..).as_ptr()), coeff);
                }

                vst1q_f32(dst.as_mut_ptr(), k);
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
        if filter_size == 2
            || filter_size == 4
            || filter_size == 6
            || filter_size == 8
            || filter_size == 10
        {
            return 0;
        }
        let paddings = ConvolvePaddings::from_filter(filter_size, filter_center);
        input_size + paddings.padding_right + paddings.padding_left
    }
}

#[cfg(test)]
mod tests {
    use crate::BorderMode;
    use crate::convolve1d::{Convolve1d, ScalarConvolution1d};
    use crate::neon::NeonConvolution1dF32;
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
                let convolve = NeonConvolution1dF32 {
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
                let convolve = NeonConvolution1dF32 {
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
                let convolve = NeonConvolution1dF32 {
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
        for i in 3..35 {
            let mut arr: Vec<f32> = vec![0.; i];
            for i in 0..i {
                arr[i] = i as f32;
            }
            for i in 0..8 {
                let v = arr.to_vec();
                let convolve = NeonConvolution1dF32 {
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
    fn test_10_taps() {
        let filter: Vec<f32> = vec![
            1. / 4.,
            1. / 4.,
            1. / 4.,
            1. / 4.,
            1. / 4.,
            1. / 4.,
            1. / 4.,
            1. / 4.,
            1. / 4.,
            1. / 4.,
        ];
        for i in 3..50 {
            let mut arr: Vec<f32> = vec![0.; i];
            for i in 0..i {
                arr[i] = i as f32;
            }
            for i in 0..10 {
                let v = arr.to_vec();
                let convolve = NeonConvolution1dF32 {
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
