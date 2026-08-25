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
use crate::sse::sse_vector_d::SseVectorD;
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
            let c0 = SseVectorD::dup(*kernel.get_unchecked(0));

            let mut p = output.as_chunks_mut::<8>().0.iter_mut().len() * 8;

            for (x, dst) in output.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let zx = x * 8;
                let shifted_src = arena.get_unchecked(zx..);

                let mut k0 = SseVectorD::load(shifted_src) * c0;
                let mut k1 = SseVectorD::load(shifted_src.get_unchecked(2..)) * c0;
                let mut k2 = SseVectorD::load(shifted_src.get_unchecked(4..)) * c0;
                let mut k3 = SseVectorD::load(shifted_src.get_unchecked(6..)) * c0;

                let mut f = 1usize;

                while f + 4 < filter_size {
                    let c0 = SseVectorD::load(kernel.get_unchecked(f..));
                    let c1 = SseVectorD::load(kernel.get_unchecked(f + 2..));
                    macro_rules! step {
                        ($i: expr, $c: expr, $k: expr) => {
                            let c = $c.duplicate_element::<$k>();
                            k0 = SseVectorD::load(shifted_src.get_unchecked($i..)).mul_add(c, k0);
                            k1 = SseVectorD::load(shifted_src.get_unchecked($i + 2..))
                                .mul_add(c, k1);
                            k2 = SseVectorD::load(shifted_src.get_unchecked($i + 4..))
                                .mul_add(c, k2);
                            k3 = SseVectorD::load(shifted_src.get_unchecked($i + 6..))
                                .mul_add(c, k3);
                        };
                    }
                    step!(f, c0, 0);
                    step!(f + 1, c0, 1);
                    step!(f + 2, c1, 0);
                    step!(f + 3, c1, 1);
                    f += 4;
                }

                for i in f..filter_size {
                    let coeff = SseVectorD::load1(kernel.get_unchecked(i..));
                    k0 = SseVectorD::load(shifted_src.get_unchecked(i..)).mul_add(coeff, k0);
                    k1 = SseVectorD::load(shifted_src.get_unchecked(i + 2..)).mul_add(coeff, k1);
                    k2 = SseVectorD::load(shifted_src.get_unchecked(i + 4..)).mul_add(coeff, k2);
                    k3 = SseVectorD::load(shifted_src.get_unchecked(i + 6..)).mul_add(coeff, k3);
                }

                k0.write(dst);
                k1.write(dst.get_unchecked_mut(2..));
                k2.write(dst.get_unchecked_mut(4..));
                k3.write(dst.get_unchecked_mut(6..));
            }

            let output = output.as_chunks_mut::<8>().1;

            for (x, dst) in output.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let zx = x * 4;
                let shifted_src = arena.get_unchecked(p + zx..);

                let mut k0 = SseVectorD::load(shifted_src) * c0;
                let mut k1 = SseVectorD::load(shifted_src.get_unchecked(2..)) * c0;

                for i in 1..filter_size {
                    let coeff = SseVectorD::load1(kernel.get_unchecked(i..));
                    k0 = SseVectorD::load(shifted_src.get_unchecked(i..)).mul_add(coeff, k0);
                    k1 = SseVectorD::load(shifted_src.get_unchecked(i + 2..)).mul_add(coeff, k1);
                }

                k0.write(dst);
                k1.write(dst.get_unchecked_mut(2..));
            }

            p += output.as_chunks_mut::<4>().0.iter_mut().len() * 4;
            let output = output.as_chunks_mut::<4>().1;

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
