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
use crate::err::OscletError;
use crate::filter_padding::write_arena_1d;
use crate::mla::fmla;
use crate::{BorderMode, WaveletSample};
use num_traits::AsPrimitive;
use std::marker::PhantomData;

pub(crate) trait Convolve1d<T> {
    fn convolve(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
        kernel: &[T],
        filter_center: isize,
    ) -> Result<(), OscletError>;
    fn scratch_size(&self, input_size: usize, filter_size: usize, filter_center: isize) -> usize;
}

pub(crate) trait ConvolveFactory<T> {
    fn make_convolution_1d(border_mode: BorderMode) -> Box<dyn Convolve1d<T> + Send + Sync>;
}

impl ConvolveFactory<f32> for f32 {
    fn make_convolution_1d(border_mode: BorderMode) -> Box<dyn Convolve1d<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonConvolution1dF32;
            Box::new(NeonConvolution1dF32 { border_mode })
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            use crate::factory::has_valid_avx;
            if has_valid_avx() {
                use crate::avx::AvxConvolution1dF32;
                return Box::new(AvxConvolution1dF32 { border_mode });
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            use crate::factory::has_valid_sse;
            if has_valid_sse() {
                use crate::sse::SseConvolution1dF32;
                return Box::new(SseConvolution1dF32 { border_mode });
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Box::new(ScalarConvolution1d {
                phantom_data: PhantomData,
                border_mode,
            })
        }
    }
}

impl ConvolveFactory<f64> for f64 {
    fn make_convolution_1d(border_mode: BorderMode) -> Box<dyn Convolve1d<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonConvolution1dF64;
            Box::new(NeonConvolution1dF64 { border_mode })
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            use crate::factory::has_valid_avx;
            if has_valid_avx() {
                use crate::avx::AvxConvolution1dF64;
                return Box::new(AvxConvolution1dF64 { border_mode });
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            use crate::factory::has_valid_sse;
            if has_valid_sse() {
                use crate::sse::SseConvolution1dF64;
                return Box::new(SseConvolution1dF64 { border_mode });
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Box::new(ScalarConvolution1d {
                phantom_data: PhantomData,
                border_mode,
            })
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct ConvolvePaddings {
    pub(crate) padding_left: usize,
    pub(crate) padding_right: usize,
}

impl ConvolvePaddings {
    pub(crate) fn from_filter(filter_size: usize, filter_center: isize) -> Self {
        let padding_left = if filter_size.is_multiple_of(2) {
            ((filter_size / 2) as isize - filter_center - 1).max(0) as usize
        } else {
            ((filter_size / 2) as isize - filter_center).max(0) as usize
        };

        let padding_right = filter_size.saturating_sub(padding_left);
        ConvolvePaddings {
            padding_left,
            padding_right,
        }
    }
}

#[allow(unused)]
pub(crate) struct ScalarConvolution1d<T> {
    pub(crate) phantom_data: PhantomData<T>,
    pub(crate) border_mode: BorderMode,
}

impl<T: WaveletSample> Convolve1d<T> for ScalarConvolution1d<T>
where
    f64: AsPrimitive<T>,
{
    fn convolve(
        &self,
        input: &[T],
        output: &mut [T],
        scratch: &mut [T],
        kernel: &[T],
        filter_center: isize,
    ) -> Result<(), OscletError> {
        if input.len() != output.len() {
            return Err(OscletError::InOutSizesMismatch(input.len(), output.len()));
        }

        let filter_size = kernel.len();

        if input.is_empty() {
            return Err(OscletError::ZeroedBaseSize);
        }

        if kernel.is_empty() {
            output.copy_from_slice(input);
            return Ok(());
        }

        let required_scratch_size = self.scratch_size(input.len(), filter_size, filter_center);

        if scratch.len() < required_scratch_size {
            return Err(OscletError::ScratchSize(
                required_scratch_size,
                scratch.len(),
            ));
        }

        let (arena, _) = scratch.split_at_mut(required_scratch_size);

        if filter_center.unsigned_abs() >= filter_size {
            return Err(OscletError::MisconfiguredFilterCenter(
                filter_center.unsigned_abs(),
                kernel.len(),
            ));
        }

        let paddings = ConvolvePaddings::from_filter(filter_size, filter_center);

        write_arena_1d(
            input,
            arena,
            paddings.padding_left,
            paddings.padding_right,
            self.border_mode,
        )?;

        let c0 = unsafe { *kernel.get_unchecked(0) };

        for (x, dst) in output.chunks_exact_mut(4).enumerate() {
            unsafe {
                let zx = x * 4;
                let shifted_src = arena.get_unchecked(zx..);

                let mut k0 = (*shifted_src.get_unchecked(0)).mul(c0);
                let mut k1 = (*shifted_src.get_unchecked(1)).mul(c0);
                let mut k2 = (*shifted_src.get_unchecked(2)).mul(c0);
                let mut k3 = (*shifted_src.get_unchecked(3)).mul(c0);

                for i in 1..filter_size {
                    let coeff = *kernel.get_unchecked(i);
                    k0 = fmla(*shifted_src.get_unchecked(i), coeff, k0);
                    k1 = fmla(*shifted_src.get_unchecked(i + 1), coeff, k1);
                    k2 = fmla(*shifted_src.get_unchecked(i + 2), coeff, k2);
                    k3 = fmla(*shifted_src.get_unchecked(i + 3), coeff, k3);
                }
                dst[0] = k0;
                dst[1] = k1;
                dst[2] = k2;
                dst[3] = k3;
            }
        }

        let p = output.chunks_exact_mut(4).len() * 4;
        let output = output.chunks_exact_mut(4).into_remainder();

        for (x, dst) in output.iter_mut().enumerate() {
            unsafe {
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
