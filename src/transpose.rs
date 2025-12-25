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
#![allow(clippy::too_many_arguments, clippy::only_used_in_recursion)]
use std::marker::PhantomData;
use std::sync::Arc;

pub(crate) trait TransposeExecutor<T> {
    fn transpose(&self, input: &[T], output: &mut [T], width: usize, height: usize);
}

#[inline(always)]
fn transpose_block<V: Copy>(
    input: &[V],
    input_stride: usize,
    output: &mut [V],
    output_stride: usize,
    start_x: usize,
    start_y: usize,
    block_width: usize,
    block_height: usize,
) {
    for inner_x in 0..block_width {
        for inner_y in 0..block_height {
            let x = start_x + inner_x;
            let y = start_y + inner_y;

            let input_y = y;
            let output_x = x;

            let input_index = x + input_y * input_stride;
            let output_index = y + output_x * output_stride;

            unsafe {
                *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
            }
        }
    }
}

#[inline(always)]
fn transpose_block_segmented<T: Copy, const BLOCK_SIZE_X: usize, const BLOCK_SIZE_Y: usize>(
    input: &[T],
    input_stride: usize,
    output: &mut [T],
    output_stride: usize,
    start_x: usize,
    start_y: usize,
) {
    let height_per_div = BLOCK_SIZE_Y / 4;
    for subblock in 0..4 {
        for inner_x in 0..BLOCK_SIZE_X {
            for inner_y in 0..height_per_div {
                let x = start_x + inner_x;
                let y = start_y + inner_y + subblock * height_per_div;

                let input_y = y;
                let output_x = x;

                let input_index = x + input_y * input_stride;
                let output_index = y + output_x * output_stride;

                unsafe {
                    *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
                }
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct TransposeArbitrary<V: Copy + Default> {
    pub(crate) phantom_data: PhantomData<V>,
}

fn transpose_tiny<T: Copy>(input: &[T], output: &mut [T], width: usize, height: usize) {
    for x in 0..width {
        for y in 0..height {
            let input_index = x + y * width;
            let output_index = y + x * height;

            unsafe {
                *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
            }
        }
    }
}

impl<V: Copy + Default> TransposeExecutor<V> for TransposeArbitrary<V> {
    fn transpose(&self, input: &[V], output: &mut [V], width: usize, height: usize) {
        if width < 31 && height < 31 {
            transpose_tiny(input, output, width, height);
            return;
        }
        transpose_arbitrary_impl(
            input, width, output, height, 0, height, 0, width, width, height,
        )
    }
}

fn transpose_arbitrary_impl<V: Copy>(
    input: &[V],
    input_stride: usize,
    output: &mut [V],
    output_stride: usize,
    start_y: usize,
    end_y: usize,
    start_x: usize,
    end_x: usize,
    width: usize,
    height: usize,
) {
    let length_y = end_y - start_y;
    let length_x = end_x - start_x;
    const LIMIT: usize = 128;
    const BLOCK_SIZE: usize = 16;
    if (length_y <= LIMIT && length_x <= LIMIT) || length_y <= 2 || length_x <= 2 {
        let x_block_count = length_x / BLOCK_SIZE;
        let y_block_count = length_y / BLOCK_SIZE;

        let remainder_x = length_x - x_block_count * BLOCK_SIZE;
        let remainder_y = length_y - y_block_count * BLOCK_SIZE;

        for y_block in 0..y_block_count {
            for x_block in 0..x_block_count {
                transpose_block_segmented::<V, BLOCK_SIZE, BLOCK_SIZE>(
                    input,
                    input_stride,
                    output,
                    output_stride,
                    start_x + x_block * BLOCK_SIZE,
                    start_y + y_block * BLOCK_SIZE,
                );
            }

            if remainder_x > 0 {
                transpose_block::<V>(
                    input,
                    input_stride,
                    output,
                    output_stride,
                    start_x + x_block_count * BLOCK_SIZE,
                    start_y + y_block * BLOCK_SIZE,
                    remainder_x,
                    BLOCK_SIZE,
                );
            }
        }

        if remainder_y > 0 {
            for x_block in 0..x_block_count {
                transpose_block::<V>(
                    input,
                    input_stride,
                    output,
                    output_stride,
                    start_x + x_block * BLOCK_SIZE,
                    start_y + y_block_count * BLOCK_SIZE,
                    BLOCK_SIZE,
                    remainder_y,
                );
            }

            if remainder_x > 0 {
                transpose_block::<V>(
                    input,
                    input_stride,
                    output,
                    output_stride,
                    start_x + x_block_count * BLOCK_SIZE,
                    start_y + y_block_count * BLOCK_SIZE,
                    remainder_x,
                    remainder_y,
                );
            }
        }
    } else if length_y >= length_x {
        transpose_arbitrary_impl::<V>(
            input,
            input_stride,
            output,
            output_stride,
            start_y,
            start_y + (length_y / 2),
            start_x,
            end_x,
            width,
            height,
        );
        transpose_arbitrary_impl::<V>(
            input,
            input_stride,
            output,
            output_stride,
            start_y + (length_y / 2),
            end_y,
            start_x,
            end_x,
            width,
            height,
        );
    } else {
        transpose_arbitrary_impl::<V>(
            input,
            input_stride,
            output,
            output_stride,
            start_y,
            end_y,
            start_x,
            start_x + (length_x / 2),
            width,
            height,
        );
        transpose_arbitrary_impl::<V>(
            input,
            input_stride,
            output,
            output_stride,
            start_y,
            end_y,
            start_x + (length_x / 2),
            end_x,
            width,
            height,
        );
    }
}

pub(crate) trait TransposeFactory {
    fn transpositor() -> Arc<dyn TransposeExecutor<Self> + Send + Sync>;
}

impl TransposeFactory for f32 {
    fn transpositor() -> Arc<dyn TransposeExecutor<Self> + Send + Sync> {
        Arc::new(TransposeArbitrary::default())
    }
}

impl TransposeFactory for f64 {
    fn transpositor() -> Arc<dyn TransposeExecutor<Self> + Send + Sync> {
        Arc::new(TransposeArbitrary::default())
    }
}

impl TransposeFactory for i16 {
    fn transpositor() -> Arc<dyn TransposeExecutor<Self> + Send + Sync> {
        Arc::new(TransposeArbitrary::default())
    }
}
