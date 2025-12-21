/*
 * Copyright (c) Radzivon Bartoshyk. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::WaveletSample;
use crate::fast_divide::{DividerIsize, RemEuclidFast};
use num_traits::AsPrimitive;
use std::fmt::{Display, Formatter};

#[repr(C)]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Default)]
/// Declares an edge handling mode
pub enum BorderMode {
    /// If kernel goes out of bounds it will be clipped to an edge and edge pixel replicated across filter
    Clamp,
    /// If filter goes out of bounds image will be replicated with rule `cdefgh|abcdefgh|abcdefg`
    #[default]
    Wrap,
    /// If filter goes out of bounds image will be replicated with rule `fedcba|abcdefgh|hgfedcb`
    Reflect,
    /// If filter goes out of bounds image will be replicated with rule `gfedcb|abcdefgh|gfedcba`
    Reflect101,
    /// If filter goes out of bounds image will be replicated with rule `000000|abcdefgh|000000`
    Zeros,
}

impl Display for BorderMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BorderMode::Clamp => f.write_str("Clamp"),
            BorderMode::Wrap => f.write_str("Wrap"),
            BorderMode::Reflect => f.write_str("Reflect"),
            BorderMode::Reflect101 => f.write_str("Reflect101"),
            BorderMode::Zeros => f.write_str("Zeros"),
        }
    }
}

impl BorderMode {
    #[inline]
    pub(crate) fn interpolate<T: WaveletSample>(
        self,
        arr: &[T],
        position: isize,
        start: isize,
        end: isize,
    ) -> T
    where
        f64: AsPrimitive<T>,
    {
        match self {
            BorderMode::Clamp => {
                let read_position = position.min(end - 1).max(start) as usize;
                unsafe { *arr.get_unchecked(read_position) }
            }
            BorderMode::Wrap => {
                let read_position = position.rem_euclid(end) as usize;
                unsafe { *arr.get_unchecked(read_position) }
            }
            BorderMode::Reflect => {
                let read_position = (end - position.rem_euclid(end) - 1) as usize;
                unsafe { *arr.get_unchecked(read_position) }
            }
            BorderMode::Reflect101 => {
                let read_position = (end - position.rem_euclid(end - 1) - 1) as usize;
                unsafe { *arr.get_unchecked(read_position) }
            }
            BorderMode::Zeros => 0f64.as_(),
        }
    }
}

pub(crate) struct BorderInterpolation {
    divider: DividerIsize,
    mode: BorderMode,
    start: isize,
    end: isize,
}

impl BorderInterpolation {
    pub(crate) fn new(mode: BorderMode, start: isize, end: isize) -> Self {
        let divider = match mode {
            BorderMode::Clamp => DividerIsize::new(1),
            BorderMode::Wrap => DividerIsize::new(end),
            BorderMode::Reflect => DividerIsize::new(end),
            BorderMode::Reflect101 => DividerIsize::new(end),
            BorderMode::Zeros => DividerIsize::new(1),
        };
        Self {
            divider,
            mode,
            start,
            end,
        }
    }

    #[inline(always)]
    pub(crate) fn interpolate<T: WaveletSample>(&self, arr: &[T], position: isize) -> T
    where
        f64: AsPrimitive<T>,
    {
        match self.mode {
            BorderMode::Clamp => {
                let read_position = position.min(self.end - 1).max(self.start) as usize;
                unsafe { *arr.get_unchecked(read_position) }
            }
            BorderMode::Wrap => {
                let read_position = position.rem_euclid(self.end) as usize;
                unsafe { *arr.get_unchecked(read_position) }
            }
            BorderMode::Reflect => {
                let read_position =
                    (self.end - position.rem_euclid_fast(&self.divider) - 1) as usize;
                unsafe { *arr.get_unchecked(read_position) }
            }
            BorderMode::Reflect101 => {
                let read_position =
                    (self.end - position.rem_euclid_fast(&self.divider) - 1) as usize;
                unsafe { *arr.get_unchecked(read_position) }
            }
            BorderMode::Zeros => 0f64.as_(),
        }
    }
}
