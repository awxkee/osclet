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

#[derive(Clone, Debug)]
pub enum OscletError {
    OutOfMemory(usize),
    ApproxDetailsSize(usize),
    ApproxDetailsNotMatches(usize, usize),
    InputDivisibleBy(usize),
    Overflow,
    OutputSizeIsTooSmall(usize, usize),
    MinFilterSize(usize, usize),
    InOutSizesMismatch(usize, usize),
    ZeroOrOddSizedWavelet,
    MisconfiguredFilterCenter(usize, usize),
    BufferWasTooSmallForLevel,
    ApproxSizeNotMatches(usize, usize),
    DetailsSizeNotMatches(usize, usize),
}

impl Error for OscletError {}

impl std::fmt::Display for OscletError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OscletError::OutOfMemory(length) => {
                f.write_fmt(format_args!("Cannot allocate {length} bytes to vector",))
            }
            OscletError::ApproxDetailsSize(size) => {
                f.write_fmt(format_args!("Approximation and details expected to be half of original signal length but it was {}", size))
            }
            OscletError::InputDivisibleBy(size) => {
                f.write_fmt(format_args!("Input must be divisible by {}", size))
            }
            OscletError::ApproxDetailsNotMatches(expected, actual) => {
                f.write_fmt(format_args!("Approx and details must match, but they don't {}x{}", expected, actual))
            }
            OscletError::Overflow => {
                f.write_str("Overflow is happened")
            }
            OscletError::OutputSizeIsTooSmall(were_length, min_length) => {
                f.write_fmt(format_args!("Output size should be {min_length}, but it was {were_length}"))
            }
            OscletError::MinFilterSize(input_size, filter_size) => {
                f.write_fmt(format_args!("Input size {input_size} can't be less than {filter_size}"))
            }
            OscletError::InOutSizesMismatch(input_size, output_size) => {
                f.write_fmt(format_args!("Input size {input_size} does not match output size {output_size}"))
            }
            OscletError::ZeroOrOddSizedWavelet => {
                f.write_fmt(format_args!("Zero or odd sized wavelet"))
            }
            OscletError::MisconfiguredFilterCenter(filter_offset, filter_size) => {
                f.write_fmt(format_args!("Filter center {filter_offset} was larger than filter size {filter_size}"))
            }
            OscletError::BufferWasTooSmallForLevel => {
                f.write_fmt(format_args!("Buffer was too small for provided level"))
            }
            OscletError::ApproxSizeNotMatches(current_size, required_size) => {
                f.write_fmt(format_args!("Approximate size {current_size} does not match required size {required_size}"))
            }
            OscletError::DetailsSizeNotMatches(current_size, required_size) => {
                f.write_fmt(format_args!("Details size {current_size} does not match required size {required_size}"))
            }
        }
    }
}

macro_rules! try_vec {
    () => {
        Vec::new()
    };
    ($elem:expr; $n:expr) => {{
        let mut v = Vec::new();
        v.try_reserve_exact($n)
            .map_err(|_| crate::err::OscletError::OutOfMemory($n))?;
        v.resize($n, $elem);
        v
    }};
}

use std::error::Error;
use std::fmt::Formatter;
pub(crate) use try_vec;
