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
use crate::{WaveletFilterProvider, WaveletSample};
use num_traits::AsPrimitive;
use std::borrow::Cow;

/// Vaidyanathan wavelet filter provider.
#[derive(Default, Copy, Clone, Debug)]
pub struct Vaidyanathan {}

impl<T: WaveletSample> WaveletFilterProvider<T> for Vaidyanathan
where
    f64: AsPrimitive<T>,
{
    fn get_wavelet(&self) -> Cow<'_, [T]> {
        static WAVELET: [f64; 24] = [
            -0.000062906118,
            0.000343631905,
            -0.000453956620,
            -0.000944897136,
            0.002843834547,
            0.000708137504,
            -0.008839103409,
            0.003153847056,
            0.019687215010,
            -0.014853448005,
            -0.035470398607,
            0.038742619293,
            0.055892523691,
            -0.077709750902,
            -0.083928884366,
            0.131971661417,
            0.135084227129,
            -0.194450471766,
            -0.263494802488,
            0.201612161775,
            0.635601059872,
            0.572797793211,
            0.250184129505,
            0.045799334111,
        ];
        Cow::Owned(WAVELET.iter().map(|x| x.as_()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[test]
    fn test_vaidyanathan() {
        let r = Vaidyanathan::default();
        let wv: Cow<[f64]> = r.get_wavelet();
        assert!(
            wv.len().is_multiple_of(2),
            "Assertion failed for daubechies {:?} with size {}",
            Vaidyanathan::default(),
            wv.len()
        );
    }
}
