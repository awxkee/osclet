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

/// Represents the 18-tap **Beylkin wavelet** filter.
///
/// The Beylkin wavelet is an orthonormal, compactly supported wavelet
/// with good smoothness and frequency localization. It is commonly used
/// in signal processing and numerical methods where higher-order accuracy
/// and minimal phase distortion are important.
#[derive(Default, Copy, Clone, Debug)]
pub struct Beylkin {}

impl<T: WaveletSample> WaveletFilterProvider<T> for Beylkin
where
    f64: AsPrimitive<T>,
{
    fn get_wavelet(&self) -> Cow<'_, [T]> {
        static WAVELET: [f64; 18] = [
            0.09930576537463466338982,
            0.42421536081355360970052,
            0.69982521405759149038391,
            0.44971825114962302665023,
            -0.11092759834796520505404,
            -0.26449723144646793113069,
            0.02690030880435691408330,
            0.15553873187754627425337,
            -0.01752074626702530482668,
            -0.08854363062309189667931,
            0.01967986604398002273822,
            0.04291638727421198802261,
            -0.01746040869616576162381,
            -0.01436580796882066675955,
            0.01004041184439451210293,
            0.00148423478246081098748,
            -0.00273603162678399313527,
            0.00064048532943888552285,
        ];
        Cow::Owned(WAVELET.iter().map(|x| x.as_()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    #[test]
    fn test_beylkin() {
        let r = Beylkin::default();
        let wv: Cow<[f64]> = r.get_wavelet();
        assert!(
            wv.len().is_multiple_of(2),
            "Assertion failed for daubechies {:?} with size {}",
            Beylkin::default(),
            wv.len()
        );
    }
}
