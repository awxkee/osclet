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
use crate::WaveletFilterProvider;
use num_traits::AsPrimitive;
use std::borrow::Cow;

/// A container struct representing the **Discrete Meyer Wavelet** (also known as the
/// **Meyer-Type Wavelet** or **Orthogonal Wavelet of Meyer and Coifman**).
///
/// Unlike Daubechies wavelets, the Meyer wavelet has **infinite support** in the
/// time domain, but its coefficients decay very rapidly (exponentially fast),
/// making it highly suitable for applications where spectral fidelity is crucial,
/// such as signal analysis and spectral methods.
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct DiscreteMeyerWavelet {}

impl<T: Copy + 'static> WaveletFilterProvider<T> for DiscreteMeyerWavelet
where
    f64: AsPrimitive<T>,
{
    fn get_wavelet(&self) -> Cow<'_, [T]> {
        static MEYER: [f64; 62] = [
            -1.0099999569414229e-012,
            8.519459636796214e-009,
            -1.111944952595278e-008,
            -1.0798819539621958e-008,
            6.0669757413511352e-008,
            -1.0866516536735883e-007,
            8.2006806503864813e-008,
            1.1783004497663934e-007,
            -5.5063405652522782e-007,
            1.1307947017916706e-006,
            -1.4895492164971559e-006,
            7.367572885903746e-007,
            3.2054419133447798e-006,
            -1.6312699734552807e-005,
            6.5543059305751491e-005,
            -0.00060115023435160925,
            -0.002704672124643725,
            0.0022025341009110021,
            0.006045814097323304,
            -0.0063877183184971563,
            -0.011061496392513451,
            0.015270015130934803,
            0.017423434103729693,
            -0.032130793990211758,
            -0.024348745906078023,
            0.063739024322801596,
            0.030655091960824263,
            -0.13284520043622938,
            -0.035087555656258346,
            0.44459300275757724,
            0.74458559231880628,
            0.44459300275757724,
            -0.035087555656258346,
            -0.13284520043622938,
            0.030655091960824263,
            0.063739024322801596,
            -0.024348745906078023,
            -0.032130793990211758,
            0.017423434103729693,
            0.015270015130934803,
            -0.011061496392513451,
            -0.0063877183184971563,
            0.006045814097323304,
            0.0022025341009110021,
            -0.002704672124643725,
            -0.00060115023435160925,
            6.5543059305751491e-005,
            -1.6312699734552807e-005,
            3.2054419133447798e-006,
            7.367572885903746e-007,
            -1.4895492164971559e-006,
            1.1307947017916706e-006,
            -5.5063405652522782e-007,
            1.1783004497663934e-007,
            8.2006806503864813e-008,
            -1.0866516536735883e-007,
            6.0669757413511352e-008,
            -1.0798819539621958e-008,
            -1.111944952595278e-008,
            8.519459636796214e-009,
            -1.0099999569414229e-012,
            0.0,
        ];
        Cow::Owned(MEYER.iter().map(|x| x.as_()).collect())
    }
}
