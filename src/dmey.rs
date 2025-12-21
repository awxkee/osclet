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
        static MEYER: [f64; 102] = [
            -0.000001509740857,
            0.000001278766757,
            0.000000449585560,
            -0.000002096568870,
            0.000001723223554,
            0.000000698082276,
            -0.000002879408033,
            0.000002383148395,
            0.000000982515602,
            -0.000004217789186,
            0.000003353501538,
            0.000001674721859,
            -0.000006034501342,
            0.000004837555802,
            0.000002402288023,
            -0.000009556309846,
            0.000007216527695,
            0.000004849078300,
            -0.000014206928581,
            0.000010503914271,
            0.000006187580298,
            -0.000024438005846,
            0.000020106387691,
            0.000014993523600,
            -0.000046428764284,
            0.000032341311914,
            0.000037409665760,
            -0.000102779005085,
            0.000024461956845,
            0.000149713515389,
            -0.000075592870255,
            -0.000139913148217,
            -0.000093512893880,
            0.000161189819725,
            0.000859500213762,
            -0.000578185795273,
            -0.002702168733939,
            0.002194775336459,
            0.006045510596456,
            -0.006386728618548,
            -0.011044641900539,
            0.015250913158586,
            0.017403888210177,
            -0.032094063354505,
            -0.024321783959519,
            0.063667300884468,
            0.030621243943425,
            -0.132696615358862,
            -0.035048287390595,
            0.444095030766529,
            0.743751004903787,
            0.444095030766529,
            -0.035048287390595,
            -0.132696615358862,
            0.030621243943425,
            0.063667300884468,
            -0.024321783959519,
            -0.032094063354505,
            0.017403888210177,
            0.015250913158586,
            -0.011044641900539,
            -0.006386728618548,
            0.006045510596456,
            0.002194775336459,
            -0.002702168733939,
            -0.000578185795273,
            0.000859500213762,
            0.000161189819725,
            -0.000093512893880,
            -0.000139913148217,
            -0.000075592870255,
            0.000149713515389,
            0.000024461956845,
            -0.000102779005085,
            0.000037409665760,
            0.000032341311914,
            -0.000046428764284,
            0.000014993523600,
            0.000020106387691,
            -0.000024438005846,
            0.000006187580298,
            0.000010503914271,
            -0.000014206928581,
            0.000004849078300,
            0.000007216527695,
            -0.000009556309846,
            0.000002402288023,
            0.000004837555802,
            -0.000006034501342,
            0.000001674721859,
            0.000003353501538,
            -0.000004217789186,
            0.000000982515602,
            0.000002383148395,
            -0.000002879408033,
            0.000000698082276,
            0.000001723223554,
            -0.000002096568870,
            0.000000449585560,
            0.000001278766757,
            -0.000001509740857,
            0.0,
        ];
        Cow::Owned(MEYER.iter().map(|x| x.as_()).collect())
    }
}
