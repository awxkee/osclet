/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
#![allow(clippy::approx_constant)]
use crate::WaveletFilterProvider;
use num_traits::AsPrimitive;

/// Represents the family of **biorthogonal wavelets** used in discrete wavelet transforms (DWT).
///
/// * **N** – number of vanishing moments of the decomposition (analysis) low-pass filter
/// * **M** – number of vanishing moments of the reconstruction (synthesis) low-pass filter
///
/// # Common Notes
/// * `Biorthogonal1_3` and `Biorthogonal1_5` are short filters used for faster transforms.
/// * `Biorthogonal4_4` corresponds to the **Cohen–Daubechies–Feauveau (CDF 9/7)** wavelet.
/// * `Biorthogonal2_2` corresponds to the **CDF 5/3** wavelet used in lossless JPEG 2000.
///
/// # Variants
///
/// - `Biorthogonal1_0` – Trivial (Haar-like) basis
/// - `Biorthogonal1_1`, `Biorthogonal1_3`, `Biorthogonal1_5` – Asymmetric short filters
/// - `Biorthogonal2_0`, `Biorthogonal2_2`, `Biorthogonal2_4`, `Biorthogonal2_6`, `Biorthogonal2_8` – Family with two analysis vanishing moments
/// - `Biorthogonal3_0` … `Biorthogonal3_9` – Increasing synthesis smoothness
/// - `Biorthogonal4_0`, `Biorthogonal4_4` – Notably includes the 9/7 CDF wavelet
/// - `Biorthogonal5_0`, `Biorthogonal5_5`, `Biorthogonal6_0`, `Biorthogonal6_8` – Higher-order smooth wavelets
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum BiorthogonalFamily {
    Biorthogonal1_0,
    Biorthogonal1_1,
    Biorthogonal1_3,
    Biorthogonal1_5,
    Biorthogonal2_0,
    Biorthogonal2_2,
    Biorthogonal2_4,
    Biorthogonal2_6,
    Biorthogonal2_8,
    Biorthogonal3_0,
    Biorthogonal3_1,
    Biorthogonal3_3,
    Biorthogonal3_5,
    Biorthogonal3_7,
    Biorthogonal3_9,
    Biorthogonal4_0,
    Biorthogonal4_4,
    Biorthogonal5_0,
    Biorthogonal5_5,
    Biorthogonal6_0,
    Biorthogonal6_8,
}

impl BiorthogonalFamily {
    pub(crate) fn get_wavelet_impl(self) -> &'static [f64] {
        match self {
            BiorthogonalFamily::Biorthogonal1_0 => [
                0.0,
                0.0,
                0.0,
                0.0,
                0.70710678118654752440084436210,
                0.70710678118654752440084436210,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal1_1 => [
                0.70710678118654752440084436210,
                0.70710678118654752440084436210,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal1_3 => [
                -0.0883883476483184405501055452631,
                0.0883883476483184405501055452631,
                0.70710678118654752440084436210,
                0.70710678118654752440084436210,
                0.0883883476483184405501055452631,
                -0.0883883476483184405501055452631,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal1_5 => [
                0.0165728151840597076031447897368,
                -0.0165728151840597076031447897368,
                -0.1215339780164378557563951247368,
                0.1215339780164378557563951247368,
                0.70710678118654752440084436210,
                0.70710678118654752440084436210,
                0.1215339780164378557563951247368,
                -0.1215339780164378557563951247368,
                -0.0165728151840597076031447897368,
                0.0165728151840597076031447897368,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal2_0 => [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3535533905932737622004221810524,
                0.7071067811865475244008443621048,
                0.3535533905932737622004221810524,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal2_2 => [
                -0.1767766952966368811002110905262,
                0.3535533905932737622004221810524,
                1.0606601717798212866012665431573,
                0.3535533905932737622004221810524,
                -0.1767766952966368811002110905262,
                0.0,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal2_4 => [
                0.0331456303681194152062895794737,
                -0.0662912607362388304125791589473,
                -0.1767766952966368811002110905262,
                0.4198446513295125926130013399998,
                0.9943689110435824561886873842099,
                0.4198446513295125926130013399998,
                -0.1767766952966368811002110905262,
                -0.0662912607362388304125791589473,
                0.0331456303681194152062895794737,
                0.0,
            ]
            .as_slice(),

            BiorthogonalFamily::Biorthogonal2_6 => [
                -0.0069053396600248781679769957237,
                0.0138106793200497563359539914474,
                0.0469563096881691715422435709210,
                -0.1077232986963880994204411332894,
                -0.1698713556366120029322340948025,
                0.4474660099696121052849093228945,
                0.9667475524034829435167794013152,
                0.4474660099696121052849093228945,
                -0.1698713556366120029322340948025,
                -0.1077232986963880994204411332894,
                0.0469563096881691715422435709210,
                0.0138106793200497563359539914474,
                -0.0069053396600248781679769957237,
                0.0,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal2_8 => [
                0.0015105430506304420992449678146,
                -0.0030210861012608841984899356291,
                -0.0129475118625466465649568669819,
                0.0289161098263541773284036695929,
                0.0529984818906909399392234421792,
                -0.1349130736077360572068505539514,
                -0.1638291834340902345352542235443,
                0.4625714404759165262773590010400,
                0.9516421218971785225243297231697,
                0.4625714404759165262773590010400,
                -0.1638291834340902345352542235443,
                -0.1349130736077360572068505539514,
                0.0529984818906909399392234421792,
                0.0289161098263541773284036695929,
                -0.0129475118625466465649568669819,
                -0.0030210861012608841984899356291,
                0.0015105430506304420992449678146,
                0.0,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal3_0 => [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1767766952966368811002110905262,
                0.5303300858899106433006332715786,
                0.5303300858899106433006332715786,
                0.1767766952966368811002110905262,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal3_1 => [
                -0.3535533905932737622004221810524,
                1.0606601717798212866012665431573,
                1.0606601717798212866012665431573,
                -0.3535533905932737622004221810524,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal3_3 => [
                0.0662912607362388304125791589473,
                -0.1988737822087164912377374768420,
                -0.1546796083845572709626847042104,
                0.9943689110435824561886873842099,
                0.9943689110435824561886873842099,
                -0.1546796083845572709626847042104,
                -0.1988737822087164912377374768420,
                0.0662912607362388304125791589473,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal3_5 => [
                -0.0138106793200497563359539914474,
                0.0414320379601492690078619743421,
                0.0524805814161890740766251675000,
                -0.2679271788089652729175074340788,
                -0.0718155324642587329469607555263,
                0.9667475524034829435167794013152,
                0.9667475524034829435167794013152,
                -0.0718155324642587329469607555263,
                -0.2679271788089652729175074340788,
                0.0524805814161890740766251675000,
                0.0414320379601492690078619743421,
                -0.0138106793200497563359539914474,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal3_7 => [
                0.0030210861012608841984899356291,
                -0.0090632583037826525954698068873,
                -0.0168317654213106405344439270765,
                0.0746639850740189951912512662623,
                0.0313329787073628846871956180962,
                -0.3011591259228349991008967259990,
                -0.0264992409453454699696117210896,
                0.9516421218971785225243297231697,
                0.9516421218971785225243297231697,
                -0.0264992409453454699696117210896,
                -0.3011591259228349991008967259990,
                0.0313329787073628846871956180962,
                0.0746639850740189951912512662623,
                -0.0168317654213106405344439270765,
                -0.0090632583037826525954698068873,
                0.0030210861012608841984899356291,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal3_9 => [
                -0.0006797443727836989446602355165,
                0.0020392331183510968339807065496,
                0.0050603192196119810324706421788,
                -0.0206189126411055346546938106687,
                -0.0141127879301758447558029850103,
                0.0991347824942321571990197448581,
                0.0123001362694193142367090236328,
                -0.3201919683607785695513833204624,
                0.0020500227115698857061181706055,
                0.9421257006782067372990864259380,
                0.9421257006782067372990864259380,
                0.0020500227115698857061181706055,
                -0.3201919683607785695513833204624,
                0.0123001362694193142367090236328,
                0.0991347824942321571990197448581,
                -0.0141127879301758447558029850103,
                -0.0206189126411055346546938106687,
                0.0050603192196119810324706421788,
                0.0020392331183510968339807065496,
                -0.0006797443727836989446602355165,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal4_0 => [
                0.0,
                -0.064538882628697058,
                -0.040689417609164058,
                0.41809227322161724,
                0.7884856164055829,
                0.41809227322161724,
                -0.040689417609164058,
                -0.064538882628697058,
                0.0,
                0.0,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal4_4 => [
                0.03782845550726404,
                -0.023849465019556843,
                -0.11062440441843718,
                0.37740285561283066,
                0.85269867900889385,
                0.37740285561283066,
                -0.11062440441843718,
                -0.023849465019556843,
                0.03782845550726404,
                0.0,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal5_0 => [
                0.013456709459118716,
                -0.0026949668801115071,
                -0.13670658466432914,
                -0.093504697400938863,
                0.47680326579848425,
                0.89950610974864842,
                0.47680326579848425,
                -0.093504697400938863,
                -0.13670658466432914,
                -0.0026949668801115071,
                0.013456709459118716,
                0.0,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal5_5 => [
                0.0,
                0.03968708834740544,
                0.0079481086372403219,
                -0.054463788468236907,
                0.34560528195603346,
                0.73666018142821055,
                0.34560528195603346,
                -0.054463788468236907,
                0.0079481086372403219,
                0.03968708834740544,
                0.0,
                0.0,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal6_0 => [
                0.0,
                0.0,
                0.0,
                0.014426282505624435,
                0.014467504896790148,
                -0.078722001062628819,
                -0.040367979030339923,
                0.41784910915027457,
                0.75890772945365415,
                0.41784910915027457,
                -0.040367979030339923,
                -0.078722001062628819,
                0.014467504896790148,
                0.014426282505624435,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            .as_slice(),
            BiorthogonalFamily::Biorthogonal6_8 => [
                0.0019088317364812906,
                -0.0019142861290887667,
                -0.016990639867602342,
                0.01193456527972926,
                0.04973290349094079,
                -0.077263173167204144,
                -0.09405920349573646,
                0.42079628460982682,
                0.82592299745840225,
                0.42079628460982682,
                -0.09405920349573646,
                -0.077263173167204144,
                0.04973290349094079,
                0.01193456527972926,
                -0.016990639867602342,
                -0.0019142861290887667,
                0.0019088317364812906,
                0.0,
            ]
            .as_slice(),
        }
    }
}

impl<T: Copy + 'static> WaveletFilterProvider<T> for BiorthogonalFamily
where
    f64: AsPrimitive<T>,
{
    fn get_wavelet(&self) -> Vec<T> {
        self.get_wavelet_impl().iter().map(|x| x.as_()).collect()
    }
}
