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
use crate::{BorderMode, IncompleteDwtExecutor};

#[inline]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn has_valid_avx() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

pub(crate) trait DwtFactory<T> {
    fn wavelet_2_taps(
        border_mode: BorderMode,
        dwt: &[T; 2],
    ) -> Box<dyn IncompleteDwtExecutor<T> + Send + Sync>;
    fn wavelet_4_taps(
        border_mode: BorderMode,
        dwt: &[T; 4],
    ) -> Box<dyn IncompleteDwtExecutor<T> + Send + Sync>;
    fn wavelet_6_taps(
        border_mode: BorderMode,
        dwt: &[T; 6],
    ) -> Box<dyn IncompleteDwtExecutor<T> + Send + Sync>;
    fn wavelet_8_taps(
        border_mode: BorderMode,
        dwt: &[T; 8],
    ) -> Box<dyn IncompleteDwtExecutor<T> + Send + Sync>;
    fn wavelet_n_taps(
        border_mode: BorderMode,
        dwt: &[T],
    ) -> Box<dyn IncompleteDwtExecutor<T> + Send + Sync>;
}

impl DwtFactory<f32> for f32 {
    fn wavelet_2_taps(
        border_mode: BorderMode,
        dwt: &[f32; 2],
    ) -> Box<dyn IncompleteDwtExecutor<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet2TapsF32;
            Box::new(NeonWavelet2TapsF32::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet2TapsF32;
                return Box::new(AvxWavelet2TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet2taps::Wavelet2Taps;
            Box::new(Wavelet2Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_4_taps(
        border_mode: BorderMode,
        dwt: &[f32; 4],
    ) -> Box<dyn IncompleteDwtExecutor<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet4TapsF32;
            Box::new(NeonWavelet4TapsF32::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet4TapsF32;
                return Box::new(AvxWavelet4TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet4taps::Wavelet4Taps;
            Box::new(Wavelet4Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_6_taps(
        border_mode: BorderMode,
        dwt: &[f32; 6],
    ) -> Box<dyn IncompleteDwtExecutor<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet6TapsF32;
            Box::new(NeonWavelet6TapsF32::new(border_mode, dwt))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet6taps::Wavelet6Taps;
            Box::new(Wavelet6Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_8_taps(
        border_mode: BorderMode,
        dwt: &[f32; 8],
    ) -> Box<dyn IncompleteDwtExecutor<f32> + Send + Sync> {
        use crate::wavelet8taps::Wavelet8Taps;
        Box::new(Wavelet8Taps::new(border_mode, dwt))
    }

    fn wavelet_n_taps(
        border_mode: BorderMode,
        dwt: &[f32],
    ) -> Box<dyn IncompleteDwtExecutor<f32> + Send + Sync> {
        use crate::wavelet_n_taps::WaveletNTaps;
        Box::new(WaveletNTaps::new(border_mode, dwt))
    }
}

impl DwtFactory<f64> for f64 {
    fn wavelet_2_taps(
        border_mode: BorderMode,
        dwt: &[f64; 2],
    ) -> Box<dyn IncompleteDwtExecutor<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet2TapsF64;
            Box::new(NeonWavelet2TapsF64::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet2TapsF64;
                return Box::new(AvxWavelet2TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet2taps::Wavelet2Taps;
            Box::new(Wavelet2Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_4_taps(
        border_mode: BorderMode,
        dwt: &[f64; 4],
    ) -> Box<dyn IncompleteDwtExecutor<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet4TapsF64;
            Box::new(NeonWavelet4TapsF64::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet4TapsF64;
                return Box::new(AvxWavelet4TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet4taps::Wavelet4Taps;
            Box::new(Wavelet4Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_6_taps(
        border_mode: BorderMode,
        dwt: &[f64; 6],
    ) -> Box<dyn IncompleteDwtExecutor<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet6TapsF64;
            Box::new(NeonWavelet6TapsF64::new(border_mode, dwt))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet6taps::Wavelet6Taps;
            Box::new(Wavelet6Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_8_taps(
        border_mode: BorderMode,
        dwt: &[f64; 8],
    ) -> Box<dyn IncompleteDwtExecutor<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet8TapsF64;
            Box::new(NeonWavelet8TapsF64::new(border_mode, dwt))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet8taps::Wavelet8Taps;
            Box::new(Wavelet8Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_n_taps(
        border_mode: BorderMode,
        dwt: &[f64],
    ) -> Box<dyn IncompleteDwtExecutor<f64> + Send + Sync> {
        use crate::wavelet_n_taps::WaveletNTaps;
        Box::new(WaveletNTaps::new(border_mode, dwt))
    }
}
