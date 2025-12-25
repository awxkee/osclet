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
use std::sync::Arc;

#[inline]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
pub(crate) fn has_valid_avx() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[inline]
#[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
pub(crate) fn has_valid_sse() -> bool {
    std::arch::is_x86_feature_detected!("sse4.2")
}

pub(crate) trait DwtFactory<T> {
    fn wavelet_2_taps(
        border_mode: BorderMode,
        dwt: &[T; 2],
    ) -> Arc<dyn IncompleteDwtExecutor<T> + Send + Sync>;
    fn wavelet_4_taps(
        border_mode: BorderMode,
        dwt: &[T; 4],
    ) -> Arc<dyn IncompleteDwtExecutor<T> + Send + Sync>;
    fn wavelet_6_taps(
        border_mode: BorderMode,
        dwt: &[T; 6],
    ) -> Arc<dyn IncompleteDwtExecutor<T> + Send + Sync>;
    fn wavelet_8_taps(
        border_mode: BorderMode,
        dwt: &[T; 8],
    ) -> Arc<dyn IncompleteDwtExecutor<T> + Send + Sync>;
    fn wavelet_10_taps(
        border_mode: BorderMode,
        dwt: &[T; 10],
    ) -> Arc<dyn IncompleteDwtExecutor<T> + Send + Sync>;
    fn wavelet_12_taps(
        border_mode: BorderMode,
        dwt: &[T; 12],
    ) -> Arc<dyn IncompleteDwtExecutor<T> + Send + Sync>;
    fn wavelet_16_taps(
        border_mode: BorderMode,
        dwt: &[T; 16],
    ) -> Arc<dyn IncompleteDwtExecutor<T> + Send + Sync>;
    fn wavelet_n_taps(
        border_mode: BorderMode,
        dwt: &[T],
    ) -> Arc<dyn IncompleteDwtExecutor<T> + Send + Sync>;
}

impl DwtFactory<f32> for f32 {
    fn wavelet_2_taps(
        border_mode: BorderMode,
        dwt: &[f32; 2],
    ) -> Arc<dyn IncompleteDwtExecutor<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet2TapsF32;
            Arc::new(NeonWavelet2TapsF32::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet2TapsF32;
                return Arc::new(AvxWavelet2TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWavelet2TapsF32;
                return Arc::new(SseWavelet2TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet2taps::Wavelet2Taps;
            Arc::new(Wavelet2Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_4_taps(
        border_mode: BorderMode,
        dwt: &[f32; 4],
    ) -> Arc<dyn IncompleteDwtExecutor<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet4TapsF32;
            Arc::new(NeonWavelet4TapsF32::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet4TapsF32;
                return Arc::new(AvxWavelet4TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWavelet4TapsF32;
                return Arc::new(SseWavelet4TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet4taps::Wavelet4Taps;
            Arc::new(Wavelet4Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_6_taps(
        border_mode: BorderMode,
        dwt: &[f32; 6],
    ) -> Arc<dyn IncompleteDwtExecutor<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet6TapsF32;
            Arc::new(NeonWavelet6TapsF32::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet6TapsF32;
                return Arc::new(AvxWavelet6TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWavelet6TapsF32;
                return Arc::new(SseWavelet6TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet6taps::Wavelet6Taps;
            Arc::new(Wavelet6Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_8_taps(
        border_mode: BorderMode,
        dwt: &[f32; 8],
    ) -> Arc<dyn IncompleteDwtExecutor<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet8TapsF32;
            Arc::new(NeonWavelet8TapsF32::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet8TapsF32;
                return Arc::new(AvxWavelet8TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWavelet8TapsF32;
                return Arc::new(SseWavelet8TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet8taps::Wavelet8Taps;
            Arc::new(Wavelet8Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_10_taps(
        border_mode: BorderMode,
        dwt: &[f32; 10],
    ) -> Arc<dyn IncompleteDwtExecutor<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet10TapsF32;
            Arc::new(NeonWavelet10TapsF32::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet10TapsF32;
                return Arc::new(AvxWavelet10TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWavelet10TapsF32;
                return Arc::new(SseWavelet10TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet10taps::Wavelet10Taps;
            Arc::new(Wavelet10Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_12_taps(
        border_mode: BorderMode,
        dwt: &[f32; 12],
    ) -> Arc<dyn IncompleteDwtExecutor<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet12TapsF32;
            Arc::new(NeonWavelet12TapsF32::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet12TapsF32;
                return Arc::new(AvxWavelet12TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWavelet12TapsF32;
                return Arc::new(SseWavelet12TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet12taps::Wavelet12Taps;
            Arc::new(Wavelet12Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_16_taps(
        border_mode: BorderMode,
        dwt: &[f32; 16],
    ) -> Arc<dyn IncompleteDwtExecutor<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet16TapsF32;
            Arc::new(NeonWavelet16TapsF32::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet16TapsF32;
                return Arc::new(AvxWavelet16TapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWaveletNTapsF32;
                return Arc::new(SseWaveletNTapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet16taps::Wavelet16Taps;
            Arc::new(Wavelet16Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_n_taps(
        border_mode: BorderMode,
        dwt: &[f32],
    ) -> Arc<dyn IncompleteDwtExecutor<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWaveletNTapsF32;
            Arc::new(NeonWaveletNTapsF32::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWaveletNTapsF32;
                return Arc::new(AvxWaveletNTapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWaveletNTapsF32;
                return Arc::new(SseWaveletNTapsF32::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet_n_taps::WaveletNTaps;
            Arc::new(WaveletNTaps::new(border_mode, dwt))
        }
    }
}

impl DwtFactory<f64> for f64 {
    fn wavelet_2_taps(
        border_mode: BorderMode,
        dwt: &[f64; 2],
    ) -> Arc<dyn IncompleteDwtExecutor<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet2TapsF64;
            Arc::new(NeonWavelet2TapsF64::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet2TapsF64;
                return Arc::new(AvxWavelet2TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWavelet2TapsF64;
                return Arc::new(SseWavelet2TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet2taps::Wavelet2Taps;
            Arc::new(Wavelet2Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_4_taps(
        border_mode: BorderMode,
        dwt: &[f64; 4],
    ) -> Arc<dyn IncompleteDwtExecutor<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet4TapsF64;
            Arc::new(NeonWavelet4TapsF64::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet4TapsF64;
                return Arc::new(AvxWavelet4TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWavelet4TapsF64;
                return Arc::new(SseWavelet4TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet4taps::Wavelet4Taps;
            Arc::new(Wavelet4Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_6_taps(
        border_mode: BorderMode,
        dwt: &[f64; 6],
    ) -> Arc<dyn IncompleteDwtExecutor<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet6TapsF64;
            Arc::new(NeonWavelet6TapsF64::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet6TapsF64;
                return Arc::new(AvxWavelet6TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWavelet6TapsF64;
                return Arc::new(SseWavelet6TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet6taps::Wavelet6Taps;
            Arc::new(Wavelet6Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_8_taps(
        border_mode: BorderMode,
        dwt: &[f64; 8],
    ) -> Arc<dyn IncompleteDwtExecutor<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet8TapsF64;
            Arc::new(NeonWavelet8TapsF64::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet8TapsF64;
                return Arc::new(AvxWavelet8TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWavelet8TapsF64;
                return Arc::new(SseWavelet8TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet8taps::Wavelet8Taps;
            Arc::new(Wavelet8Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_10_taps(
        border_mode: BorderMode,
        dwt: &[f64; 10],
    ) -> Arc<dyn IncompleteDwtExecutor<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet10TapsF64;
            Arc::new(NeonWavelet10TapsF64::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet10TapsF64;
                return Arc::new(AvxWavelet10TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWavelet10TapsF64;
                return Arc::new(SseWavelet10TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet10taps::Wavelet10Taps;
            Arc::new(Wavelet10Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_12_taps(
        border_mode: BorderMode,
        dwt: &[f64; 12],
    ) -> Arc<dyn IncompleteDwtExecutor<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet12TapsF64;
            Arc::new(NeonWavelet12TapsF64::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet12TapsF64;
                return Arc::new(AvxWavelet12TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWavelet12TapsF64;
                return Arc::new(SseWavelet12TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet12taps::Wavelet12Taps;
            Arc::new(Wavelet12Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_16_taps(
        border_mode: BorderMode,
        dwt: &[f64; 16],
    ) -> Arc<dyn IncompleteDwtExecutor<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWavelet16TapsF64;
            Arc::new(NeonWavelet16TapsF64::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWavelet16TapsF64;
                return Arc::new(AvxWavelet16TapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWaveletNTapsF64;
                return Arc::new(SseWaveletNTapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet16taps::Wavelet16Taps;
            Arc::new(Wavelet16Taps::new(border_mode, dwt))
        }
    }

    fn wavelet_n_taps(
        border_mode: BorderMode,
        dwt: &[f64],
    ) -> Arc<dyn IncompleteDwtExecutor<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonWaveletNTapsF64;
            Arc::new(NeonWaveletNTapsF64::new(border_mode, dwt))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxWaveletNTapsF64;
                return Arc::new(AvxWaveletNTapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"), feature = "sse"))]
        {
            if has_valid_sse() {
                use crate::sse::SseWaveletNTapsF64;
                return Arc::new(SseWaveletNTapsF64::new(border_mode, dwt));
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::wavelet_n_taps::WaveletNTaps;
            Arc::new(WaveletNTaps::new(border_mode, dwt))
        }
    }
}
