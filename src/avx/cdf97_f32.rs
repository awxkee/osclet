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
use crate::err::{OscletError, try_vec};
use crate::{
    Dwt, DwtExecutor, DwtForwardExecutor, DwtInverseExecutor, IncompleteDwtExecutor, MultiDwt,
};

const ALPHA: f32 = -1.5861343420693648;
const BETA: f32 = -0.0529801185718856;
const GAMMA: f32 = 0.8829110755411875;
const DELTA: f32 = 0.4435068520511142;
const K: f32 = 1.1496043988602418;
const INV_K: f32 = 0.86986445162478079;

#[derive(Default)]
pub(crate) struct AvxCdf97F32 {}

#[target_feature(enable = "avx2", enable = "fma")]
fn dwt97_forward_update_even(approx: &mut [f32], details: &mut [f32], c: f32) {
    approx[0] = f32::mul_add(details[0] + details[0], c, approx[0]);

    let details_next = &details[1..];
    let approx_len = approx.len();
    let approx_k = &mut approx[1..approx_len - 1];

    for ((dst, detail), detail_next) in approx_k
        .iter_mut()
        .zip(details.iter())
        .zip(details_next.iter())
    {
        let d_left = *detail;
        let d_right = *detail_next;
        *dst = f32::mul_add(d_left + d_right, c, *dst);
    }

    if approx.len() > 1 {
        let i = approx.len() - 1;
        let d_left = details[i - 1];
        let d_right = *details.last().unwrap();
        approx[i] = f32::mul_add(d_left + d_right, c, approx[i]);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
fn dwt97_forward_update_odd(approx: &mut [f32], details: &mut [f32], c: f32) {
    let approx_next = &approx[1..];

    for ((dst, &src_left), &src_right) in details
        .iter_mut()
        .zip(approx.iter())
        .zip(approx_next.iter())
    {
        *dst = f32::mul_add(src_left + src_right, c, *dst);
    }

    if approx.len() == details.len() {
        let i = details.len();
        let src_left = approx[i - 1];
        // at boundary, mirror last sample
        let src_right = src_left;
        details[i - 1] = f32::mul_add(src_left + src_right, c, details[i - 1]);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
fn dwt97_scale(approx: &mut [f32], details: &mut [f32], k: f32, inv_k: f32) {
    for (a_dst, d_dst) in approx.iter_mut().zip(details.iter_mut()) {
        *a_dst *= k;
        *d_dst *= inv_k;
    }
    if approx.len() < details.len() {
        let det_left = &mut details[approx.len()..];
        for x in det_left.iter_mut() {
            *x *= inv_k;
        }
    } else if approx.len() > details.len() {
        let app_left = &mut approx[details.len()..];
        for x in app_left.iter_mut() {
            *x *= k;
        }
    }
}

impl DwtForwardExecutor<f32> for AvxCdf97F32 {
    /// Perform the forward CDF 9/7 DWT (lifting scheme) on a 1D signal.
    /// Splits the signal into approximation (low-pass) and detail (high-pass) coefficients.
    fn execute_forward(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_forward_impl(input, approx, details) }
    }
}

impl AvxCdf97F32 {
    /// Perform the forward CDF 9/7 DWT (lifting scheme) on a 1D signal.
    /// Splits the signal into approximation (low-pass) and detail (high-pass) coefficients.
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_forward_impl(
        &self,
        input: &[f32],
        approx: &mut [f32],
        details: &mut [f32],
    ) -> Result<(), OscletError> {
        let n = input.len();
        if n < 4 {
            return Err(OscletError::MinFilterSize(n, 4));
        }
        if approx.len() != n.div_ceil(2) {
            return Err(OscletError::ApproxSizeNotMatches(approx.len(), n));
        }
        if details.len() != n / 2 {
            return Err(OscletError::ApproxSizeNotMatches(details.len(), n));
        }

        let signal_n = &input[1..];
        let signal_n2 = &input[2..];

        // First lifting step
        details[0] = f32::mul_add(input[0] + input[1], ALPHA, input[0]);

        for (((dst, &s_previous), &s_current), &s_next) in details
            .iter_mut()
            .zip(input.iter().step_by(2))
            .zip(signal_n.iter().step_by(2))
            .zip(signal_n2.iter().step_by(2))
        {
            let left = s_previous;
            let right = s_next;
            *dst = f32::mul_add(left + right, ALPHA, s_current);
        }

        // Handle last odd index if n is even (boundary)
        if n.is_multiple_of(2) {
            let i = n - 1;
            let left = input[i - 1];
            *details.last_mut().unwrap() = f32::mul_add(left + left, ALPHA, input[i]);
        }

        for (dst, src) in approx.iter_mut().zip(input.iter().step_by(2)) {
            *dst = *src;
        }

        // Update 1: even += beta * (odd_left + odd_right)
        dwt97_forward_update_even(approx, details, BETA);

        // Predict 2: odd += gamma * (even_left + even_right)
        dwt97_forward_update_odd(approx, details, GAMMA);

        // Update 2: even += delta * (odd_left + odd_right)
        dwt97_forward_update_even(approx, details, DELTA);

        dwt97_scale(approx, details, K, INV_K);

        Ok(())
    }
}

impl DwtInverseExecutor<f32> for AvxCdf97F32 {
    /// Perform the inverse CDF 9/7 DWT (lifting scheme) to reconstruct the original signal.
    fn execute_inverse(
        &self,
        approx: &[f32],
        details: &[f32],
        output: &mut [f32],
    ) -> Result<(), OscletError> {
        unsafe { self.execute_inverse_impl(approx, details, output) }
    }
}

impl AvxCdf97F32 {
    /// Perform the inverse CDF 9/7 DWT (lifting scheme) to reconstruct the original signal.
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_inverse_impl(
        &self,
        approx: &[f32],
        details: &[f32],
        output: &mut [f32],
    ) -> Result<(), OscletError> {
        let n = approx.len() + details.len();
        if n != output.len() {
            return Err(OscletError::OutputSizeIsTooSmall(output.len(), n));
        }
        if n < 4 {
            return Err(OscletError::MinFilterSize(n, 4));
        }

        let mut approx_inv = approx.to_vec();
        let mut detail_inv = details.to_vec();

        dwt97_scale(&mut approx_inv, &mut detail_inv, INV_K, K);

        // Inverse update 2: even -= delta * (odd_left + odd_right)
        dwt97_forward_update_even(&mut approx_inv, &mut detail_inv, -DELTA);

        // Inverse predict 2: odd -= gamma * (even_left + even_right)
        dwt97_forward_update_odd(&mut approx_inv, &mut detail_inv, -GAMMA);

        // Inverse update 1: even -= beta * (odd_left + odd_right)
        dwt97_forward_update_even(&mut approx_inv, &mut detail_inv, -BETA);

        // Inverse predict 1: odd -= alpha * (even_left + even_right)
        dwt97_forward_update_odd(&mut approx_inv, &mut detail_inv, -ALPHA);

        // Interleave approx and detail to reconstruct signal
        for ((dst, &src_even), &src_odd) in output
            .chunks_exact_mut(2)
            .zip(approx_inv.iter())
            .zip(detail_inv.iter())
        {
            dst[0] = src_even;
            dst[1] = src_odd;
        }
        if !output.len().is_multiple_of(2) {
            *output.last_mut().unwrap() = *approx_inv.last().unwrap();
        }

        Ok(())
    }
}

impl IncompleteDwtExecutor<f32> for AvxCdf97F32 {
    fn filter_length(&self) -> usize {
        10
    }
}

impl DwtExecutor<f32> for AvxCdf97F32 {
    fn dwt(&self, signal: &[f32], level: usize) -> Result<Dwt<f32>, OscletError> {
        if signal.len() < 4 {
            return Err(OscletError::BufferWasTooSmallForLevel);
        }
        if level == 0 || level == 1 {
            let mut approx = try_vec![f32::default(); signal.len().div_ceil(2)];
            let mut details = try_vec![f32::default(); signal.len() / 2];

            self.execute_forward(signal, &mut approx, &mut details)?;

            Ok(Dwt {
                approximations: approx,
                details,
            })
        } else {
            let mut current_signal = signal.to_vec();
            let mut approx = vec![];
            let mut details = vec![];

            for _ in 0..level {
                if current_signal.len() < 4 {
                    return Err(OscletError::BufferWasTooSmallForLevel);
                }

                approx = try_vec![f32::default(); current_signal.len().div_ceil(2)];
                details = try_vec![f32::default(); current_signal.len() / 2];

                // Forward DWT on current signal
                self.execute_forward(&current_signal, &mut approx, &mut details)?;

                // Next level uses only the approximation
                current_signal = approx.to_vec();
            }

            Ok(Dwt {
                approximations: approx,
                details,
            })
        }
    }

    fn multi_dwt(&self, signal: &[f32], levels: usize) -> Result<MultiDwt<f32>, OscletError> {
        if signal.len() < 4 {
            return Err(OscletError::BufferWasTooSmallForLevel);
        }
        if levels == 0 || levels == 1 {
            let mut approx = try_vec![f32::default(); signal.len().div_ceil(2)];
            let mut details = try_vec![f32::default(); signal.len() / 2];

            self.execute_forward(signal, &mut approx, &mut details)?;

            Ok(MultiDwt {
                levels: vec![Dwt {
                    approximations: approx,
                    details,
                }],
            })
        } else {
            let mut current_signal = signal.to_vec();
            let mut approx;
            let mut details;

            let mut levels_store = Vec::with_capacity(levels);

            for _ in 0..levels {
                if current_signal.len() < 4 {
                    return Err(OscletError::BufferWasTooSmallForLevel);
                }

                approx = try_vec![f32::default(); current_signal.len().div_ceil(2)];
                details = try_vec![f32::default(); current_signal.len() / 2];

                // Forward DWT on current signal
                self.execute_forward(&current_signal, &mut approx, &mut details)?;

                // Next level uses only the approximation
                current_signal = approx.to_vec();

                levels_store.push(Dwt {
                    approximations: approx,
                    details,
                });
            }

            Ok(MultiDwt {
                levels: levels_store,
            })
        }
    }

    fn idwt(&self, dwt: &Dwt<f32>) -> Result<Vec<f32>, OscletError> {
        let mut output = try_vec![f32::default(); dwt.details.len() + dwt.approximations.len()];
        self.execute_inverse(&dwt.approximations, &dwt.details, &mut output)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factory::has_valid_avx;

    #[test]
    fn test_cdf97() {
        if !has_valid_avx() {
            return;
        }
        let m_cdf97 = AvxCdf97F32 {};
        let o_signal = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 52, 63, 13, 255, 63, 42, 32, 12, 52, 54, 23, 125, 23,
            255, 43, 23, 123, 54, 34, 255, 255, 23, 255, 32, 13, 15, 65, 23, 5, 7, 7, 3, 9,
        ]
        .iter()
        .map(|&x| x as f32)
        .collect::<Vec<_>>();

        let mut approx: Vec<f32> = vec![0.; (o_signal.len() + 1) / 2];
        let mut details: Vec<f32> = vec![0.; o_signal.len() / 2];

        let mut restored = vec![0.; o_signal.len()];

        m_cdf97
            .execute_forward(&o_signal, &mut approx, &mut details)
            .unwrap();

        m_cdf97
            .execute_inverse(&approx, &details, &mut restored)
            .unwrap();
        o_signal.iter().zip(restored.iter()).enumerate().for_each(|(idx, (o, re))| {
            assert!((o - re).abs() < 1e-3, "Reconstruction difference should be less than 1e-3, but it's not for original o {o}, restored {re} at idx {idx}");
        });
    }

    #[test]
    fn test_cdf97_1() {
        if !has_valid_avx() {
            return;
        }
        let m_cdf97 = AvxCdf97F32 {};
        let o_signal = vec![
            10i16, 20, 30, 40, 50, 60, 70, 80, 52, 63, 13, 255, 63, 42, 32, 12, 52, 54, 23, 125,
            23, 255, 43, 23, 123, 54, 34, 255, 255, 23, 255, 32, 13, 15, 65, 23, 5, 7, 7, 3, 9, 1,
        ]
        .iter()
        .map(|&x| x as f32)
        .collect::<Vec<_>>();

        let mut approx: Vec<f32> = vec![0.; (o_signal.len() + 1) / 2];
        let mut details: Vec<f32> = vec![0.; o_signal.len() / 2];

        let mut restored = vec![0.; o_signal.len()];

        m_cdf97
            .execute_forward(&o_signal, &mut approx, &mut details)
            .unwrap();

        m_cdf97
            .execute_inverse(&approx, &details, &mut restored)
            .unwrap();
        o_signal.iter().zip(restored.iter()).enumerate().for_each(|(idx, (o, re))| {
            assert!((o - re).abs() < 1e-3, "Reconstruction difference should be less than 1e-3, but it's not for original o {o}, restored {re} at idx {idx}");
        });
    }

    #[test]
    fn test_cdf97_2() {
        if !has_valid_avx() {
            return;
        }
        let m_cdf97 = AvxCdf97F32 {};
        let o_signal = vec![
            1, 55, 523, 40, 8, 32, 45, 166, 52, 63, 13, 255, 63, 42, 32, 12, 52, 54, 23, 125, 23,
            255, 43, 23, 123, 54, 34, 255, 255, 23, 255, 32, 13, 15, 65, 23, 5, 7, 7, 3, 9, 1,
        ]
        .iter()
        .map(|&x| x as f32)
        .collect::<Vec<_>>();

        let mut approx: Vec<f32> = vec![0.; (o_signal.len() + 1) / 2];
        let mut details: Vec<f32> = vec![0.; o_signal.len() / 2];

        let mut restored = vec![0.; o_signal.len()];

        m_cdf97
            .execute_forward(&o_signal, &mut approx, &mut details)
            .unwrap();

        m_cdf97
            .execute_inverse(&approx, &details, &mut restored)
            .unwrap();
        o_signal.iter().zip(restored.iter()).enumerate().for_each(|(idx, (o, re))| {
            assert!((o - re).abs() < 1e-3, "Reconstruction difference should be less than 1e-3, but it's not for original o {o}, restored {re} at idx {idx}");
        });
    }
}
