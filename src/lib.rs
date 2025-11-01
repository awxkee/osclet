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
#![allow(clippy::excessive_precision)]
extern crate core;

use crate::err::OscletError;
use num_traits::{AsPrimitive, MulAdd};
use std::fmt::Debug;
use std::ops::{Add, Mul, Neg};

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod border_mode;
mod coiflet;
mod completed;
mod convolve1d;
mod daubechies;
mod err;
mod factory;
mod filter_padding;
mod mla;
mod modwt;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod symlets;
mod util;
mod wavelet2taps;
mod wavelet4taps;
mod wavelet6taps;
mod wavelet8taps;
mod wavelet_n_taps;

use crate::completed::CompletedDwtExecutor;
use crate::convolve1d::ConvolveFactory;
use crate::factory::DwtFactory;
use crate::modwt::{MoDwtHandler, Sqrt2Provider};
use crate::util::fill_wavelet;
pub use border_mode::BorderMode;
pub use coiflet::CoifletFamily;
pub use daubechies::DaubechiesFamily;
pub use modwt::MoDwtExecutor;
pub use symlets::SymletFamily;
pub use util::{dwt_length, idwt_length};

/// Provides wavelet filter coefficients for discrete wavelet transforms.
///
/// # Type Parameters
/// - `T`: The numeric type of the coefficients (e.g., `f32` or `f64`).
pub trait WaveletFilterProvider<T> {
    /// Returns the wavelet filter coefficients as a `Vec<T>`.
    fn get_wavelet(&self) -> Vec<T>;
}

/// Trait for performing the **forward discrete wavelet transform (DWT)**.
///
/// # Type Parameters
/// - `T`: The numeric type of the input signal (e.g., `f32` or `f64`).
pub trait DwtForwardExecutor<T> {
    /// Executes the forward DWT on a 1D input signal.
    ///
    /// # Parameters
    /// - `input`: Slice of the input signal.
    /// - `approx`: Mutable slice to store the approximation (low-pass) coefficients.
    /// - `details`: Mutable slice to store the detail (high-pass) coefficients.
    ///
    /// # Returns
    /// `Ok(())` on success, or an `OscletError` if sizes mismatch or computation fails.
    fn execute_forward(
        &self,
        input: &[T],
        approx: &mut [T],
        details: &mut [T],
    ) -> Result<(), OscletError>;
}

/// Trait for performing the **inverse discrete wavelet transform (IDWT)**.
///
/// # Type Parameters
/// - `T`: The numeric type of the coefficients (e.g., `f32` or `f64`).
pub trait DwtInverseExecutor<T> {
    /// Reconstructs a signal from approximation and detail coefficients.
    ///
    /// # Parameters
    /// - `approx`: Slice of approximation (low-pass) coefficients.
    /// - `details`: Slice of detail (high-pass) coefficients.
    /// - `output`: Mutable slice to store the reconstructed signal.
    ///
    /// # Returns
    /// `Ok(())` on success, or an `OscletError` if sizes mismatch or computation fails.
    fn execute_inverse(
        &self,
        approx: &[T],
        details: &[T],
        output: &mut [T],
    ) -> Result<(), OscletError>;
}

/// Combines forward and inverse DWT operations into a single executor.
///
/// Provides access to the underlying filter length and supports thread-safe execution.
pub trait IncompleteDwtExecutor<T>:
    DwtForwardExecutor<T> + DwtInverseExecutor<T> + Send + Sync
{
    /// Returns the number of coefficients in the wavelet filter.
    fn filter_length(&self) -> usize;
}

/// Represents the result of a **single-level DWT**.
pub struct Dwt<T> {
    /// Approximation (low-pass) coefficients of the signal.
    pub approximations: Vec<T>,
    /// Detail (high-pass) coefficients of the signal.
    pub details: Vec<T>,
}

/// Represents the result of a **multi-level DWT**.
pub struct MultiDwt<T> {
    /// Approximations for each level (outer Vec: levels, inner Vec: coefficients).
    pub levels: Vec<Dwt<T>>,
}

/// Full DWT executor trait combining forward, inverse, and multi-level operations.
///
/// # Type Parameters
/// - `T`: Numeric type of the signal and coefficients.
pub trait DwtExecutor<T>: IncompleteDwtExecutor<T> + Send + Sync {
    /// Performs a multi-level DWT decomposition on the input signal.
    ///
    /// # Parameters
    /// - `signal`: Slice of the input signal.
    /// - `level`: Number of decomposition levels.
    ///
    /// # Returns
    /// A `Dwt<T>` containing the approximation and detail coefficients of the final level,
    /// or an `OscletError` if computation fails.
    fn dwt(&self, signal: &[T], level: usize) -> Result<Dwt<T>, OscletError>;
    /// Performs a full multi-level DWT decomposition and stores all intermediate levels.
    ///
    /// # Parameters
    /// - `signal`: Slice containing the input signal samples.
    /// - `levels`: Number of decomposition levels to compute.
    fn multi_dwt(&self, signal: &[T], levels: usize) -> Result<MultiDwt<T>, OscletError>;
    /// Performs the inverse Discrete Wavelet Transform (IDWT) to reconstruct the original signal.
    ///
    /// # Parameters
    /// - `dwt`: A [`Dwt<T>`] structure containing the approximation and detail coefficients
    ///   from a previous DWT decomposition.
    ///
    /// # Returns
    /// - `Ok(Vec<T>)` containing the reconstructed time-domain signal.
    /// - `Err(OscletError)` if reconstruction fails, for example due to invalid coefficient lengths.
    fn idwt(&self, dwt: &Dwt<T>) -> Result<Vec<T>, OscletError>;
}

/// Factory and utility struct for creating wavelet and MODWT executors.
///
/// Provides methods to construct executors for Daubechies, Coiflet, Symlet wavelets,
/// and Maximal Overlap Discrete Wavelet Transform (MODWT) for both single and double precision.
pub struct Osclet {}

impl Osclet {
    /// Internal helper to create a wavelet executor based on a provided wavelet filter.
    ///
    /// Chooses a specialized constructor for common filter lengths (2, 4, 6, 8 taps),
    /// or a generic N-tap constructor otherwise.
    fn default_factory<T: DwtFactory<T> + 'static + Copy, W: WaveletFilterProvider<T>>(
        border_mode: BorderMode,
        db: W,
    ) -> Box<dyn IncompleteDwtExecutor<T> + Send + Sync>
    where
        f64: AsPrimitive<T>,
    {
        let filter = db.get_wavelet();
        match filter.len() {
            2 => T::wavelet_2_taps(border_mode, filter.as_slice().try_into().unwrap()),
            4 => T::wavelet_4_taps(border_mode, filter.as_slice().try_into().unwrap()),
            6 => T::wavelet_6_taps(border_mode, filter.as_slice().try_into().unwrap()),
            8 => T::wavelet_8_taps(border_mode, filter.as_slice().try_into().unwrap()),
            _ => T::wavelet_n_taps(border_mode, filter.as_slice()),
        }
    }

    /// Internal implementation for creating a Daubechies wavelet executor.
    fn make_daubechies_impl<
        T: DwtFactory<T>
            + 'static
            + Copy
            + Default
            + Send
            + Sync
            + MulAdd<T, Output = T>
            + Add<T, Output = T>
            + Mul<T, Output = T>,
    >(
        db: DaubechiesFamily,
        border_mode: BorderMode,
    ) -> Box<dyn DwtExecutor<T> + Send + Sync>
    where
        f64: AsPrimitive<T>,
    {
        let intercept = Self::default_factory(border_mode, db);
        Box::new(CompletedDwtExecutor::new(intercept))
    }

    /// Internal implementation for creating a Coiflet wavelet executor.
    fn make_coiflet_impl<
        T: DwtFactory<T>
            + 'static
            + Copy
            + Default
            + Send
            + Sync
            + MulAdd<T, Output = T>
            + Add<T, Output = T>
            + Mul<T, Output = T>,
    >(
        coif: CoifletFamily,
        border_mode: BorderMode,
    ) -> Box<dyn DwtExecutor<T> + Send + Sync>
    where
        f64: AsPrimitive<T>,
    {
        let intercept = Self::default_factory(border_mode, coif);
        Box::new(CompletedDwtExecutor::new(intercept))
    }

    /// Internal implementation for creating a Symlet wavelet executor.
    fn make_symlet_impl<
        T: DwtFactory<T>
            + 'static
            + Copy
            + Default
            + Send
            + Sync
            + MulAdd<T, Output = T>
            + Add<T, Output = T>
            + Mul<T, Output = T>,
    >(
        sym: SymletFamily,
        border_mode: BorderMode,
    ) -> Box<dyn DwtExecutor<T> + Send + Sync>
    where
        f64: AsPrimitive<T>,
    {
        let intercept = Self::default_factory(border_mode, sym);
        Box::new(CompletedDwtExecutor::new(intercept))
    }

    /// Internal implementation for creating a MODWT executor.
    ///
    /// # Parameters
    /// - `provider`: Supplies the wavelet filter coefficients.
    /// - `border_mode`: How signal boundaries are handled during convolution.
    ///
    /// # Returns
    /// A boxed `MoDwtExecutor<T>` if successful, or an `OscletError` if the wavelet
    /// is empty or has an odd number of coefficients.
    fn make_modwt<
        T: ConvolveFactory<T>
            + 'static
            + Copy
            + Default
            + Send
            + Sync
            + MulAdd<T, Output = T>
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + Neg<Output = T>
            + Sqrt2Provider<T>
            + Debug,
    >(
        provider: Box<dyn WaveletFilterProvider<T> + Send + Sync>,
        border_mode: BorderMode,
    ) -> Result<Box<dyn MoDwtExecutor<T> + Send + Sync>, OscletError>
    where
        f64: AsPrimitive<T>,
    {
        let wave = provider.get_wavelet();
        if wave.is_empty() || wave.len() % 2 != 0 {
            return Err(OscletError::ZeroOrOddSizedWavelet);
        }
        Ok(Box::new(MoDwtHandler::new(
            fill_wavelet(&wave)?,
            T::make_convolution_1d(border_mode),
        )))
    }

    /// Creates a Daubechies wavelet executor for `f32` signals.
    ///
    /// # Parameters
    /// - `db`: The Daubechies wavelet family to use (e.g., db1, db2, etc.).
    /// - `border_mode`: How the signal edges are handled (e.g., zero-padding, symmetric).
    ///
    /// # Returns
    /// A boxed `DwtExecutor<f32>` that can perform discrete wavelet transforms.
    pub fn make_daubechies_f32(
        db: DaubechiesFamily,
        border_mode: BorderMode,
    ) -> Box<dyn DwtExecutor<f32> + Send + Sync> {
        Self::make_daubechies_impl(db, border_mode)
    }

    /// Creates a Daubechies wavelet executor for `f64` signals.
    ///
    /// Same as `make_daubechies_f32`, but for double-precision signals.
    pub fn make_daubechies_f64(
        db: DaubechiesFamily,
        border_mode: BorderMode,
    ) -> Box<dyn DwtExecutor<f64> + Send + Sync> {
        Self::make_daubechies_impl(db, border_mode)
    }

    /// Creates a Coiflet wavelet executor for `f32` signals.
    ///
    /// # Parameters
    /// - `coif`: The Coiflet wavelet family to use (e.g., coif1, coif2, etc.).
    /// - `border_mode`: How the signal edges are handled.
    ///
    /// # Returns
    /// A boxed `DwtExecutor<f32>` for performing discrete wavelet transforms.
    pub fn make_coiflet_f32(
        coif: CoifletFamily,
        border_mode: BorderMode,
    ) -> Box<dyn DwtExecutor<f32> + Send + Sync> {
        Self::make_coiflet_impl(coif, border_mode)
    }

    /// Creates a Coiflet wavelet executor for `f64` signals.
    ///
    /// Same as `make_coiflet_f32`, but for double-precision signals.
    pub fn make_coiflet_f64(
        coif: CoifletFamily,
        border_mode: BorderMode,
    ) -> Box<dyn DwtExecutor<f64> + Send + Sync> {
        Self::make_coiflet_impl(coif, border_mode)
    }

    /// Creates a Symlet wavelet executor for `f32` signals.
    ///
    /// # Parameters
    /// - `sym`: The Symlet wavelet family to use (e.g., sym2, sym3, etc.).
    /// - `border_mode`: How the signal edges are handled.
    ///
    /// # Returns
    /// A boxed `DwtExecutor<f32>` for performing discrete wavelet transforms.
    pub fn make_symlet_f32(
        sym: SymletFamily,
        border_mode: BorderMode,
    ) -> Box<dyn DwtExecutor<f32> + Send + Sync> {
        Self::make_symlet_impl(sym, border_mode)
    }

    /// Creates a Symlet wavelet executor for `f64` signals.
    ///
    /// Same as `make_symlet_f32`, but for double-precision signals.
    pub fn make_symlet_f64(
        sym: SymletFamily,
        border_mode: BorderMode,
    ) -> Box<dyn DwtExecutor<f64> + Send + Sync> {
        Self::make_symlet_impl(sym, border_mode)
    }

    /// Creates a MODWT (Maximal Overlap Discrete Wavelet Transform) executor for `f32` signals using a provided wavelet filter.
    ///
    /// # Parameters
    /// - `provider`: A boxed provider supplying wavelet filter coefficients.
    /// - `border_mode`: How the signal edges are handled.
    ///
    /// # Returns
    /// A `Result` containing a boxed `MoDwtExecutor<f32>` if successful, or an `OscletError`.
    pub fn make_modwt_f32(
        provider: Box<dyn WaveletFilterProvider<f32> + Send + Sync>,
        border_mode: BorderMode,
    ) -> Result<Box<dyn MoDwtExecutor<f32> + Send + Sync>, OscletError> {
        Self::make_modwt(provider, border_mode)
    }

    /// Creates a MODWT (Maximal Overlap Discrete Wavelet Transform) executor for `f64` signals using a provided wavelet filter.
    ///
    /// Same as `make_modwt_f32`, but for double-precision signals.
    pub fn make_modwt_f64(
        provider: Box<dyn WaveletFilterProvider<f64> + Send + Sync>,
        border_mode: BorderMode,
    ) -> Result<Box<dyn MoDwtExecutor<f64> + Send + Sync>, OscletError> {
        Self::make_modwt(provider, border_mode)
    }
}
