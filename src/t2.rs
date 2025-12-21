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
use crate::err::try_vec;
use crate::transpose::TransposeExecutor;
use crate::{DwtExecutor, OscletError};
use std::fmt::Debug;
use std::sync::Arc;

/// Executes a 2-D Discrete Wavelet Transform (DWT) and its inverse.
///
/// Implementors perform a separable 2-D DWT by applying a 1-D transform
/// horizontally and vertically, producing four sub-bands (LL, LH, HL, HH).
pub trait Dwt2DExecutor<T: Debug + Clone + Debug + Copy> {
    /// Performs a **forward 2-D DWT** using preallocated output buffers.
    ///
    /// # Parameters
    /// - `src`: Input image data in row-major order, length `width × height`.
    /// - `width`, `height`: Dimensions of the input image.
    /// - `ll_band`, `lh_band`, `hl_band`, `hh_band`: Mutable slices to store
    ///   the respective sub-band coefficients.
    /// - `scratch`: Temporary buffer required for intermediate computations.
    ///
    /// # Returns
    /// `Ok(())` on success, or an `OscletError` if sizes mismatch or computation fails.
    #[allow(clippy::too_many_arguments)]
    fn execute_forward(
        &self,
        src: &[T],
        width: usize,
        height: usize,
        ll_band: &mut [T],
        lh_band: &mut [T],
        hl_band: &mut [T],
        hh_band: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), OscletError>;

    /// Performs a **forward 2-D DWT** and returns a `FilterBank2DMut<T>` with all sub-bands.
    ///
    /// # Parameters
    /// - `src`: Input image data in row-major order, length `width × height`.
    /// - `width`, `height`: Dimensions of the input image.
    ///
    /// # Returns
    /// `Ok(FilterBank2DMut<T>)` containing the LL, LH, HL, HH sub-bands.
    fn dwt(
        &self,
        src: &[T],
        width: usize,
        height: usize,
    ) -> Result<FilterBank2DMut<T>, OscletError>;

    /// Returns the required size of the temporary scratch buffer for a given image size.
    ///
    /// # Parameters
    /// - `width`, `height`: Dimensions of the input image.
    ///
    /// # Returns
    /// Number of elements required in the scratch buffer.
    fn required_scratch_size(&self, width: usize, height: usize) -> usize;

    /// Performs the **inverse 2-D DWT** and reconstructs the original image.
    ///
    /// # Parameters
    /// - `dst`: Mutable slice to store the reconstructed image, length `width × height`.
    /// - `filter_bank`: Reference to the DWT sub-bands.
    /// - `width`, `height`: Dimensions of the output image.
    /// - `scratch`: Temporary buffer required for intermediate computations.
    ///
    /// # Returns
    /// `Ok(())` on success, or an `OscletError` if sizes mismatch or computation fails.
    fn execute_inverse(
        &self,
        dst: &mut [T],
        filter_bank: &FilterBank2D<T>,
        width: usize,
        height: usize,
        scratch: &mut [T],
    ) -> Result<(), OscletError>;

    /// Convenience method to perform the inverse 2-D DWT and return the reconstructed image.
    ///
    /// Allocates a new buffer for the output.
    ///
    /// # Parameters
    /// - `filter_bank`: Reference to the DWT sub-bands.
    /// - `width`, `height`: Dimensions of the output image.
    ///
    /// # Returns
    /// `Ok(Vec<T>)` containing the reconstructed image.
    fn idwt(
        &self,
        filter_bank: &FilterBank2D<T>,
        width: usize,
        height: usize,
    ) -> Result<Vec<T>, OscletError>;
}

/// Immutable view of a 2-D DWT filter bank.
///
/// Contains references to the four sub-bands produced by a 2-D DWT.
#[derive(Debug, Clone)]
pub struct FilterBank2D<'a, T: Debug + Clone + Debug> {
    /// low-pass horizontally and vertically (approximation).
    pub ll_band: &'a [T],
    /// low-pass horizontally, high-pass vertically.
    pub lh_band: &'a [T],
    /// high-pass horizontally, low-pass vertically.
    pub hl_band: &'a [T],
    /// high-pass horizontally and vertically (details).
    pub hh_band: &'a [T],
}

/// Owned 2-D DWT filter bank.
///
/// Contains four sub-bands stored as owned `Vec<T>`. Can be converted
/// to an immutable view via `to_filter_bank()`.
#[derive(Debug)]
pub struct FilterBank2DMut<T: Debug + Clone + Debug + Copy> {
    /// low-pass horizontally and vertically (approximation).
    pub ll_band: Vec<T>,
    /// low-pass horizontally, high-pass vertically.
    pub lh_band: Vec<T>,
    /// high-pass horizontally, low-pass vertically.
    pub hl_band: Vec<T>,
    /// high-pass horizontally and vertically (details).
    pub hh_band: Vec<T>,
}

impl<T: Debug + Clone + Debug + Copy> FilterBank2DMut<T> {
    /// Returns an immutable view of this filter bank.
    pub fn to_filter_bank(&self) -> FilterBank2D<'_, T> {
        FilterBank2D {
            lh_band: self.lh_band.as_slice(),
            ll_band: self.ll_band.as_slice(),
            hh_band: self.hh_band.as_slice(),
            hl_band: self.hl_band.as_slice(),
        }
    }
}

pub(crate) struct Wavelet2DTransform<T> {
    pub(crate) filter: Arc<dyn DwtExecutor<T> + Send + Sync>,
    pub(crate) transpose: Arc<dyn TransposeExecutor<T> + Send + Sync>,
}

impl<T: Debug + Clone + Debug + Copy + Default> Dwt2DExecutor<T> for Wavelet2DTransform<T> {
    fn execute_forward(
        &self,
        src: &[T],
        width: usize,
        height: usize,
        ll_band: &mut [T],
        lh_band: &mut [T],
        hl_band: &mut [T],
        hh_band: &mut [T],
        scratch: &mut [T],
    ) -> Result<(), OscletError> {
        if width == 0 || height == 0 {
            return Err(OscletError::ZeroedBaseSize);
        }

        let total_size = width.checked_mul(height).ok_or(OscletError::Overflow)?;

        if src.len() != total_size {
            return Err(OscletError::InputSize(total_size, src.len()));
        }

        let dwt_size_width = self.filter.dwt_size(width);

        let required_scratch = self.required_scratch_size(width, height);

        if scratch.len() < required_scratch {
            return Err(OscletError::InputSize(required_scratch, scratch.len()));
        }

        let (scratch, _) = scratch.split_at_mut(required_scratch);

        let (scratch0, scratch1) = scratch.split_at_mut(required_scratch / 2);

        let (approx, details) = scratch0.split_at_mut(dwt_size_width.approx_length * height);

        let dwt_scratch_width = self.filter.required_scratch_size(width);
        let dwt_scratch_height = self.filter.required_scratch_size(height);

        let mut dwt_scratch = try_vec![T::default(); dwt_scratch_width.max(dwt_scratch_height)];

        for ((app, det), src) in approx
            .chunks_exact_mut(dwt_size_width.approx_length)
            .zip(details.chunks_exact_mut(dwt_size_width.details_length))
            .zip(src.chunks_exact(width))
        {
            self.filter
                .execute_forward_with_scratch(src, app, det, &mut dwt_scratch)?;
        }

        let (transposed_approx, transposed_details) =
            scratch1.split_at_mut(dwt_size_width.approx_length * height);

        self.transpose.transpose(
            approx,
            transposed_approx,
            dwt_size_width.approx_length,
            height,
        );

        self.transpose.transpose(
            details,
            transposed_details,
            dwt_size_width.details_length,
            height,
        );

        let dwt_size_height = self.filter.dwt_size(height);

        for ((ll, lh), src) in ll_band
            .chunks_exact_mut(dwt_size_height.approx_length)
            .zip(lh_band.chunks_exact_mut(dwt_size_height.details_length))
            .zip(transposed_approx.chunks_exact(height))
        {
            self.filter
                .execute_forward_with_scratch(src, ll, lh, &mut dwt_scratch)?;
        }

        for ((hl, hh), src) in hl_band
            .chunks_exact_mut(dwt_size_height.approx_length)
            .zip(hh_band.chunks_exact_mut(dwt_size_height.details_length))
            .zip(transposed_details.chunks_exact(height))
        {
            self.filter
                .execute_forward_with_scratch(src, hl, hh, &mut dwt_scratch)?;
        }

        Ok(())
    }

    fn dwt(
        &self,
        src: &[T],
        width: usize,
        height: usize,
    ) -> Result<FilterBank2DMut<T>, OscletError> {
        if width == 0 || height == 0 {
            return Err(OscletError::ZeroedBaseSize);
        }

        let total_size = width.checked_mul(height).ok_or(OscletError::Overflow)?;

        if src.len() != total_size {
            return Err(OscletError::InputSize(total_size, src.len()));
        }

        let dwt_size_width = self.filter.dwt_size(width);
        let dwt_size_height = self.filter.dwt_size(height);

        let mut ll_band =
            try_vec![T::default(); dwt_size_width.approx_length * dwt_size_height.approx_length]; // width = dwt_size_height.approx_length
        let mut lh_band =
            try_vec![T::default(); dwt_size_width.approx_length * dwt_size_height.details_length]; // width = dwt_size_height.details_length
        let mut hl_band =
            try_vec![T::default(); dwt_size_width.details_length * dwt_size_height.approx_length]; // width = dwt_size_height.approx_length
        let mut hh_band =
            try_vec![T::default(); dwt_size_width.details_length * dwt_size_height.details_length]; // width = dwt_size_height.details_length

        let mut scratch = try_vec![T::default(); self.required_scratch_size(width, height)];

        self.execute_forward(
            src,
            width,
            height,
            &mut ll_band,
            &mut lh_band,
            &mut hl_band,
            &mut hh_band,
            &mut scratch,
        )?;

        Ok(FilterBank2DMut {
            ll_band,
            hh_band,
            hl_band,
            lh_band,
        })
    }

    fn required_scratch_size(&self, width: usize, height: usize) -> usize {
        let dwt_size_width = self.filter.dwt_size(width);
        (dwt_size_width.approx_length * height + dwt_size_width.details_length * height) * 2
    }

    fn execute_inverse(
        &self,
        dst: &mut [T],
        filter_bank: &FilterBank2D<T>,
        width: usize,
        height: usize,
        scratch: &mut [T],
    ) -> Result<(), OscletError> {
        if width == 0 || height == 0 {
            return Err(OscletError::ZeroedBaseSize);
        }

        let total_size = width.checked_mul(height).ok_or(OscletError::Overflow)?;

        if dst.len() != total_size {
            return Err(OscletError::InputSize(total_size, dst.len()));
        }

        let dwt_size_width = self.filter.dwt_size(width);
        let dwt_size_height = self.filter.dwt_size(height);

        let required_scratch = self.required_scratch_size(width, height);

        if scratch.len() < required_scratch {
            return Err(OscletError::InputSize(required_scratch, scratch.len()));
        }

        let (scratch, _) = scratch.split_at_mut(required_scratch);

        let (scratch0, scratch1) = scratch.split_at_mut(required_scratch / 2);

        let (transposed_approx, transposed_details) =
            scratch0.split_at_mut(dwt_size_width.approx_length * height);

        for ((ll, lh), dst) in filter_bank
            .ll_band
            .chunks_exact(dwt_size_height.approx_length)
            .zip(
                filter_bank
                    .lh_band
                    .chunks_exact(dwt_size_height.details_length),
            )
            .zip(transposed_approx.chunks_exact_mut(height))
        {
            self.filter.execute_inverse(ll, lh, dst)?;
        }

        for ((hl, hh), dst) in filter_bank
            .hl_band
            .chunks_exact(dwt_size_height.approx_length)
            .zip(
                filter_bank
                    .hh_band
                    .chunks_exact(dwt_size_height.details_length),
            )
            .zip(transposed_details.chunks_exact_mut(height))
        {
            self.filter.execute_inverse(hl, hh, dst)?;
        }

        let (approx, details) = scratch1.split_at_mut(dwt_size_width.approx_length * height);

        self.transpose.transpose(
            transposed_approx,
            approx,
            height,
            dwt_size_width.approx_length,
        );

        self.transpose.transpose(
            transposed_details,
            details,
            height,
            dwt_size_width.details_length,
        );

        for ((app, det), dst) in approx
            .chunks_exact(dwt_size_width.approx_length)
            .zip(details.chunks_exact(dwt_size_width.details_length))
            .zip(dst.chunks_exact_mut(width))
        {
            self.filter.execute_inverse(app, det, dst)?;
        }

        Ok(())
    }

    fn idwt(
        &self,
        filter_bank: &FilterBank2D<T>,
        width: usize,
        height: usize,
    ) -> Result<Vec<T>, OscletError> {
        if width == 0 || height == 0 {
            return Err(OscletError::ZeroedBaseSize);
        }

        let total_size = width.checked_mul(height).ok_or(OscletError::Overflow)?;

        let mut output = try_vec![T::default(); total_size];

        let mut scratch = try_vec![T::default(); self.required_scratch_size(width, height)];

        self.execute_inverse(&mut output, filter_bank, width, height, &mut scratch)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use crate::Osclet;

    #[test]
    fn test_2d_dwt_le_gall5_3() {
        let mut image = vec![0f32; 9 * 9];
        for (i, dst) in image.iter_mut().enumerate() {
            *dst = i as f32;
        }
        let filter = Osclet::make_cdf53_f32();
        let td2 = Osclet::make_2d_dwt_f32(filter);
        let q = td2.dwt(&image, 9, 9).unwrap();
        let inverse = td2.idwt(&q.to_filter_bank(), 9, 9).unwrap();
        assert_eq!(image, inverse);
    }

    #[test]
    fn test_2d_dwt_le_gall5_3_i16() {
        let mut image = vec![0i16; 9 * 9];
        for (i, dst) in image.iter_mut().enumerate() {
            *dst = i as i16;
        }
        let filter = Osclet::make_cdf53_i16();
        let td2 = Osclet::make_2d_dwt_i16(filter);
        let q = td2.dwt(&image, 9, 9).unwrap();
        let inverse = td2.idwt(&q.to_filter_bank(), 9, 9).unwrap();
        assert_eq!(image, inverse);
    }
}
