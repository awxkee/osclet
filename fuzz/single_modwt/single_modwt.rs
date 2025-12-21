#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use osclet::{BorderMode, Osclet, WaveletFilterProvider};
use std::borrow::Cow;
use std::sync::Arc;

#[derive(Arbitrary, Debug)]
struct Data {
    length: u8,
    wavelet_length: u8,
}

struct WaveletProvider {
    wavelet: Vec<f32>,
}

impl WaveletFilterProvider<f32> for WaveletProvider {
    fn get_wavelet(&self) -> Cow<'_, [f32]> {
        Cow::Borrowed(self.wavelet.as_slice())
    }
}

fuzz_target!(|data: Data| {
    if data.length == 0 || data.wavelet_length == 0 || !data.wavelet_length.is_multiple_of(2) {
        return;
    }
    if data.length < data.wavelet_length {
        return;
    }
    let mut data = data;
    data.wavelet_length = 8;
    let mut wavelet = vec![0.; data.wavelet_length as usize];
    for i in 0..data.wavelet_length as usize {
        wavelet[i] = i as f32 / data.wavelet_length as f32;
    }
    let mut signal = vec![0.; data.length as usize];
    for i in 0..data.length as usize {
        signal[i] = i as f32 / data.length as f32;
    }
    let executor =
        Osclet::make_modwt_f32(Arc::new(WaveletProvider { wavelet }), BorderMode::Wrap).unwrap();
    let dwt = executor.dwt(&signal, 1).unwrap();
    _ = executor.idwt(&dwt.to_ref(), 1).unwrap();
});
