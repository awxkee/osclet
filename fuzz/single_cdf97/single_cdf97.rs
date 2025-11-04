#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use osclet::{BorderMode, Osclet, WaveletFilterProvider};

#[derive(Arbitrary, Debug)]
struct Data {
    length: u8,
    wavelet_length: u8,
}

struct WaveletProvider {
    wavelet: Vec<f32>,
}

impl WaveletFilterProvider<f32> for WaveletProvider {
    fn get_wavelet(&self) -> Vec<f32> {
        self.wavelet.to_vec()
    }
}

fuzz_target!(|data: Data| {
    if data.length == 0 || data.wavelet_length == 0 || !data.wavelet_length.is_multiple_of(2) {
        return;
    }
    if data.length < data.wavelet_length {
        return;
    }
    let mut wavelet = vec![0.; data.wavelet_length as usize];
    for i in 0..data.wavelet_length as usize {
        wavelet[i] = i as f32 / data.wavelet_length as f32;
    }
    let mut signal = vec![0.; data.length as usize];
    for i in 0..data.length as usize {
        signal[i] = i as f32 / data.length as f32;
    }
    let executor = Osclet::make_cdf97_f32();
    let dwt = executor.dwt(&signal, 1).unwrap();
    _ = executor.idwt(&dwt).unwrap();
});
