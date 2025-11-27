#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use osclet::Osclet;

#[derive(Arbitrary, Debug)]
struct Data {
    length: u8,
}

fuzz_target!(|data: Data| {
    if data.length == 0 {
        return;
    }
    if data.length < 4 {
        return;
    }
    let mut signal = vec![0.; data.length as usize];
    for i in 0..data.length as usize {
        signal[i] = i as f32 / data.length as f32;
    }
    let executor = Osclet::make_cdf97_f32();
    let dwt = executor.dwt(&signal, 1).unwrap();
    _ = executor.idwt(&dwt).unwrap();
});
