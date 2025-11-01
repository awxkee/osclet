# Osclet

**Osclet** is a high-performance Rust library for **Discrete Wavelet Transform (DWT)** and **Inverse DWT (IDWT)**. It
supports both **f32** and **f64** types and provides optimized implementations for modern architectures including **ARM
NEON (AArch64)** and **x86 SIMD**. Osclet is designed for signal processing, audio analysis, and image compression
workloads where speed and precision are critical.

---

## Features

- Multi-level **DWT decomposition** for `f32` and `f64` signals.
- **IDWT reconstruction** with exact recovery of the original signal.
- Support for **Daubechies wavelets** of various orders (db2, db3, db4, etc.).
- Optimized for **ARM NEON** and **x86 SIMD** instructions.
- Thread-safe and can be used in parallel contexts (`Send + Sync`).
- Customizable **border modes** for signal padding

```rust
let signal: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let dwt_executor = Osclet::make_daubechies_f64(DaubechiesFamily::Db4, BorderMode::Wrap);

let dwt = dwt_executor.dwt(&signal, 1).unwrap();
println!("Approximation coefficients: {:?}", dwt.approx);
println!("Detail coefficients: {:?}", dwt.detail);
```

----

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.