# Osclet

Osclet is a high-performance Rust library for Discrete Wavelet Transform (DWT), Stationary Wavelet Transform (SWT), and their inverses.
It supports f32, f64, and integer signals, with highly optimized implementations for modern architectures including ARM NEON (AArch64) and x86 SIMD (AVX/AVX2).

---

## Features

- Multi-level DWT decomposition for 1D signals (f32, f64, i16, i32).
- Inverse DWT/SWT for exact or energy-preserving reconstruction.
- Stationary Wavelet Transform (SWT) / Maximal Overlap DWT (MODWT) for shift-invariant analysis.
- Support for multiple wavelet families:
    Daubechies (db1, db2, …)
    
    Symlets
    
    Coiflets
    
    Biorthogonal

    CDF 5/3 and CDF 9/7 (integer and floating-point variants)
    
    Custom wavelets via user-provided filter coefficients.
- Border modes: zero-padding, symmetric, mirror, and circular wrap.
- Thread-safe and parallel-ready.
- Hardware-optimized using SIMD instructions:
  ARM NEON (AArch64)
  x86 AVX2/SSE
- 2D separable wavelet transforms for images with row/column decomposition.

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