[![Crates.io](https://img.shields.io/crates/v/ggblas.svg)](https://crates.io/crates/ggblas)
[![Documentation](https://docs.rs/ggblas/badge.svg)](https://docs.rs/ggblas/)
[![Dependency status](https://deps.rs/repo/github/Narsil/ggblas/status.svg?path=ggblas)](https://deps.rs/repo/github/Narsil/ggblas)

# ggblas

ggblas is a library aimed to provide a simple and ergonomic access
to the matrixmuliplication implemented in [ggml](https://github.com/ggerganov/llama.cpp/)

This library adds on top a [threadpool](https://docs.rs/threadpool/latest/threadpool/)
with the physical number of cores each thread being pinned to their respective
counterpart.

## Usage

```rust
use ggblas::batched_sgemm;

let a = vec![1., 2., 3., 4.];
let b = vec![1., 2., 3., 4.];
let mut c = vec![0., 0., 0., 0.];

batched_sgemm(&a, &b, &mut c, 2, 2, 2);
assert_eq!(c, &[7., 10., 15., 22.]);

let mut c = vec![0.];
batched_sgemm(&a, &b, &mut c, 1, 1, 4);
assert_eq!(c, &[30.]);
```

## Performance

Current performance can be see [here](https://nodata.dev/ggblas/dev/bench/)

### Intel

i5-9300 (avx2)

```bash
test bench_ggblas_n ... bench:     469,739 ns/iter (+/- 3,111)
test bench_ggblas_t ... bench:     317,049 ns/iter (+/- 5,450)
test bench_mkl_n    ... bench:     140,561 ns/iter (+/- 1,095)
test bench_mkl_t    ... bench:     185,928 ns/iter (+/- 2,781)
# (cblas)
test bench_blas_n   ... bench:   5,955,545 ns/iter (+/- 87,172)
test bench_blas_t   ... bench:  10,153,008 ns/iter (+/- 528,645)
# (matrixmultiply+threading)
test bench_matrixmultiply_n ... bench:     869,372 ns/iter (+/- 205,883)
test bench_matrixmultiply_t ... bench:     841,705 ns/iter (+/- 12,706)
```

### M1 (neon)

test bench_ggml_n           ... bench:     640,552 ns/iter (+/- 21,558)
test bench_ggml_t           ... bench:     270,919 ns/iter (+/- 10,761)
test bench_matrixmultiply_n ... bench:     944,152 ns/iter (+/- 38,737)
test bench_matrixmultiply_t ... bench:     809,709 ns/iter (+/- 13,350)
test bench_blas_n ... bench:      97,389 ns/iter (+/- 701)
test bench_blas_t ... bench:     628,720 ns/iter (+/- 87,855)



License: Apache-2.0
