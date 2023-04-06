#![feature(test)]

extern crate test;
use test::{black_box, Bencher};

#[cfg(any(feature = "intel-mkl", feature = "cblas", feature = "matrixmultiply"))]
use ggblas::tests::matmul;
use ggblas::tests::Tensor;
use ggblas::{batched_sgemm, batched_sgemm_t};

const M: usize = 6;
const N: usize = 768 * 3;
const K: usize = 768;

#[bench]
#[cfg(feature = "intel-mkl")]
fn bench_mkl_n(bench: &mut Bencher) {
    let a = Tensor {
        shape: vec![M, K],
        data: vec![0.0; M * K],
    };
    let b = Tensor {
        shape: vec![K, N],
        data: vec![0.0; N * K],
    };
    let mut c = Tensor {
        shape: vec![M, N],
        data: vec![0.0; M * N],
    };
    bench.iter(|| black_box(matmul::<false>(&a, &b, &mut c)));
}

#[bench]
#[cfg(feature = "intel-mkl")]
fn bench_mkl_t(bench: &mut Bencher) {
    let a = Tensor {
        shape: vec![M, K],
        data: vec![0.0; M * K],
    };
    let b = Tensor {
        shape: vec![N, K],
        data: vec![0.0; N * K],
    };
    let mut c = Tensor {
        shape: vec![M, N],
        data: vec![0.0; M * N],
    };
    bench.iter(|| black_box(matmul::<true>(&a, &b, &mut c)));
}

#[bench]
#[cfg(feature = "cblas")]
fn bench_blas_n(bench: &mut Bencher) {
    let a = Tensor {
        shape: vec![M, K],
        data: vec![0.0; M * K],
    };
    let b = Tensor {
        shape: vec![K, N],
        data: vec![0.0; N * K],
    };
    let mut c = Tensor {
        shape: vec![M, N],
        data: vec![0.0; M * N],
    };
    bench.iter(|| black_box(matmul::<false>(&a, &b, &mut c)));
}

#[bench]
#[cfg(feature = "cblas")]
fn bench_blas_t(bench: &mut Bencher) {
    let a = Tensor {
        shape: vec![M, K],
        data: vec![0.0; M * K],
    };
    let b = Tensor {
        shape: vec![N, K],
        data: vec![0.0; N * K],
    };
    let mut c = Tensor {
        shape: vec![M, N],
        data: vec![0.0; M * N],
    };
    bench.iter(|| black_box(matmul::<true>(&a, &b, &mut c)));
}

#[bench]
#[cfg(feature = "matrixmultiply")]
fn bench_matrixmultiply_n(bench: &mut Bencher) {
    let a = Tensor {
        shape: vec![M, K],
        data: vec![0.0; M * K],
    };
    let b = Tensor {
        shape: vec![K, N],
        data: vec![0.0; N * K],
    };
    let mut c = Tensor {
        shape: vec![M, N],
        data: vec![0.0; M * N],
    };
    bench.iter(|| black_box(matmul::<false>(&a, &b, &mut c)));
}

#[bench]
#[cfg(feature = "matrixmultiply")]
fn bench_matrixmultiply_t(bench: &mut Bencher) {
    let a = Tensor {
        shape: vec![M, K],
        data: vec![0.0; M * K],
    };
    let b = Tensor {
        shape: vec![N, K],
        data: vec![0.0; N * K],
    };
    let mut c = Tensor {
        shape: vec![M, N],
        data: vec![0.0; M * N],
    };
    bench.iter(|| black_box(matmul::<true>(&a, &b, &mut c)));
}

#[bench]
fn bench_ggml_t(bench: &mut Bencher) {
    let a = Tensor {
        shape: vec![M, K],
        data: vec![0.0; M * K],
    };
    let b = Tensor {
        shape: vec![N, K],
        data: vec![0.0; N * K],
    };
    let mut c = Tensor {
        shape: vec![M, N],
        data: vec![0.0; M * N],
    };
    bench.iter(|| {
        black_box(batched_sgemm_t(
            a.data(),
            1,
            b.data(),
            1,
            c.data_mut(),
            1,
            M,
            N,
            K,
            1,
        ))
    });
}
#[bench]
fn bench_ggml_n(bench: &mut Bencher) {
    let a = Tensor {
        shape: vec![M, K],
        data: vec![0.0; M * K],
    };
    let b = Tensor {
        shape: vec![K, N],
        data: vec![0.0; N * K],
    };
    let mut c = Tensor {
        shape: vec![M, N],
        data: vec![0.0; M * N],
    };
    bench.iter(|| {
        black_box(batched_sgemm(
            a.data(),
            1,
            &b.data(),
            1,
            c.data_mut(),
            1,
            M,
            N,
            K,
            1,
        ))
    });
}
