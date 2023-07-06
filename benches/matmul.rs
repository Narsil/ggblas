#![feature(test)]

extern crate test;
use half::f16;
use test::{black_box, Bencher};

#[cfg(any(
    feature = "intel-mkl",
    feature = "cblas",
    feature = "matrixmultiply",
    feature = "faer-rs"
))]
use ggblas::tests::matmul;
use ggblas::tests::Tensor;
use ggblas::{batched_sgemm, batched_sgemm_t, batched_sgemm_t_f16_mixed, batched_sgemm_t_f16_pure};

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
#[cfg(feature = "faer-rs")]
fn bench_faer_rs_n(bench: &mut Bencher) {
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
#[cfg(feature = "faer-rs")]
fn bench_faer_rs_t(bench: &mut Bencher) {
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
fn bench_ggblas_t_f32(bench: &mut Bencher) {
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
    bench.iter(|| black_box(batched_sgemm_t(a.data(), b.data(), c.data_mut(), M, N, K)));
}

#[bench]
fn bench_ggblas_t_f16_mixed(bench: &mut Bencher) {
    let a_data = vec![0.0; M * K];
    let b_data = vec![f16::from_f32(0.0); N * K];
    let mut c_data = vec![0.0; M * N];
    bench.iter(|| {
        black_box(batched_sgemm_t_f16_mixed(
            &a_data,
            &b_data,
            &mut c_data,
            M,
            N,
            K,
        ))
    });
}

#[bench]
fn bench_ggblas_t_f16_pure(bench: &mut Bencher) {
    let a_data = vec![f16::from_f32(0.0); M * K];
    let b_data = vec![f16::from_f32(0.0); N * K];
    let mut c_data = vec![f16::from_f32(0.0); M * N];
    bench.iter(|| {
        black_box(batched_sgemm_t_f16_pure(
            &a_data,
            &b_data,
            &mut c_data,
            M,
            N,
            K,
        ))
    });
}
#[bench]
fn bench_ggblas_n(bench: &mut Bencher) {
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
    bench.iter(|| black_box(batched_sgemm(a.data(), b.data(), c.data_mut(), M, N, K)));
}
