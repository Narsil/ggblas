#![feature(test)]

extern crate test;
use test::{black_box, Bencher};

use rblas::{ggml_matmul, matmul, Tensor};

#[bench]
fn bench_mkl(bench: &mut Bencher) {
    let m = 6;
    let n = 768 * 3;
    let k = 768;

    let a = Tensor {
        shape: vec![m, k],
        data: vec![0.0; m * k],
    };
    let b = Tensor {
        shape: vec![n, k],
        data: vec![0.0; n * k],
    };
    let mut c = Tensor {
        shape: vec![m, n],
        data: vec![0.0; m * n],
    };
    bench.iter(|| black_box(matmul::<true>(&a, &b, &mut c)));
}

#[bench]
fn bench_ggml(bench: &mut Bencher) {
    let m = 6;
    let n = 768 * 3;
    let k = 768;

    let a = Tensor {
        shape: vec![m, k],
        data: vec![0.0; m * k],
    };
    let b = Tensor {
        shape: vec![n, k],
        data: vec![0.0; n * k],
    };
    let mut c = Tensor {
        shape: vec![m, n],
        data: vec![0.0; m * n],
    };
    bench.iter(|| black_box(ggml_matmul::<true>(&a, &b, &mut c)));
}
