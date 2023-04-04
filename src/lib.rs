mod ggml;
mod raw;

use raw::{ggml_compute_forward_mul_mat, ggml_compute_forward_mul_mat_t};

use std::sync::Once;
use threadpool::ThreadPool;
static mut HANDLE: Option<ThreadPool> = None;
static GUARD: Once = Once::new();

unsafe fn get_pool() -> Option<&'static ThreadPool> {
    GUARD.call_once(|| {
        HANDLE = Some(ThreadPool::new(num_cpus::get()));
    });
    HANDLE.as_ref()
}

pub unsafe fn batched_sgemm_t(
    ap: &[f32],
    a_skip: usize,
    bp: &[f32],
    b_skip: usize,
    cp: &mut [f32],
    c_skip: usize,
    m: usize,
    n: usize,
    k: usize,
    batching: usize,
) {
    ggml_compute_forward_mul_mat_t(
        ap,
        a_skip,
        bp,
        b_skip,
        cp,
        c_skip,
        m,
        n,
        k,
        batching,
        &get_pool().unwrap(),
    );
}

pub unsafe fn batched_sgemm(
    ap: &[f32],
    a_skip: usize,
    bp: &[f32],
    b_skip: usize,
    cp: &mut [f32],
    c_skip: usize,
    m: usize,
    n: usize,
    k: usize,
    batching: usize,
) {
    ggml_compute_forward_mul_mat(
        ap,
        a_skip,
        bp,
        b_skip,
        cp,
        c_skip,
        m,
        n,
        k,
        batching,
        &get_pool().unwrap(),
    );
}

pub mod tests {
    use super::*;

    #[cfg(any(feature = "cblas", feature = "intel-mkl"))]
    use cblas_sys::{
        cblas_sgemm as sgemm, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
        CblasRowMajor as RowMajor, CblasTrans as Tr,
    };

    pub struct Tensor {
        pub shape: Vec<usize>,
        pub data: Vec<f32>,
    }

    impl Tensor {
        pub fn shape(&self) -> &[usize] {
            &self.shape
        }
        pub fn data(&self) -> &[f32] {
            &self.data
        }
        pub fn data_mut(&mut self) -> &mut [f32] {
            &mut self.data
        }
    }

    #[derive(Debug)]
    pub enum Error {
        Dim,
    }

    #[cfg(any(feature = "cblas", feature = "intel-mkl"))]
    #[inline]
    pub fn matmul<const TRANSPOSE: bool>(
        a: &Tensor,
        b: &Tensor,
        c: &mut Tensor,
    ) -> Result<(), Error> {
        let dim = a.shape().len();

        if dim < 2 {
            return Err(Error::Dim);
        }
        if b.shape().len() != dim {
            return Err(Error::Dim);
        }
        if c.shape().len() != dim {
            return Err(Error::Dim);
        }

        let m = a.shape()[dim - 2];
        let k = a.shape()[dim - 1];

        let mut expected_c = a.shape().to_vec();
        let mut expected_b = a.shape().to_vec();

        let (expected_b, n) = if TRANSPOSE {
            let n = b.shape()[dim - 2];
            expected_b[dim - 2] = n;
            expected_b[dim - 1] = k;
            (expected_b, n)
        } else {
            let n = b.shape()[dim - 1];
            expected_b[dim - 2] = k;
            expected_b[dim - 1] = n;
            (expected_b, n)
        };

        expected_c[dim - 2] = m;
        expected_c[dim - 1] = n;

        if expected_b != b.shape() {
            return Err(Error::Dim);
        }

        if expected_c != c.shape() {
            return Err(Error::Dim);
        }

        // Zero out c
        // c.data_mut().iter_mut().for_each(|v| *v = 0.0);

        let batching: usize = a.shape()[..dim - 2].iter().product();
        let a_skip: usize = m * k;
        let b_skip: usize = n * k;
        let c_skip: usize = m * n;

        let ar = k as isize;
        let ac = 1;
        let (br, bc) = if TRANSPOSE {
            (1, b.shape()[dim - 1] as isize)
        } else {
            (b.shape()[dim - 1] as isize, 1)
        };
        let cr = n as isize;
        let cc = 1;

        (0..batching).for_each(|step| {
            let ap = &a.data()[step * a_skip..];
            let bp = &b.data()[step * b_skip..];
            let cp = &mut c.data_mut()[step * c_skip..];

            unsafe {
                let (m, n, k) = (m as libc::c_int, n as libc::c_int, k as libc::c_int);
                let (layout, a_tr, b_tr, lda, ldb, ldc) = if cr < cc {
                    let (lda, a_tr) = if ar < ac { (m, NoTr) } else { (k, Tr) };
                    let (ldb, b_tr) = if br < bc { (k, NoTr) } else { (n, Tr) };
                    (ColMajor, a_tr, b_tr, lda, ldb, m)
                } else {
                    let (lda, a_tr) = if ar < ac { (m, Tr) } else { (k, NoTr) };
                    let (ldb, b_tr) = if br < bc { (k, Tr) } else { (n, NoTr) };
                    (RowMajor, a_tr, b_tr, lda, ldb, n)
                };
                sgemm(
                    layout,
                    a_tr,
                    b_tr,
                    m,
                    n,
                    k,
                    1.0,
                    ap.as_ptr(),
                    lda,
                    // a_skip as i32,
                    bp.as_ptr(),
                    ldb,
                    // b_skip as i32,
                    1.0,
                    cp.as_mut_ptr(),
                    ldc,
                    // c_skip as i32,
                    // batching as i32,
                )
            }
        });
        Ok(())
    }

    #[inline]
    pub fn ggml_matmul<const TRANSPOSE: bool>(
        a: &Tensor,
        b: &Tensor,
        c: &mut Tensor,
        pool: &ThreadPool,
    ) -> Result<(), Error> {
        let dim = a.shape().len();

        if dim < 2 {
            return Err(Error::Dim);
        }
        if b.shape().len() != dim {
            return Err(Error::Dim);
        }
        if c.shape().len() != dim {
            return Err(Error::Dim);
        }

        let m = a.shape()[dim - 2];
        let k = a.shape()[dim - 1];

        let mut expected_c = a.shape().to_vec();
        let mut expected_b = a.shape().to_vec();

        let (expected_b, n) = if TRANSPOSE {
            let n = b.shape()[dim - 2];
            expected_b[dim - 2] = n;
            expected_b[dim - 1] = k;
            (expected_b, n)
        } else {
            let n = b.shape()[dim - 1];
            expected_b[dim - 2] = k;
            expected_b[dim - 1] = n;
            (expected_b, n)
        };

        expected_c[dim - 2] = m;
        expected_c[dim - 1] = n;

        if expected_b != b.shape() {
            return Err(Error::Dim);
        }

        if expected_c != c.shape() {
            return Err(Error::Dim);
        }

        // Zero out c
        // c.data_mut().iter_mut().for_each(|v| *v = 0.0);

        let batching: usize = a.shape()[..dim - 2].iter().product();
        let a_skip: usize = m * k;
        let b_skip: usize = n * k;
        let c_skip: usize = m * n;

        if TRANSPOSE {
            unsafe {
                ggml_compute_forward_mul_mat_t(
                    a.data(),
                    a_skip,
                    b.data(),
                    b_skip,
                    c.data_mut(),
                    c_skip,
                    m,
                    n,
                    k,
                    batching,
                    pool,
                );
            }
        } else {
            unsafe {
                ggml_compute_forward_mul_mat(
                    a.data(),
                    a_skip,
                    b.data(),
                    b_skip,
                    c.data_mut(),
                    c_skip,
                    m,
                    n,
                    k,
                    batching,
                    pool,
                );
            }
        }

        Ok(())
    }

    #[test]
    fn ggml_simple_t() {
        let m = 3;
        let n = 2;
        let k = 4;

        let a = Tensor {
            shape: vec![m, k],
            data: (0..m * k).map(|s| (s + 1) as f32).collect(),
        };
        let b = Tensor {
            shape: vec![n, k],
            data: (0..n * k).map(|s| (s + 1) as f32).collect(),
        };
        let mut c = Tensor {
            shape: vec![m, n],
            data: vec![0.0; m * n],
        };
        let pool = ThreadPool::new(1);
        ggml_matmul::<true>(&a, &b, &mut c, &pool).unwrap();

        assert_eq!(c.data(), [30.0, 70.0, 70.0, 174.0, 110.0, 278.0]);
    }

    #[test]
    fn ggml_simple2() {
        let m = 2;
        let n = 2;
        let k = 2;

        let a = Tensor {
            shape: vec![m, k],
            data: (0..m * k).map(|s| (s + 1) as f32).collect(),
        };
        let b = Tensor {
            shape: vec![k, n],
            data: (0..n * k).map(|s| (s + 1) as f32).collect(),
        };
        let mut c = Tensor {
            shape: vec![m, n],
            data: vec![0.0; m * n],
        };
        unsafe {
            batched_sgemm(a.data(), 1, b.data(), 1, c.data_mut(), 1, m, n, k, 1);
        }

        assert_eq!(c.data(), [7., 10., 15., 22.]);
    }

    #[test]
    fn ggml_simple() {
        let m = 3;
        let n = 2;
        let k = 4;

        let a = Tensor {
            shape: vec![m, k],
            data: (0..m * k).map(|s| (s + 1) as f32).collect(),
        };
        let b = Tensor {
            shape: vec![k, n],
            data: (0..n * k).map(|s| (s + 1) as f32).collect(),
        };
        let mut c = Tensor {
            shape: vec![m, n],
            data: vec![0.0; m * n],
        };
        let pool = ThreadPool::new(1);
        ggml_matmul::<false>(&a, &b, &mut c, &pool).unwrap();

        assert_eq!(c.data(), [50., 60., 114., 140., 178., 220.]);
    }

    #[test]
    #[cfg(any(feature = "cblas", feature = "intel-mkl"))]
    fn mkl_simple() {
        let m = 3;
        let n = 2;
        let k = 4;

        let a = Tensor {
            shape: vec![m, k],
            data: (0..m * k).map(|s| (s + 1) as f32).collect(),
        };
        let b = Tensor {
            shape: vec![n, k],
            data: (0..n * k).map(|s| (s + 1) as f32).collect(),
        };
        let mut c = Tensor {
            shape: vec![m, n],
            data: vec![0.0; m * n],
        };
        matmul::<true>(&a, &b, &mut c).unwrap();

        assert_eq!(c.data(), [30.0, 70.0, 70.0, 174.0, 110.0, 278.0]);
    }

    #[test]
    #[cfg(any(feature = "cblas", feature = "intel-mkl"))]
    fn ggml_comparison_t() {
        let m = 6;
        let n = 768 * 3;
        let k = 768;

        let a = Tensor {
            shape: vec![m, k],
            data: (0..m * k).map(|s| (s + 1) as f32).collect(),
        };
        let b = Tensor {
            shape: vec![n, k],
            data: (0..n * k).map(|s| (s + 1) as f32).collect(),
        };
        let mut c = Tensor {
            shape: vec![m, n],
            data: vec![0.0; m * n],
        };
        let mut c2 = Tensor {
            shape: vec![m, n],
            data: vec![0.0; m * n],
        };
        let pool = ThreadPool::new(1);
        matmul::<true>(&a, &b, &mut c).unwrap();
        ggml_matmul::<true>(&a, &b, &mut c2, &pool).unwrap();
        assert_close(&c.data(), &c2.data());
    }

    #[test]
    #[cfg(any(feature = "cblas", feature = "intel-mkl"))]
    fn ggml_comparison() {
        let m = 6;
        let n = 768 * 3;
        let k = 768;

        let a = Tensor {
            shape: vec![m, k],
            data: (0..m * k).map(|s| (s + 1) as f32).collect(),
        };
        let b = Tensor {
            shape: vec![k, n],
            data: (0..n * k).map(|s| (s + 1) as f32).collect(),
        };
        let mut c = Tensor {
            shape: vec![m, n],
            data: vec![0.0; m * n],
        };
        let mut c2 = Tensor {
            shape: vec![m, n],
            data: vec![0.0; m * n],
        };
        let pool = ThreadPool::new(1);
        matmul::<false>(&a, &b, &mut c).unwrap();
        ggml_matmul::<false>(&a, &b, &mut c2, &pool).unwrap();
        assert_close(&c.data(), &c2.data());
    }

    #[cfg(any(feature = "cblas", feature = "intel-mkl"))]
    pub fn assert_close(a: &[f32], b: &[f32]) {
        a.iter().zip(b.iter()).for_each(|(&a, &b)| {
            if ((a - b) / a).abs() > 1e-5 {
                panic!("{a:?} != {b:?}");
            }
        });
    }
}
