mod raw;
use cblas_sys::{
    cblas_sgemm as sgemm, CblasColMajor as ColMajor, CblasNoTrans as NoTr,
    CblasRowMajor as RowMajor, CblasTrans as Tr,
};
use raw::ggml_compute_forward_mul_mat;

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> &[f32] {
        &self.data
    }
    fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

#[derive(Debug)]
pub enum Error {
    Dim,
}

#[inline]
pub fn matmul<const TRANSPOSE: bool>(a: &Tensor, b: &Tensor, c: &mut Tensor) -> Result<(), Error> {
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

        let mut w = vec![0.0; 18432];

        let params = ggml_raw::ggml_compute_params {
            type_: ggml_raw::ggml_task_type::GGML_TASK_COMPUTE,
            ith: 0,
            nth: 1,
            wsize: w.len() * std::mem::size_of::<f32>(),
            wdata: w.as_mut_ptr() as *mut libc::c_void,
        };
        let a_tensor = ggml_raw::ggml_tensor {
            type_: ggml_raw::GGML_TYPE_F32,
            n_dims: 4,
            ne: [768, 2340, 1, 1],
            nb: [9216, 4, 7077888, 7077888],
            op: ggml_raw::GGML_OP_MUL_MAT,
            is_param: false,
            grad: std::ptr::null_mut(),
            src0: std::ptr::null_mut(),
            src1: std::ptr::null_mut(),
            opt: [std::ptr::null_mut(); 4],
            n_tasks: 1,
            perf_runs: 0,
            perf_cycles: 0,
            perf_time_us: 0,
            data: ap.as_ptr() as *mut libc::c_void,
            padding: [0; 8],
        };
        let b_tensor = ggml_raw::ggml_tensor {
            type_: ggml_raw::GGML_TYPE_F32,
            n_dims: 4,
            ne: [768, 6, 1, 1],
            nb: [4, 3072, 18432, 18432],
            op: ggml_raw::GGML_OP_MUL_MAT,
            is_param: false,
            grad: std::ptr::null_mut(),
            src0: std::ptr::null_mut(),
            src1: std::ptr::null_mut(),
            opt: [std::ptr::null_mut(); 4],
            n_tasks: 1,
            perf_runs: 0,
            perf_cycles: 0,
            perf_time_us: 0,
            data: bp.as_ptr() as *mut libc::c_void,
            padding: [0; 8],
        };
        let mut c_tensor = ggml_raw::ggml_tensor {
            type_: ggml_raw::GGML_TYPE_F32,
            n_dims: 4,
            ne: [2304, 6, 1, 1],
            nb: [4, 9216, 55296, 55296],
            op: ggml_raw::GGML_OP_MUL_MAT,
            is_param: false,
            grad: std::ptr::null_mut(),
            src0: std::ptr::null_mut(),
            src1: std::ptr::null_mut(),
            opt: [std::ptr::null_mut(); 4],
            n_tasks: 1,
            perf_runs: 0,
            perf_cycles: 0,
            perf_time_us: 0,
            data: cp.as_ptr() as *mut libc::c_void,
            padding: [0; 8],
        };

        unsafe {
            // ggml_raw::ggml_soft_max(std::ptr::null_mut(), std::ptr::null_mut());
            // ggml_raw::ggml_compute_forward_mul_mat(
            ggml_compute_forward_mul_mat(
                &params as *const ggml_raw::ggml_compute_params,
                &a_tensor as *const ggml_raw::ggml_tensor,
                &b_tensor as *const ggml_raw::ggml_tensor,
                &mut c_tensor as *mut ggml_raw::ggml_tensor,
            );
        }
    });
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ggml() {
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
        ggml_matmul::<true>(&a, &b, &mut c).unwrap();
    }
}
