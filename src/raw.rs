use crate::ggml::{vec_dot_f32, vec_mad_f32};

use crate::ThreadPool;

pub unsafe fn ggml_compute_forward_mul_mat(
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
    pool: &ThreadPool,
) {
    let ap = ap.as_ptr();
    let bp = bp.as_ptr();
    let cp = cp.as_mut_ptr();
    let total = batching * m;

    let n_cpu = pool.max_count();
    let ap = ap as usize;
    let bp = bp as usize;
    let cp = cp as usize;
    let total_th = (total / n_cpu) + 1;

    (0..n_cpu).for_each(|ith| {
        pool.execute(move || {
            (ith * total_th..std::cmp::min(total, (ith + 1) * total_th)).for_each(|iter| {
                let step = iter / m;
                let i = iter % m;
                (0..k).for_each(|kk| {
                    let a_start = step * a_skip + i * k + kk;
                    let b_start = step * b_skip + kk * n;
                    let c_start = step * c_skip + (i * n);

                    unsafe {
                        let ap = ap as *const f32;
                        let bp = bp as *const f32;
                        let cp = cp as *mut f32;
                        let av = *ap.add(a_start);
                        let b_row = bp.add(b_start);
                        let c_row = cp.add(c_start);
                        vec_mad_f32(b_row, c_row, av, n);
                    }
                });
            });
        });
    });
    pool.join();
}

pub unsafe fn ggml_compute_forward_mul_mat_t(
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
    pool: &ThreadPool,
) {
    let ap = ap.as_ptr();
    let bp = bp.as_ptr();
    let cp = cp.as_mut_ptr();
    let total = batching * m * n;

    let n_cpu = pool.max_count();

    let ap = ap as usize;
    let bp = bp as usize;
    let cp = cp as usize;
    let total_th = (total / n_cpu) + 1;

    (0..n_cpu).for_each(|ith| {
        pool.execute(move || {
            (ith * total_th..std::cmp::min(total, (ith + 1) * total_th)).for_each(|iter| {
                let step = iter / (m * n);
                let i = (iter / n) % m;
                let j = iter % n;
                let a_start = step * a_skip + i * k;
                let b_start = step * b_skip + j * k;
                let c_start = step * c_skip + (i * n + j);

                unsafe {
                    let ap = ap as *const f32;
                    let bp = bp as *const f32;
                    let cp = cp as *mut f32;
                    let a_row = ap.add(a_start);
                    let b_row = bp.add(b_start);
                    let c_ptr = cp.add(c_start);
                    vec_dot_f32(a_row, b_row, c_ptr, k);
                }
            });
        });
    });
    pool.join();
}

#[cfg(feature = "f16")]
pub mod f16 {
    use super::ThreadPool;
    use crate::ggml::f16::{f32_to_f16, vec_dot_f16};
    use half::f16;

    pub unsafe fn ggml_compute_forward_mul_mat_t_f16_mixed(
        ap: &[f32],
        a_skip: usize,
        bp: &[f16],
        b_skip: usize,
        cp: &mut [f32],
        c_skip: usize,
        m: usize,
        n: usize,
        k: usize,
        batching: usize,
        pool: &ThreadPool,
    ) {
        let mut a_16 = vec![f16::from_f32_const(0.0); ap.len()];
        f32_to_f16(ap.as_ptr(), a_16.as_mut_ptr(), ap.len());
        let ap = a_16;
        let ap = ap.as_ptr();
        let bp = bp.as_ptr();
        let cp = cp.as_mut_ptr();
        let total = batching * m * n;

        let n_cpu = pool.max_count();

        let ap = ap as usize;
        let bp = bp as usize;
        let cp = cp as usize;
        let total_th = (total / n_cpu) + 1;

        (0..n_cpu).for_each(|ith| {
            pool.execute(move || {
                (ith * total_th..std::cmp::min(total, (ith + 1) * total_th)).for_each(|iter| {
                    let step = iter / (m * n);
                    let i = (iter / n) % m;
                    let j = iter % n;
                    let a_start = step * a_skip + i * k;
                    let b_start = step * b_skip + j * k;
                    let c_start = step * c_skip + (i * n + j);

                    unsafe {
                        let ap = ap as *const f16;
                        let bp = bp as *const f16;
                        let cp = cp as *mut f32;
                        let a_row = ap.add(a_start);
                        let b_row = bp.add(b_start);
                        let c_ptr = cp.add(c_start);
                        vec_dot_f16(a_row, b_row, c_ptr, k);
                    }
                });
            });
        });
        pool.join();
    }

    pub unsafe fn ggml_compute_forward_mul_mat_t_f16_pure(
        ap: &[f16],
        a_skip: usize,
        bp: &[f16],
        b_skip: usize,
        cp: &mut [f16],
        c_skip: usize,
        m: usize,
        n: usize,
        k: usize,
        batching: usize,
        pool: &ThreadPool,
    ) {
        let ap = ap.as_ptr();
        let bp = bp.as_ptr();
        let cp = cp.as_mut_ptr();
        let total = batching * m * n;

        let n_cpu = pool.max_count();

        let ap = ap as usize;
        let bp = bp as usize;
        let cp = cp as usize;
        let total_th = (total / n_cpu) + 1;

        (0..n_cpu).for_each(|ith| {
            pool.execute(move || {
                (ith * total_th..std::cmp::min(total, (ith + 1) * total_th)).for_each(|iter| {
                    let step = iter / (m * n);
                    let i = (iter / n) % m;
                    let j = iter % n;
                    let a_start = step * a_skip + i * k;
                    let b_start = step * b_skip + j * k;
                    let c_start = step * c_skip + (i * n + j);

                    unsafe {
                        let ap = ap as *const f16;
                        let bp = bp as *const f16;
                        let cp = cp as *mut f16;
                        let a_row = ap.add(a_start);
                        let b_row = bp.add(b_start);
                        let c_ptr = cp.add(c_start);
                        let mut sum = 0.0;
                        vec_dot_f16(a_row, b_row, &mut sum, k);
                        *c_ptr = f16::from_f32(sum);
                    }
                });
            });
        });
        pool.join();
    }
}
