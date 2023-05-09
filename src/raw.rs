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
