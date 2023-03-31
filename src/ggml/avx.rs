#[cfg(target_arch = "x86")]
use std::arch::x86::{_mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_mul_ps,
    _mm256_setzero_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
};

pub const GGML_F32_STEP: usize = 32;
pub const GGML_F32_EPR: usize = 8;
pub const GGML_F32_ARR: usize = GGML_F32_STEP / GGML_F32_EPR;

macro_rules! GGML_F32_VEC_ZERO {
    () => {
        _mm256_setzero_ps()
    };
}

macro_rules! GGML_F32_VEC_LOAD {
    ($e:expr) => {
        _mm256_loadu_ps($e)
    };
}

macro_rules! GGML_F32_VEC_FMA {
    ($a:expr, $b:expr, $c:expr) => {
        _mm256_add_ps(_mm256_mul_ps($b, $c), $a)
    };
}

macro_rules! GGML_F32_VEC_REDUCE {
    ($res:expr, $x:expr) => {
        for i in 0..GGML_F32_ARR / 2 {
            $x[2 * i] = _mm256_add_ps($x[2 * i], $x[2 * i + 1]);
        }
        for i in 0..GGML_F32_ARR / 4 {
            $x[4 * i] = _mm256_add_ps($x[4 * i], $x[4 * i + 2]);
        }
        for i in 0..GGML_F32_ARR / 8 {
            $x[8 * i] = _mm256_add_ps($x[8 * i], $x[8 * i + 4]);
        }
        let t0 = _mm_add_ps(
            _mm256_castps256_ps128($x[0]),
            _mm256_extractf128_ps($x[0], 1),
        );
        let t1 = _mm_hadd_ps(t0, t0);
        $res = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
    };
}

pub unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    let np = k & !(GGML_F32_STEP - 1);

    let mut sum = [GGML_F32_VEC_ZERO!(); GGML_F32_ARR];
    let mut ax = [GGML_F32_VEC_ZERO!(); GGML_F32_ARR];
    let mut ay = [GGML_F32_VEC_ZERO!(); GGML_F32_ARR];

    for i in (0..np).step_by(GGML_F32_STEP) {
        for j in 0..GGML_F32_ARR {
            ax[j] = GGML_F32_VEC_LOAD!(a_row.offset((i + j * GGML_F32_EPR) as isize));
            ay[j] = GGML_F32_VEC_LOAD!(b_row.offset((i + j * GGML_F32_EPR) as isize));

            sum[j] = GGML_F32_VEC_FMA!(sum[j], ax[j], ay[j]);
        }
    }

    GGML_F32_VEC_REDUCE!(*c, sum);

    // leftovers
    for i in np..k {
        *c += *a_row.offset(i as isize) * (*b_row.offset(i as isize));
    }
}
