#[cfg(target_arch = "arm")]
use core::arch::arm::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

pub const GGML_F32_STEP: usize = 16;
pub const GGML_F32_EPR: usize = 4;
pub const GGML_F32_ARR: usize = GGML_F32_STEP / GGML_F32_EPR;

macro_rules! GGML_F32_VEC_ZERO {
    () => {
        vdupq_n_f32(0.0)
    };
}

macro_rules! GGML_F32_VEC_LOAD {
    ($e:expr) => {
	vld1q_f32($e)
    };
}

macro_rules! GGML_F32_VEC_FMA {
    ($a:expr, $b:expr, $c:expr) => {
	vfmaq_f32($a, $b, $c)
    };
}

#[cfg(target_arch="aarch64")]
macro_rules! GGML_F32x4_REDUCE_ONE{
	($x:expr) => {
		vaddvq_f32($x)
			
	}
}

#[cfg(target_arch="arm")]
macro_rules! GGML_F32x4_REDUCE_ONE{
	($x:expr) => {

    vgetq_lane_f32($x, 0) +          
     vgetq_lane_f32($x, 1) +          
     vgetq_lane_f32($x, 2) +          
     vgetq_lane_f32($x, 3)
}
			
	}

macro_rules! GGML_F32_VEC_REDUCE {
    ($res:expr, $x:expr) => {
        for i in 0..GGML_F32_ARR / 2 {
            $x[2 * i] = vaddq_f32($x[2 * i], $x[2 * i + 1]);
        }
        for i in 0..GGML_F32_ARR / 4 {
            $x[4 * i] = vaddq_f32($x[4 * i], $x[4 * i + 2]);
        }
        for i in 0..GGML_F32_ARR / 8 {
            $x[8 * i] = vaddq_f32($x[8 * i], $x[8 * i + 4]);
        }
        $res = GGML_F32x4_REDUCE_ONE!($x[0]);
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
