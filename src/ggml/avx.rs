use super::Cpu;
#[cfg(target_arch = "x86")]
use std::arch::x86::{__m256, _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m256, _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_loadu_ps,
    _mm256_mul_ps, _mm256_setzero_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
};

pub struct CurrentCpu {}

const STEP: usize = 32;
const EPR: usize = 8;
const ARR: usize = STEP / EPR;

impl Cpu<ARR> for CurrentCpu {
    type Unit = __m256;
    type Array = [__m256; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;

    fn n() -> usize {
        ARR
    }

    unsafe fn zero() -> Self::Unit {
        _mm256_setzero_ps()
    }

    unsafe fn zero_array() -> Self::Array {
        [Self::zero(); ARR]
    }

    unsafe fn load(mem_addr: *const f32) -> Self::Unit {
        _mm256_loadu_ps(mem_addr)
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        _mm256_add_ps(_mm256_mul_ps(b, c), a)
    }

    unsafe fn vec_reduce(mut x: Self::Array, y: *mut f32) {
        for i in 0..ARR / 2 {
            x[2 * i] = _mm256_add_ps(x[2 * i], x[2 * i + 1]);
        }
        for i in 0..ARR / 4 {
            x[4 * i] = _mm256_add_ps(x[4 * i], x[4 * i + 2]);
        }
        for i in 0..ARR / 8 {
            x[8 * i] = _mm256_add_ps(x[8 * i], x[8 * i + 4]);
        }
        let t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]), _mm256_extractf128_ps(x[0], 1));
        let t1 = _mm_hadd_ps(t0, t0);
        *y = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
    }
}
