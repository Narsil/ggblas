trait Cpu<const ARR: usize> {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;

    fn n() -> usize;
    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const f32) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut f32, a: Self::Unit);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "avx")]
pub mod avx;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "avx")]
pub use avx::CurrentCpu;

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[cfg(target_feature = "neon")]
pub use neon::CurrentCpu;

#[cfg(any(target_feature = "neon", target_feature = "avx"))]
pub unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    let np = k & !(CurrentCpu::STEP - 1);

    let mut sum = CurrentCpu::zero_array();
    let mut ax = CurrentCpu::zero_array();
    let mut ay = CurrentCpu::zero_array();

    for i in (0..np).step_by(CurrentCpu::STEP) {
        for j in 0..CurrentCpu::n() {
            ax[j] = CurrentCpu::load(a_row.add(i + j * CurrentCpu::EPR));
            ay[j] = CurrentCpu::load(b_row.add(i + j * CurrentCpu::EPR));

            sum[j] = CurrentCpu::vec_fma(sum[j], ax[j], ay[j]);
        }
    }

    CurrentCpu::vec_reduce(sum, c);

    // leftovers
    for i in np..k {
        *c += *a_row.add(i) * (*b_row.add(i));
    }
}

#[cfg(any(target_feature = "neon", target_feature = "avx"))]
pub unsafe fn vec_mad_f32(b_row: *const f32, c_row: *mut f32, v: f32, n: usize) {
    let np = n & !(CurrentCpu::STEP - 1);

    let vx = CurrentCpu::from_f32(v);
    let mut ax = CurrentCpu::zero_array();
    let mut ay = CurrentCpu::zero_array();

    for i in (0..np).step_by(CurrentCpu::STEP) {
        for j in 0..CurrentCpu::n() {
            ax[j] = CurrentCpu::load(b_row.add(i + j * CurrentCpu::EPR));
            ay[j] = CurrentCpu::load(c_row.add(i + j * CurrentCpu::EPR));
            ay[j] = CurrentCpu::vec_fma(ay[j], ax[j], vx);
            CurrentCpu::vec_store(c_row.add(i + j * CurrentCpu::EPR), ay[j]);
        }
    }

    // leftovers
    for i in np..n {
        *c_row.add(i) += *b_row.add(i) * v;
    }
}

#[cfg(not(any(target_feature = "neon", target_feature = "avx")))]
pub unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    // leftovers
    for i in 0..k {
        *c += *a_row.add(i) * (*b_row.add(i));
    }
}

#[cfg(not(any(target_feature = "neon", target_feature = "avx")))]
pub unsafe fn vec_mad_f32(a_row: *const f32, c_row: *mut f32, v: f32, n: usize) {
    for i in 0..n {
        *c_row.add(i) += *a_row.add(i) * v;
    }
}
