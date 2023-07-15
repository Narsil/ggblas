use super::f16::CpuF16;
use core::arch::wasm32::*;
use half::f16;

const STEP: usize = 16;
const EPR: usize = 4;
const ARR: usize = STEP / EPR;

pub struct CurrentCpuF16 {}

impl CpuF16<ARR> for CurrentCpuF16 {
    type Unit = v128;
    type Array = [v128; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;

    fn n() -> usize {
        ARR
    }

    unsafe fn zero() -> Self::Unit {
        f32x4_splat(0.0f32)
    }

    unsafe fn zero_array() -> Self::Array {
        [Self::zero(); ARR]
    }

    unsafe fn from_f32(v: f32) -> Self::Unit {
        f32x4_splat(v)
    }

    unsafe fn load(mem_addr: *const f16) -> Self::Unit {
        let mut tmp = [0.0f32; 4];
        for i in 0..4 {
            tmp[i] = (*mem_addr.add(i)).to_f32();
        }
        v128_load(tmp.as_ptr() as *const v128)
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        f32x4_add(f32x4_mul(b, c), a)
    }

    unsafe fn vec_store(mem_addr: *mut f16, a: Self::Unit) {
        let mut tmp = [0.0f32; 4];
        v128_store(tmp.as_mut_ptr() as *mut v128, a);
        for i in 0..4 {
            *mem_addr.add(i) = f16::from_f32(tmp[i]);
        }
    }

    unsafe fn vec_reduce(mut x: Self::Array, y: *mut f32) {
        let mut offset = ARR >> 1;
        for i in 0..offset {
            x[i] = f32x4_add(x[i], x[offset + i]);
        }
        offset >>= 1;
        for i in 0..offset {
            x[i] = f32x4_add(x[i], x[offset + i]);
        }
        offset >>= 1;
        for i in 0..offset {
            x[i] = f32x4_add(x[i], x[offset + i]);
        }
        *y = f32x4_extract_lane::<0>(x[0])
            + f32x4_extract_lane::<1>(x[0])
            + f32x4_extract_lane::<2>(x[0])
            + f32x4_extract_lane::<3>(x[0]);
    }
}
