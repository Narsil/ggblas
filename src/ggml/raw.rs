pub unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    // leftovers
    for i in 0..k {
        *c += *a_row.offset(i as isize) * (*b_row.offset(i as isize));
    }
}
