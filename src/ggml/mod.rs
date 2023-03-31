#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "avx")]
pub mod avx;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(target_feature = "avx")]
pub use avx::vec_dot_f32;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(not(target_feature = "avx"))]
pub mod raw;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(not(target_feature = "avx"))]
pub use raw::vec_dot_f32;

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[cfg(target_feature = "neon")]
pub use neon::vec_dot_f32;

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[cfg(target_feature = "neon")]
#[cfg(not(target_feature = "neon"))]
pub use raw::vec_dot_f32;
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[cfg(not(target_feature = "neon"))]
pub use raw::vec_dot_f32;
