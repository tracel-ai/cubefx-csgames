mod fft;
mod layout;
mod phase_shift;

#[cfg(test)]
mod tests;

use cubecl::{CubeCount, CubeDim, ir::HardwareProperties};
pub use fft::*;
pub(crate) use layout::*;
pub use phase_shift::*;

pub fn cube_selection(
    hw: &HardwareProperties,
    num_iter: usize,
    force_no_cube_dim_gpu: bool,
) -> (CubeDim, CubeCount, bool) {
    let plane_size = hw.plane_size_min;
    let x = match hw.num_cpu_cores {
        Some(num_cores) => num_cores,
        None => {
            if force_no_cube_dim_gpu {
                return (
                    CubeDim::new_single(),
                    CubeCount::new_1d(num_iter as u32),
                    force_no_cube_dim_gpu,
                );
            }
            1
        }
    };

    let x = u32::min(x, num_iter as u32 / plane_size);
    let x = u32::max(x, 1);

    let cube_dim = CubeDim::new_2d(plane_size, x);
    let cube_count = num_iter as u32 / cube_dim.num_elems();
    let cube_count = CubeCount::new_1d(cube_count + 1);

    (cube_dim, cube_count, false)
}
