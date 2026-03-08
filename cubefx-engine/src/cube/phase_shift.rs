use cubecl::{
    prelude::*,
    std::tensor::{AsView as _, AsViewExpand, AsViewMut, AsViewMutExpand, TensorHandle},
};

use crate::cube::{BatchSignalLayout, cube_selection};

/// Per-bin phase shift effect kernel
///
/// Creates the output (real and imaginary) tensors
/// then launches the per-bin phase shift kernel to fill them with the right values
pub fn phase_shift<R: Runtime>(
    input_re: TensorHandle<R>,
    input_im: TensorHandle<R>,
    alpha: f32,
) -> (TensorHandle<R>, TensorHandle<R>) {
    let client = <R as Runtime>::client(&Default::default());
    let shape = input_re.shape.clone();
    let num_elements = shape.iter().product::<usize>();
    let dtype = input_re.dtype;

    let output_re = TensorHandle::new_contiguous(
        shape.clone(),
        client.empty(num_elements * dtype.size()),
        dtype,
    );

    let output_im =
        TensorHandle::new_contiguous(shape, client.empty(num_elements * dtype.size()), dtype);

    phase_shift_launch::<R>(
        &client,
        input_re.as_ref(),
        input_im.as_ref(),
        output_re.as_ref(),
        output_im.as_ref(),
        alpha,
        dtype,
    )
    .unwrap();

    (output_re, output_im)
}

/// Launches the per-bin phase shift with the specified Cube Count, Cube Dim and vectorization (line size)
pub fn phase_shift_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input_re: TensorHandleRef<R>,
    input_im: TensorHandleRef<R>,
    output_re: TensorHandleRef<R>,
    output_im: TensorHandleRef<R>,
    alpha: f32,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let num_iter = input_re.shape[0] * input_re.shape[1];
    let (cube_dim, cube_count, _) = cube_selection(&client.properties().hardware, num_iter, false);

    let vectorization = 1;

    phase_shift_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        input_re.as_tensor_arg(vectorization),
        input_im.as_tensor_arg(vectorization),
        output_re.as_tensor_arg(vectorization),
        output_im.as_tensor_arg(vectorization),
        InputScalar::new(alpha, dtype),
        dtype,
    )
}

#[cube(launch)]
/// Kernel that loops over each window and applies the effect on each
pub(crate) fn phase_shift_kernel<F: Float>(
    input_re: &Tensor<Line<F>>,
    input_im: &Tensor<Line<F>>,
    output_re: &mut Tensor<Line<F>>,
    output_im: &mut Tensor<Line<F>>,
    alpha: InputScalar,
    #[define(F)] _dtype: StorageType,
) {
    phase_shift_kernel_one_window(
        input_re,
        input_im,
        output_re,
        output_im,
        ABSOLUTE_POS,
        alpha,
    );
}

#[cube]
/// Applies the effect on one window.
/// Iterates on the frequency bins and applies the scaled phase shift to each
pub(crate) fn phase_shift_kernel_one_window<F: Float>(
    input_re: &Tensor<Line<F>>,
    input_im: &Tensor<Line<F>>,
    output_re: &mut Tensor<Line<F>>,
    output_im: &mut Tensor<Line<F>>,
    window_index: usize,
    alpha: InputScalar,
) {
    let num_freq_bins = input_re.shape(input_re.rank() - 1);

    // The following code allow to ignore the batch index and assume only one window
    let input_re_layout = BatchSignalLayout::new(input_re, window_index);
    let input_im_layout = BatchSignalLayout::new(input_im, window_index);
    let output_re_layout = BatchSignalLayout::new(input_im, window_index);
    let output_im_layout = BatchSignalLayout::new(input_im, window_index);
    let input_re_view = input_re.view(input_re_layout);
    let input_im_view = input_im.view(input_im_layout);
    let mut output_re_view = output_re.view_mut(output_re_layout);
    let mut output_im_view = output_im.view_mut(output_im_layout);

    for k in 0..num_freq_bins {
        let k = k % num_freq_bins;

        // Warning: if line size > 1, this will duplicate the same k, while we would want something like [x, x+1, x+2, x+3...
        let theta = Line::cast_from(alpha.get::<F>() * F::cast_from(k));

        let input_re = input_re_view[k];
        let input_im = input_im_view[k];

        output_re_view[k] = input_re * theta.cos() - input_im * theta.sin();
        output_im_view[k] = input_re * theta.sin() + input_im * theta.cos();
    }
}
