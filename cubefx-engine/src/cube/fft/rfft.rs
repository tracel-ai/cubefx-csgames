use cubecl::{prelude::*, std::tensor::TensorHandle};

use crate::cube::{FftMode, cube_selection};

use cubecl::std::tensor::{
    AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand, layout::plain::PlainLayout,
};

use crate::cube::{BatchSignalLayout, fft::fft_inner_compute};

/// Real-valued Fast Fourier Transform kernel.
///
/// Creates spectrum (real and imaginary) tensors
/// then launches the RFFT kernel to fill them with the right values
pub fn rfft<R: Runtime>(
    signal: TensorHandle<R>,
    dtype: StorageType,
) -> (TensorHandle<R>, TensorHandle<R>) {
    // Assumes fft always done on last dim
    let dim = signal.shape.len() - 1;
    assert!(
        signal.shape[dim].is_power_of_two(),
        "RFFT requires power-of-2 length"
    );
    let client = <R as Runtime>::client(&Default::default());

    let mut spectrum_shape = signal.shape.clone();
    spectrum_shape[dim] = signal.shape[dim] / 2 + 1;

    let spectrum_re = TensorHandle::new_contiguous(
        spectrum_shape.clone(),
        client.empty(spectrum_shape.iter().product::<usize>() * dtype.size()),
        dtype,
    );

    let spectrum_im = TensorHandle::new_contiguous(
        spectrum_shape.clone(),
        client.empty(spectrum_shape.iter().product::<usize>() * dtype.size()),
        dtype,
    );

    rfft_launch::<R>(
        &client,
        signal.as_ref(),
        spectrum_re.as_ref(),
        spectrum_im.as_ref(),
        dtype,
    )
    .unwrap();

    (spectrum_re, spectrum_im)
}

/// Launches the RFFT with the specified Cube Count, Cube Dim and vectorization (line size)
pub fn rfft_launch<R: Runtime>(
    client: &ComputeClient<R>,
    signal: TensorHandleRef<R>,
    spectrum_re: TensorHandleRef<R>,
    spectrum_im: TensorHandleRef<R>,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let num_iter = signal.shape[0] * signal.shape[1];
    let (cube_dim, cube_count) = cube_selection(&client.properties().hardware, num_iter);
    let num_samples = *signal.shape.last().unwrap();
    let vectorization_input = client
        .io_optimized_line_sizes(&dtype)
        .filter(|c| num_samples % c == 0)
        .take(1)
        .last()
        .unwrap_or(1);
    let vectorization = 1;

    unsafe {
        rfft_kernel::launch_unchecked::<R>(
            &client,
            cube_count,
            cube_dim,
            signal.as_tensor_arg(vectorization_input),
            spectrum_re.as_tensor_arg(vectorization),
            spectrum_im.as_tensor_arg(vectorization),
            num_samples,
            dtype,
        )
    }
}

#[cube(launch_unchecked)]
/// Kernel that loops over each window and applies the RFFT on each
pub(crate) fn rfft_kernel<F: Float>(
    signal: &Tensor<Line<F>>,
    spectrums_re: &mut Tensor<Line<F>>,
    spectrums_im: &mut Tensor<Line<F>>,
    #[comptime] num_samples: usize,
    #[define(F)] _dtype: StorageType,
) {
    let num_batch = signal.shape(0) * signal.shape(1);
    let batch_index = ABSOLUTE_POS;

    if batch_index >= num_batch {
        terminate!()
    }

    rfft_kernel_one_window(signal, spectrums_re, spectrums_im, batch_index, num_samples);
}

#[cube]
/// Applies the RFFT on one window.
/// Starts by putting all the window in shared memory, where the compute will occur
/// Then stores back the content of the shared memory
pub(crate) fn rfft_kernel_one_window<F: Float>(
    signal: &Tensor<Line<F>>,
    spectrums_re: &mut Tensor<Line<F>>,
    spectrums_im: &mut Tensor<Line<F>>,
    window_index: usize,
    #[comptime] num_samples: usize,
) {
    // The following code allow to ignore the batch index and assume only one window
    // - signal has shape: [num_samples]
    // - spectrums have shape [num_freq_bins]
    let signal_layout = BatchSignalLayout::new(signal, window_index);
    let spectrums_re_layout = BatchSignalLayout::new(spectrums_re, window_index);
    let spectrums_im_layout = BatchSignalLayout::new(spectrums_im, window_index);
    let signal_view = signal.view(signal_layout);
    let spectrums_re_view = spectrums_re.view_mut(spectrums_re_layout);
    let spectrums_im_view = spectrums_im.view_mut(spectrums_im_layout);

    // The shared memories are not vectorized because the inner FFT compute will need to work independantly on each element
    let mut spectrum_re = Array::<F>::new(num_samples).view_mut(PlainLayout::new(num_samples));
    let mut spectrum_im = Array::<F>::new(num_samples).view_mut(PlainLayout::new(num_samples));

    let line_size_input = signal_view.line_size();

    // Load all samples of the window to shared memory
    for i in 0..comptime!(num_samples / line_size_input) {
        let input = signal_view.read(i * line_size_input);

        #[unroll]
        for j in 0..line_size_input {
            let inner_index = i * line_size_input + j;
            spectrum_re[inner_index] = input[j];
            spectrum_im[inner_index] = F::cast_from(0);
        }
    }

    fft_inner_compute(&mut spectrum_re, &mut spectrum_im, FftMode::Forward);

    for i in 0..spectrums_re_view.shape() {
        // Warning: this assumes that spectrum views have lines of 1 element
        // If lines had more elements, the ith element would be duplicated as it is
        spectrums_re_view.write(i, Line::cast_from(spectrum_re[i]));
        spectrums_im_view.write(i, Line::cast_from(spectrum_im[i]));
    }
}
