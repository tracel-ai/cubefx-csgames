use cubecl::prelude::*;
use cubecl::std::tensor::layout::plain::PlainLayout;
use cubecl::std::tensor::{
    AsView as _, AsViewExpand, AsViewMut as _, AsViewMutExpand, TensorHandle,
};

use crate::cube::{BatchSignalLayout, FftMode, cube_selection, fft_inner_compute};

/// Inverse Real-valued Fast Fourier Transform kernel.
///
/// Creates signal tensor
/// then launches the IRFFT kernel to fill it with the right values
pub fn irfft<R: Runtime>(
    spectrum_re: TensorHandle<R>,
    spectrum_im: TensorHandle<R>,
    dtype: StorageType,
) -> TensorHandle<R> {
    // Assumes fft always done on last dim
    let dim = spectrum_re.shape.len() - 1;
    assert!(
        spectrum_re.shape == spectrum_im.shape,
        "Spectrum's real and imaginary parts should be the same shape, got {:?} and {:?}",
        spectrum_re.shape,
        spectrum_im.shape
    );

    let client = <R as Runtime>::client(&Default::default());

    let mut signal_shape = spectrum_re.shape.clone();
    signal_shape[dim] = (spectrum_re.shape[dim] - 1) * 2;
    let num_elems = signal_shape.iter().product::<usize>();
    let signal =
        TensorHandle::new_contiguous(signal_shape, client.empty(num_elems * dtype.size()), dtype);

    irfft_launch::<R>(
        &client,
        spectrum_re.as_ref(),
        spectrum_im.as_ref(),
        signal.as_ref(),
        dtype,
    )
    .unwrap();

    signal
}

/// Launches the IRFFT with the specified Cube Count, Cube Dim and vectorization (line size)
pub fn irfft_launch<R: Runtime>(
    client: &ComputeClient<R>,
    spectrum_re: TensorHandleRef<R>,
    spectrum_im: TensorHandleRef<R>,
    signal: TensorHandleRef<R>,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let vectorization = 1;

    let num_iter = signal.shape[0] * signal.shape[1];
    let (cube_dim, cube_count, is_gpu) =
        cube_selection(&client.properties().hardware, num_iter, true);
    let num_samples = *signal.shape.last().unwrap();

    unsafe {
        irfft_kernel::launch_unchecked::<R>(
            &client,
            cube_count,
            cube_dim,
            spectrum_re.as_tensor_arg(vectorization),
            spectrum_im.as_tensor_arg(vectorization),
            signal.as_tensor_arg(vectorization),
            num_samples,
            is_gpu,
            dtype,
        )
    }
}

#[cube(launch_unchecked)]
/// Kernel that loops over each window and applies the IRFFT on each
pub(crate) fn irfft_kernel<F: Float>(
    spectrums_re: &Tensor<Line<F>>,
    spectrums_im: &Tensor<Line<F>>,
    signal: &mut Tensor<Line<F>>,
    #[comptime] num_samples: usize,
    #[comptime] is_gpu: bool,
    #[define(F)] _dtype: StorageType,
) {
    let num_batch = signal.shape(0) * signal.shape(1);
    let batch_index = ABSOLUTE_POS;

    if batch_index >= num_batch {
        terminate!()
    }

    irfft_kernel_one_batch(
        spectrums_re,
        spectrums_im,
        signal,
        batch_index,
        num_samples,
        is_gpu,
    );
}

#[cube(launch)]
/// Applies the IRFFT on one window.
/// Starts by putting all the window in shared memory, where the compute will occur
/// Then stores back the content of the shared memory
/// There are a few extra steps for normalization compared to forward RFFT
pub(crate) fn irfft_kernel_one_batch<F: Float>(
    spectrums_re: &Tensor<Line<F>>,
    spectrums_im: &Tensor<Line<F>>,
    signal: &mut Tensor<Line<F>>,
    window_index: usize,
    #[comptime] num_samples: usize,
    #[comptime] is_gpu: bool,
) {
    // The following code allow to ignore the batch index and assume only one window
    // - spectrums have shape: [num_freq_bins]
    // - signal has shape: [num_samples]
    let spectrums_re_layout = BatchSignalLayout::new(spectrums_re, window_index);
    let spectrums_im_layout = BatchSignalLayout::new(spectrums_im, window_index);
    let signal_layout = BatchSignalLayout::new(signal, window_index);
    let spectrums_re_view = spectrums_re.view(spectrums_re_layout);
    let spectrums_im_view = spectrums_im.view(spectrums_im_layout);
    let signal_view = signal.view_mut(signal_layout);

    let num_freq_bins = spectrums_re_view.shape();

    let (mut spectrum_re, mut spectrum_im) = match is_gpu {
        true => (
            SharedMemory::<F>::new(num_samples).view_mut(PlainLayout::new(num_samples)),
            SharedMemory::<F>::new(num_samples).view_mut(PlainLayout::new(num_samples)),
        ),
        false => (
            Array::<F>::new(num_samples).view_mut(PlainLayout::new(num_samples)),
            Array::<F>::new(num_samples).view_mut(PlainLayout::new(num_samples)),
        ),
    };

    // Load all the frequency bins to shared memory
    for i in 0..num_freq_bins {
        // Warning: this assumes that spectrum views have lines of 1 element
        // For larger lines, iterate over the line's content
        // You can get the line_size of a tensor/view with .line_size()
        spectrum_re[i] = spectrums_re_view.read(i)[0];
        spectrum_im[i] = spectrums_im_view.read(i)[0];
    }

    // Fill the Hermitian-conjugate mirrored bins
    for k in 1..num_freq_bins - 1 {
        spectrum_re[num_samples - k] = spectrum_re[k];
        spectrum_im[num_samples - k] = -spectrum_im[k]; // conjugate
    }

    // Run inverse FFT
    fft_inner_compute(&mut spectrum_re, &mut spectrum_im, FftMode::Inverse);

    // Normalize by number of samples
    for i in 0..num_samples {
        let output = spectrum_re[i] / F::cast_from(num_samples);
        signal_view.write(i, Line::cast_from(output));
    }
}
