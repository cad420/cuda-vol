#include "compute_kernel.hpp"

#include <cuda/array.hpp>

namespace vol
{
texture<Voxel, 3, cudaReadModeNormalizedFloat> tex;

__global__ void compute_kernel_impl( cuda::ImageView<Pixel> view )
{
	for ( int i = 0; i != view.height(); ++i ) {
		for ( int j = 0; j != view.width(); ++j ) {
			view.at_device( j, i ) = { j * 255 / view.width(), i * 255 / view.height(),
									   127, 255 };
		}
	}
}

VOL_DEFINE_CUDA_KERNEL( compute_kernel, compute_kernel_impl );

void bind_texture( cuda::Array3D<Voxel> const &arr )
{
	arr.bind_to_texture( tex );
}

}  // namespace vol
