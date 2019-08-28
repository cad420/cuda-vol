#include "compute_kernel.hpp"

using VoxelType = unsigned char;

texture<VoxelType, cudaReadModeNormalizedFloat> tex;

namespace vol
{
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
}  // namespace vol
