#include "compute_kernel.hpp"

using VoxelType = unsigned char;

texture<VoxelType, cudaReadModeNormalizedFloat> tex;

__global__ void compute_kernel_impl()
{
}

namespace vol
{

VOL_DEFINE_CUDA_KERNEL( compute_kernel, compute_kernel_impl );

}
