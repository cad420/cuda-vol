#include "compute_kernel.hpp"

__global__ void compute_kernel_impl()
{
}

namespace vol
{
VOL_DEFINE_CUDA_KERNEL( compute_kernel, compute_kernel_impl );

}
