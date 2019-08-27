#pragma once

#include <utils/cuda_adapter.hpp>

namespace vol
{
extern cuda::Kernel<void()> compute_kernel;

}
