#pragma once

#include <VMUtils/modules.hpp>

VM_BEGIN_MODULE(vol)

texture<Voxel, 3, cudaReadModeNormalizedFloat> cache;

dim3 cache_dim;
float3 inv_cache_dim;

void set_cache_dim(dim3 dim)
{
    cache_dim = dim;
    inv_cache_dim = 1.f / float3{dim.x, dim.y, dim.z};
}

void set_cache_array(cuda::Array3D<Voxel> const &arr)
{
    arr.bind_to_texture( cache );
}



VM_END_MODULE()
