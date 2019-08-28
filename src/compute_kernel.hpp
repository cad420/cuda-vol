#pragma once

#include <cuda/image.hpp>
#include <cuda/kernel.hpp>

#include <utils/volume.hpp>

namespace vol
{
struct Pixel
{
	void write_to( char dst[ 4 ] ) { memcpy( dst, x, sizeof( char ) * 4 ); }

	unsigned char x[ 4 ];
};

using Voxel = unsigned char;

extern cuda::Kernel<void( cuda::ImageView<Pixel> view )> compute_kernel;
extern void bind_texture( cuda::Array3D<Voxel> const &arr );

}  // namespace vol
