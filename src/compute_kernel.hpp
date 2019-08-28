#pragma once

#include <cuda/image.hpp>
#include <cuda/kernel.hpp>

namespace vol
{
struct Pixel
{
	void write_to( char dst[ 4 ] ) { memcpy( dst, x, sizeof( char ) * 4 ); }

	unsigned char x[ 4 ];
};

extern cuda::Kernel<void( cuda::ImageView<Pixel> view )> compute_kernel;
}  // namespace vol
