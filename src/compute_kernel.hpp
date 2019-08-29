#pragma once

#include <cuda/image.hpp>
#include <cuda/kernel.hpp>

#include <utils/volume.hpp>
#include <utils/math.hpp>

namespace vol
{
struct Pixel
{
	void write_to( unsigned char dst[ 4 ] )
	{
		float4 v = clamp( _, float4{ 0, 0, 0, 0 }, float4{ 1, 1, 1, 1 } );
		dst[ 0 ] = (unsigned char)( _.x * 255 );
		dst[ 1 ] = (unsigned char)( _.y * 255 );
		dst[ 2 ] = (unsigned char)( _.z * 255 );
		dst[ 3 ] = (unsigned char)( 255 );
	}

	float4 _;
};

using Voxel = unsigned char;

extern cuda::Kernel<void( cuda::ImageView<Pixel> view )> render_kernel;
extern void bind_texture( cuda::Array3D<Voxel> const &arr );

}  // namespace vol
