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

struct Camera
{
	VOL_DEFINE_ATTRIBUTE( float3, p );
	VOL_DEFINE_ATTRIBUTE( float3, d );
	VOL_DEFINE_ATTRIBUTE( float3, u );
	VOL_DEFINE_ATTRIBUTE( float3, v );

public:
	struct Builder
	{
		VOL_DEFINE_ATTRIBUTE( float3, pos ) = float3{ 0, 0, 4 };
		VOL_DEFINE_ATTRIBUTE( float3, up ) = float3{ 0, -1, 0 };
		// using a fixed fov of tg(theta) = 1/2

	public:
		Camera build()
		{
			auto d = normalize( -pos );
			auto u = normalize( cross( d, up ) );
			auto v = normalize( cross( d, u ) );
			return Camera{}
			  .set_p( pos )
			  .set_d( d * 2 )
			  .set_u( u )
			  .set_v( v );
		}
	};
};

using Voxel = unsigned char;

extern cuda::Kernel<void( cuda::ImageView<Pixel> view,
						  Camera camera,
						  Box3D box,
						  float3 inner_scale )>
  render_kernel;
extern void bind_texture( cuda::Array3D<Voxel> const &arr );

}  // namespace vol
