#pragma once

#include <cuda/image.hpp>
#include <cuda/kernel.hpp>

#include <utils/volume.hpp>
#include <utils/math.hpp>

namespace vol
{
template <std::size_t N = 1>
/* Nx MSAA of Microsoft Standard Simple Patterns */
struct Pixel
{
	void write_to( unsigned char dst[ 4 ] )
	{
		auto v = float4{ 0, 0, 0, 0 };
		for ( int i = 0; i != N; ++i ) {
			v += this->_[ i ].v;
		}
		v = clamp( v * 255.f / N,
				   float4{ 0, 0, 0, 0 },
				   float4{ 255, 255, 255, 255 } );
		dst[ 0 ] = (unsigned char)( v.x );
		dst[ 1 ] = (unsigned char)( v.y );
		dst[ 2 ] = (unsigned char)( v.z );
		dst[ 3 ] = (unsigned char)( 255 );
	}

	struct
	{
		float4 v;
		float t;
	} _[ N ] = { 0 };
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

struct RenderOptions
{
	VOL_DEFINE_ATTRIBUTE( Camera, camera );
	VOL_DEFINE_ATTRIBUTE( Box3D, box );
	VOL_DEFINE_ATTRIBUTE( float3, inner_scale );
	VOL_DEFINE_ATTRIBUTE( float3, block_index );
};

extern void bind_texture( cuda::Array3D<Voxel> const &arr );

extern cuda::Kernel<void( cuda::ImageView<Pixel<1>> view, RenderOptions opts )> render_kernel;
extern cuda::Kernel<void( cuda::ImageView<Pixel<2>> view, RenderOptions opts )> render_kernel_2x;
extern cuda::Kernel<void( cuda::ImageView<Pixel<4>> view, RenderOptions opts )> render_kernel_4x;
extern cuda::Kernel<void( cuda::ImageView<Pixel<8>> view, RenderOptions opts )> render_kernel_8x;
extern cuda::Kernel<void( cuda::ImageView<Pixel<16>> view, RenderOptions opts )> render_kernel_16x;

}  // namespace vol
