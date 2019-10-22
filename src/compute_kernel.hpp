#pragma once

#include <cudafx/image.hpp>
#include <cudafx/kernel.hpp>
#include <VMUtils/modules.hpp>
#include <VMUtils/attributes.hpp>
#include "utils/math.hpp"

VM_BEGIN_MODULE( vol )

namespace cuda = cufx;

VM_EXPORT
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
		VM_DEFINE_ATTRIBUTE( float3, p );
		VM_DEFINE_ATTRIBUTE( float3, d );
		VM_DEFINE_ATTRIBUTE( float3, u );
		VM_DEFINE_ATTRIBUTE( float3, v );

	public:
		struct Builder
		{
			VM_DEFINE_ATTRIBUTE( float3, pos ) = float3{ 0, 0, 4 };
			VM_DEFINE_ATTRIBUTE( float3, up ) = float3{ 0, -1, 0 };
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
		VM_DEFINE_ATTRIBUTE( Camera, camera );
		VM_DEFINE_ATTRIBUTE( Box3D, box );
		VM_DEFINE_ATTRIBUTE( float3, inner_scale );
		VM_DEFINE_ATTRIBUTE( float3, block_index );
	};

	extern void bind_texture( cuda::Array3D<Voxel> const &arr );

	extern cuda::Kernel<void( cuda::ImageView<Pixel<1>> view, RenderOptions opts )> render_kernel;
	extern cuda::Kernel<void( cuda::ImageView<Pixel<2>> view, RenderOptions opts )> render_kernel_2x;
	extern cuda::Kernel<void( cuda::ImageView<Pixel<4>> view, RenderOptions opts )> render_kernel_4x;
	extern cuda::Kernel<void( cuda::ImageView<Pixel<8>> view, RenderOptions opts )> render_kernel_8x;
	extern cuda::Kernel<void( cuda::ImageView<Pixel<16>> view, RenderOptions opts )> render_kernel_16x;
}

VM_END_MODULE()
