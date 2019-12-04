#pragma once

#include <cudafx/image.hpp>
#include <cudafx/kernel.hpp>
#include <VMUtils/modules.hpp>
#include <VMUtils/attributes.hpp>
#include "utils/json_cuda.hpp"
#include <nv/helper_math.h>

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
			auto v = float3{ 0, 0, 0 };
			for ( int i = 0; i != N; ++i ) {
				auto &v1 = this->_[ i ].v;
				v += float3{ v1.x, v1.y, v1.z };
			}
			v = clamp( v * 255.f / N,
					   float3{ 0, 0, 0 },
					   float3{ 255, 255, 255 } );
			dst[ 0 ] = (unsigned char)( v.x );
			dst[ 1 ] = (unsigned char)( v.y );
			dst[ 2 ] = (unsigned char)( v.z );
			dst[ 3 ] = (unsigned char)( 255 );
		}

		struct
		{
			// local space position
			float3 p;
			// remaining steps
			int n;
			// value
			float4 v;
			// float t;
		} _[ N ] = { 0 };
	};

	struct Camera : vm::json::Serializable<Camera>
	{
		VM_JSON_FIELD( float3, p );
		VM_JSON_FIELD( float3, d );
		VM_JSON_FIELD( float3, u );
		VM_JSON_FIELD( float3, v );

	public:
		struct Builder : vm::json::Serializable<Builder>
		{
			VM_JSON_FIELD( float3, pos ) = { 0, 0, 4 };
			VM_JSON_FIELD( float3, up ) = { 0, -1, 0 };
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

		friend std::ostream &operator<<( std::ostream &os, Camera const &cc )
		{
			vm::fprint( os, "{}", std::make_tuple( cc.p, cc.d, cc.u, cc.v ) );
			return os;
		}
	};

	using Voxel = unsigned char;

	void set_cache_dim()
	{
	}

	struct RenderOptions : vm::json::Serializable<RenderOptions>
	{
		VM_JSON_FIELD( Camera, camera );
		// VM_JSON_FIELD( Box3D, box );
		VM_JSON_FIELD( float3, inner_scale );
		VM_JSON_FIELD( float3, block_index );
	};

	extern void bind_texture( cuda::Array3D<Voxel> const &arr );

	template <std::size_t N>
	struct RenderGroup
	{
		cuda::Kernel<void( cuda::ImageView<Pixel<N>> view, Camera camera )> position_precompute;
		cuda::Kernel<void( cuda::ImageView<Pixel<N>> view, float3 dim )> mark_block_id;
	};

#define DECLARE_MSAA_RENDER_GROUP( N ) \
	extern RenderGroup<N> render_group_##N##x

	DECLARE_MSAA_RENDER_GROUP( 1 );
	DECLARE_MSAA_RENDER_GROUP( 2 );
	DECLARE_MSAA_RENDER_GROUP( 4 );
	DECLARE_MSAA_RENDER_GROUP( 8 );
	DECLARE_MSAA_RENDER_GROUP( 16 );

	// extern cuda::Kernel<void( cuda::ImageView<Pixel<1>> view, RenderOptions opts )> render_kernel;
	// extern cuda::Kernel<void( cuda::ImageView<Pixel<2>> view, RenderOptions opts )> render_kernel_2x;
	// extern cuda::Kernel<void( cuda::ImageView<Pixel<4>> view, RenderOptions opts )> render_kernel_4x;
	// extern cuda::Kernel<void( cuda::ImageView<Pixel<8>> view, RenderOptions opts )> render_kernel_8x;
	// extern cuda::Kernel<void( cuda::ImageView<Pixel<16>> view, RenderOptions opts )> render_kernel_16x;
}

VM_END_MODULE()
