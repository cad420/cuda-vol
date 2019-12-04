#include <cudafx/array.hpp>

#include "compute_kernel.hpp"
#include "math.cuh"
#include "msaa_sample.cuh"
#include "cache_manager.cuh"

VM_BEGIN_MODULE( vol )

texture<float4, 1, cudaReadModeElementType> transfer_tex;

#define RAYCAST_STEP (1e-3f)
#define DENSITY (.05f)
#define OPACITY_THRESHOLD (0.95f)

template <std::size_t N>
__device__ bool screen_space_xy(cuda::ImageView<Pixel<N>> const &out, uint &x, uint &y)
{
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;

	return x < out.width() && y < out.height();
}

__device__ void block_id_offset(float3 const &p, float3 const &dim, float3 &iblkid, float3 &blkoff)
{
	float3 p_time_dim = p * dim;
	iblkid = clamp(floorf(p_time_dim), float3{0, 0, 0}, dim - 1);
	blkoff = p_time_dim - iblkid;
}

__device__ bool require_block(float3 const &blkid, float3 &id)
{
	id = blkid;
	return true;
}

__device__ float sample_block(float3 const &id, float3 const &blkoff)
{
	
}

template <std::size_t N>
__global__ void position_precompute_kernel_impl( 
	cuda::ImageView<Pixel<N>> out, Camera camera )
{
	uint x, y;
	if (!screen_space_xy( out, x, y )) { return; }

	float invw = 1.f / float( out.width() );
	float invh = 1.f / float( out.height() );

	float u = x * invw * 2.f - 1.f;
	float v = y * invh * 2.f - 1.f;

	for ( int i = 0; i != N; ++i ) {
		auto &sample = out.at_device( x, y )._[ i ];
		
		auto box = Box3D{{0, 0, 0}, {1, 1, 1}};
		auto msaa = msaa_sample<N>( i );
		auto eye = Ray3D{
			camera.p,
			normalize( camera.d +
				camera.u * ( u + msaa.x * invw ) +
				camera.v * ( v + msaa.y * invh ) )
		};
		float tnear, tfar;
		if ( eye.intersect( box, tnear, tfar ) ) {
			sample.p = eye.o + eye.d * tnear;
			sample.n = floor((tfar - tnear) / RAYCAST_STEP);
		} else {
			sample.p = float3{0, 0, 0};
			sample.n = 0;
		}
	}
}

template <std::size_t N>
__global__ void mark_position_kernel_impl(
	cuda::ImageView<Pixel<N>> out)
{
	uint x, y;
	if (!screen_space_xy( out, x, y )) { return; }

	for ( int i = 0; i != N; ++i ) {
		auto &sample = out.at_device( x, y )._[ i ];
		sample.v = float4{sample.p.x, sample.p.y, sample.p.z, 0};
	}
}

template <std::size_t N>
__global__ void mark_block_id_kernel_impl(
	cuda::ImageView<Pixel<N>> out, float3 dim)
{
	uint x, y;
	if (!screen_space_xy( out, x, y )) { return; }

	for ( int i = 0; i != N; ++i ) {
		auto &sample = out.at_device( x, y )._[ i ];
		float3 blkid, blkoff;
		block_id_offset(sample.p, dim, blkid, blkoff);
		sample.v = float4{blkoff.x, blkoff.y, blkoff.z, 0};
	}
}

template <std::size_t N>
__global__ void render_kernel_impl(
	cuda::ImageView<Pixel<N>> out, float3 eye, float3 dim)
{
	uint x, y;
	if (!screen_space_xy( out, x, y )) { return; }
	
	for ( int i = 0; i != N; ++i ) {
		auto &sample = out.at_device( x, y )._[ i ];
		float3 p = sample.p;
		float4 v = sample.v;
		float3 step = normalize(p - e) * RAYCAST_STEP;
		float3 blkid = {-1, -1, -1};
		int n = sample.n;
		for (; n; --n, p += step) {
			float3 curr_blkid, blkoff;
			block_id_offset(p, dim, curr_blkid, blkoff);
			if (curr_blkid != blkid) {
				// block changed
				blkid = curr_blkid;
				if (!require_block(blkid)) { break; }
			}
			float e = sample_tex(blkoff);
			float4 col = float4{e, e, e, 1} * DENSITY;
			v += col * (1.f - v.w);
			if (v.w > OPACITY_THRESHOLD) { n = 0; break; }
		}
		sample.p = p;
		sample.n = n;
		sample.v = v;
	}
}

#define DEFINE_KERNEL( kernel_name, N ) \
	CUFX_DEFINE_KERNEL( kernel_name ## _kernel_ ## N ## x, kernel_name ## _kernel_impl<N> )

	DEFINE_KERNEL(position_precompute, 1);
	DEFINE_KERNEL(position_precompute, 2);
	DEFINE_KERNEL(position_precompute, 4);
	DEFINE_KERNEL(position_precompute, 8);
	DEFINE_KERNEL(position_precompute, 16);

	DEFINE_KERNEL(mark_position, 1);
	DEFINE_KERNEL(mark_position, 2);
	DEFINE_KERNEL(mark_position, 4);
	DEFINE_KERNEL(mark_position, 8);
	DEFINE_KERNEL(mark_position, 16);

	DEFINE_KERNEL(mark_block_id, 1);
	DEFINE_KERNEL(mark_block_id, 2);
	DEFINE_KERNEL(mark_block_id, 4);
	DEFINE_KERNEL(mark_block_id, 8);
	DEFINE_KERNEL(mark_block_id, 16);

	DEFINE_KERNEL(render, 1);
	DEFINE_KERNEL(render, 2);
	DEFINE_KERNEL(render, 4);
	DEFINE_KERNEL(render, 8);
	DEFINE_KERNEL(render, 16);

VM_EXPORT
{

#define DEFINE_MSAA_RENDER_GROUP( N )             \
	RenderGroup<N> render_group_##N##x = {        \
		position_precompute_kernel_##N##x,        \
		mark_block_id_kernel_##N##x,              \
	}

	DEFINE_MSAA_RENDER_GROUP( 1 );
	DEFINE_MSAA_RENDER_GROUP( 2 );
	DEFINE_MSAA_RENDER_GROUP( 4 );
	DEFINE_MSAA_RENDER_GROUP( 8 );
	DEFINE_MSAA_RENDER_GROUP( 16 );

}

// template <std::size_t N>
// __global__ void render_kernel_impl( cuda::ImageView<Pixel<N>> out, RenderOptions opts )
// {
// 	// const int max_steps = 500;
// 	// const float tstep = 0.01f;
// 	// const float opacity_threshold = 0.95f;
// 	// const float density = .05f;

// 	uint x = blockIdx.x * blockDim.x + threadIdx.x;
// 	uint y = blockIdx.y * blockDim.y + threadIdx.y;

// 	if ( x >= out.width() || y >= out.height() ) {
// 		return;
// 	}

// 	float invw = 1.f / float( out.width() );
// 	float invh = 1.f / float( out.height() );

// 	float u = x * invw * 2.f - 1.f;
// 	float v = y * invh * 2.f - 1.f;

// 	for ( int i = 0; i != N; ++i ) {
// 		auto msaa = msaa_sample<N>( i );

// 		auto eye = Ray3D{
// 			opts.camera.p,
// 			normalize( opts.camera.d +
// 				opts.camera.u * ( u + msaa.x * invw ) +
// 				opts.camera.v * ( v + msaa.y * invh ) )
// 		};

		
// 		const auto box = Box3D{{-1, -1, -1}, {1, 1, 1}};

// 		float tnear, tfar;
// 		if ( !eye.intersect( box, tnear, tfar ) ) {
// 			return;
// 		}

// 		auto &sample = out.at_device( x, y )._[ i ];

// 		sample.src = eye.o + eye.d * tnear;
// 		sample.dst = eye.o + eye.d * tfar;
// 		// auto sum = sample.v;

// 		// auto t = fmaxf( tnear, sample.t );
// 		// auto x0 = eye.o + eye.d * t;
// 		// auto x1 = opts.box.center() + ( x0 - opts.box.center() ) * opts.inner_scale;
// 		// auto box_scale = 1.f / ( opts.box.max - opts.box.min );
// 		// auto pos = ( x1 - opts.box.min ) * box_scale;
// 		// // auto p = pos;
// 		// auto step = eye.d * tstep * box_scale * opts.inner_scale;

// 		// for ( int i = 0; i < max_steps; ++i ) {
// 		// 	float sample = tex3D( tex, pos.x, pos.y, pos.z );
// 		// 	float4 col = tex1D( transfer_tex, sample ) * density;
// 		// 	sum += col * ( 1.f - sum.w );
// 		// 	if ( sum.w > opacity_threshold ) break;
// 		// 	t += tstep;
// 		// 	if ( t > tfar ) break;
// 		// 	pos += step;
// 		// }

// 		// sample.v = sum;
// 		// sample.t = fmaxf( t, tfar );
// 	}
// }

// VM_EXPORT
// {
// 	CUFX_DEFINE_KERNEL( render_kernel, render_kernel_impl<1> );
// 	CUFX_DEFINE_KERNEL( render_kernel_2x, render_kernel_impl<2> );
// 	CUFX_DEFINE_KERNEL( render_kernel_4x, render_kernel_impl<4> );
// 	CUFX_DEFINE_KERNEL( render_kernel_8x, render_kernel_impl<8> );
// 	CUFX_DEFINE_KERNEL( render_kernel_16x, render_kernel_impl<16> );
// }

namespace _
{
static int __ = [] {
	tex.normalized = true;
	tex.filterMode = cudaFilterModeLinear;
	tex.addressMode[ 0 ] = cudaAddressModeClamp;
	tex.addressMode[ 1 ] = cudaAddressModeClamp;

	transfer_tex.filterMode = cudaFilterModeLinear;
	transfer_tex.normalized = true;
	transfer_tex.addressMode[ 0 ] = cudaAddressModeClamp;

	static float4 transfer_fn[] = {
		{ 0., 0., 0., 0. },
		{ 1., 0., 0., 1. },
		{ 1., .5, 0., 1. },
		{ 1., 1., 0., 1. },
		{ 0., 1., 0., 1. },
		{ 0., 1., 1., 1. },
		{ 0., 0., 1., 1. },
		{ 1., 0., 1., 1. },
		{ 0., 0., 0., 0. },
	};
	static cuda::Array1D<float4>
	  transfer_arr( sizeof( transfer_fn ) / sizeof( transfer_fn[ 0 ] ) );
	auto transfer_fn_view = cuda::MemoryView1D<float4>( transfer_fn, transfer_arr.size() );
	cuda::memory_transfer( transfer_arr, transfer_fn_view ).launch().unwrap();
	transfer_arr.bind_to_texture( transfer_tex );

	return 0;
}();
}

VM_EXPORT
{
	void bind_texture( cuda::Array3D<Voxel> const &arr )
	{
		arr.bind_to_texture( tex );
	}
}

VM_END_MODULE()
