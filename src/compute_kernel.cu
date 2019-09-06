#include "compute_kernel.hpp"
#include <cuda/array.hpp>
#include <utils/math.hpp>

namespace vol
{

texture<Voxel, 3, cudaReadModeNormalizedFloat> tex;
texture<float4, 1, cudaReadModeElementType> transfer_tex;

template <std::size_t N>
__device__ float2 msaa_sample( int idx );

#define MS_SAMPLE( x, y ) { float( x / 10.f / .8f ), float( y / 10.f / .8f ) }

template <>
__device__ float2 msaa_sample<1>( int )
{
	return float2{ 0, 0 };
}

template <>
__device__ float2 msaa_sample<2>( int i )
{
	static float2 _[] = { MS_SAMPLE( 4, 4 ),
	                      MS_SAMPLE( -4, -4 ) };
	return _[ i ];
}

template <>
__device__ float2 msaa_sample<4>( int i )
{
	static float2 _[] = { MS_SAMPLE( -2, -6 ),
	                      MS_SAMPLE( 6, -2 ),
	                      MS_SAMPLE( -6, 2 ),
	                      MS_SAMPLE( 2, 6 ) };
	return _[ i ];
}

template <>
__device__ float2 msaa_sample<8>( int i )
{
	static float2 _[] = { MS_SAMPLE( 1, -3 ),
	                      MS_SAMPLE( -1, 3 ),
	                      MS_SAMPLE( 5, 1 ),
	                      MS_SAMPLE( -3, -5 ),
	                      MS_SAMPLE( -5, 5 ),
	                      MS_SAMPLE( -7, -1 ),
	                      MS_SAMPLE( 3, 7 ),
	                      MS_SAMPLE( 7, -7 ) };
	return _[ i ];
}

template <>
__device__ float2 msaa_sample<16>( int i )
{
	static float2 _[] = { MS_SAMPLE( 1, 1 ),
	                      MS_SAMPLE( -1, -3 ),
	                      MS_SAMPLE( -3, 2 ),
	                      MS_SAMPLE( 4, -1 ),
	                      MS_SAMPLE( -5, -2 ),
	                      MS_SAMPLE( 2, 5 ),
	                      MS_SAMPLE( 5, 3 ),
	                      MS_SAMPLE( 3, -5 ),
	                      MS_SAMPLE( -2, 6 ),
	                      MS_SAMPLE( 0, -7 ),
	                      MS_SAMPLE( -4, -6 ),
	                      MS_SAMPLE( -6, 4 ),
	                      MS_SAMPLE( -8, 0 ),
	                      MS_SAMPLE( 7, -4 ),
	                      MS_SAMPLE( 6, 7 ),
	                      MS_SAMPLE( -7, -8 ) };
	return _[ i ];
}

template <std::size_t N>
__global__ void render_kernel_impl( cuda::ImageView<Pixel<N>> out, RenderOptions opts )
{
	const int max_steps = 500;
	const float tstep = 0.01f;
	const float opacity_threshold = 0.95f;
	const float density = .05f;

	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if ( x >= out.width() || y >= out.height() ) {
		return;
	}

	float invw = 1.f / float( out.width() );
	float invh = 1.f / float( out.height() );

	float u = x * invw * 2.f - 1.f;
	float v = y * invh * 2.f - 1.f;

	for ( int i = 0; i != N; ++i ) {

		auto msaa = msaa_sample<N>( i );

		auto eye = Ray3D{};
		eye.o = opts.camera.p;
		eye.d = normalize( opts.camera.d + 
						   opts.camera.u * ( u + msaa.x * invw ) + 
						   opts.camera.v * ( v + msaa.y * invh ) );

		float tnear, tfar;
		if ( !eye.intersect( opts.box, tnear, tfar ) ) {
			return;
		}

		auto &sample = out.at_device( x, y )._[ i ];
		auto sum = sample.v;

		auto t = fmaxf( tnear, sample.t );
		auto x0 = eye.o + eye.d * t;
		auto x1 = opts.box.center() + ( x0 - opts.box.center() ) * opts.inner_scale;
		auto box_scale = 1.f / ( opts.box.max - opts.box.min );
		auto pos = ( x1 - opts.box.min ) * box_scale;
		// auto p = pos;
		auto step = eye.d * tstep * box_scale * opts.inner_scale;

		for ( int i = 0; i < max_steps; ++i ) {
			float sample = tex3D( tex, pos.x, pos.y, pos.z );
			float4 col = tex1D( transfer_tex, sample ) * density;
			sum += col * ( 1.f - sum.w );
			if ( sum.w > opacity_threshold ) break;
			t += tstep;
			if ( t > tfar ) break;
			pos += step;
		}

		sample.v = sum;
		sample.t = fmaxf( t, tfar );

	}
	// pixel._ = float4{block_index.x, block_index.y, block_index.z, 0.f} * .5 + .5;
	// pixel._ = float4{t, t, t, 0.f} - 5.f;
}

VOL_DEFINE_CUDA_KERNEL( render_kernel, render_kernel_impl<1> );
VOL_DEFINE_CUDA_KERNEL( render_kernel_2x, render_kernel_impl<2> );
VOL_DEFINE_CUDA_KERNEL( render_kernel_4x, render_kernel_impl<4> );
VOL_DEFINE_CUDA_KERNEL( render_kernel_8x, render_kernel_impl<8> );
VOL_DEFINE_CUDA_KERNEL( render_kernel_16x, render_kernel_impl<16> );

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

void bind_texture( cuda::Array3D<Voxel> const &arr )
{
	arr.bind_to_texture( tex );
}

}  // namespace vol
