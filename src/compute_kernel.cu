#include "compute_kernel.hpp"
#include <cuda/array.hpp>
#include <utils/math.hpp>

namespace vol
{
texture<Voxel, 3, cudaReadModeNormalizedFloat> tex;
texture<float4, 1, cudaReadModeElementType> transfer_tex;

__global__ void render_kernel_impl( cuda::ImageView<Pixel> out,
									Camera camera,
									Box3D box, 
									float3 inner_scale,
									float3 block_index )
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

	float u = x / float( out.width() ) * 2.f - 1.f;
	float v = y / float( out.height() ) * 2.f - 1.f;

	auto eye = Ray3D{};
	eye.o = camera.p;
	eye.d = normalize( camera.d + camera.u * u + camera.v * v );

	float tnear, tfar;
	if ( !eye.intersect( box, tnear, tfar ) ) {
		return;
	}

	auto &pixel = out.at_device( x, y );
	auto sum = pixel._;

	// if ( tnear <= pixel.t ) return;

	auto t = fmaxf( tnear, pixel.t );
	auto x0 = eye.o + eye.d * t;
	auto x1 = box.center() + ( x0 - box.center() ) * inner_scale;
	auto box_scale = 1.f / ( box.max - box.min );
	auto pos = ( x1 - box.min ) * box_scale;
	// auto p = pos;
	auto step = eye.d * tstep * box_scale * inner_scale;

	for ( int i = 0; i < max_steps; ++i ) {
		float sample = tex3D( tex, pos.x, pos.y, pos.z );
		float4 col = tex1D( transfer_tex, sample ) * density;
		sum += col * ( 1.f - sum.w );
		if ( sum.w > opacity_threshold ) break;
		t += tstep;
		if ( t > tfar ) break;
		pos += step;
	}

	pixel._ = sum;
	// pixel._ = float4{block_index.x, block_index.y, block_index.z, 0.f} * .5 + .5;
	// pixel._ = float4{t, t, t, 0.f} - 5.f;
	pixel.t = fmaxf( t, tfar );
	// out.at_device( x, y )._ = { float( i ) / 2 / max_steps + .5, 0, 0, 1 };
	// out.at_device( x, y )._ = { p.x, p.y, p.z, 1 };
}

VOL_DEFINE_CUDA_KERNEL( render_kernel, render_kernel_impl );

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
