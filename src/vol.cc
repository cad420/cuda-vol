#include <iostream>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cxxopts.hpp>

#include <cudafx/device.hpp>
#include <compute_kernel.hpp>
#include <utils/volume.hpp>

using namespace std;
using namespace vol;
namespace cuda = cufx;

int round_up_div( int a, int b )
{
	return a % b == 0 ? a / b : a / b + 1;
}

int main( int argc, char **argv )
{
	cxxopts::Options options( "vol", "Cuda based offline volume renderer" );
	options.add_options()( "i,input", "input compressed file name", cxxopts::value<string>() )(
	  "h,help", "show this help message" )(
	  "d,dim", "output image dim", cxxopts::value<unsigned>()->default_value( "2048" ) )(
	  "o,output", "place the output image into <file>", cxxopts::value<string>()->default_value( "a.png" ) )(
	  "s,sample", "msaa sample points", cxxopts::value<std::size_t>()->default_value( "1" ) )(
	  "no-hwaccel", "disable hardware acceleration" )(
	  "x", "x", cxxopts::value<float>()->default_value( "0" ) )(
	  "y", "y", cxxopts::value<float>()->default_value( "0" ) )(
	  "z", "z", cxxopts::value<float>()->default_value( "4" ) );

	auto opts = options.parse( argc, argv );
	if ( opts.count( "h" ) ) {
		cout << options.help() << endl;
		exit( 0 );
	}
	auto in = opts[ "i" ].as<string>();
	auto out = opts[ "o" ].as<string>();
	auto img_size = opts[ "d" ].as<unsigned>();

	auto center = float3{ opts[ "x" ].as<float>(),
						  opts[ "y" ].as<float>(),
						  opts[ "z" ].as<float>() };

	auto camera = Camera::Builder{}
					.set_pos( center )
					.build();
	vm::println( "camera: {}", camera );

	cuda::PendingTasks tasks;

	auto devices = cuda::Device::scan();
	auto device = devices[ 0 ];
	bool direct_gpu_memory = !opts.count( "no-hwaccel" );

	auto nx_msaa_impl = [&]( int nx, auto pixel, auto &grp ) {
		using Pixel = decltype( pixel );
		cout << "sample count: " << nx << "x" << endl;
		cufx::Stream zero_stream;

		cuda::Image<Pixel> image( img_size, img_size );
		auto device_swap = device.alloc_image_swap( image );
		auto img_view = image.view().with_device_memory( device_swap.second );
		tasks.add( img_view.copy_to_device().launch_async( zero_stream ) );

		auto kernel_block_dim = dim3( 32, 32 );
		auto launch_info = cuda::KernelLaunchInfo{}
							 .set_device( devices[ 0 ] )
							 .set_grid_dim( round_up_div( img_view.width(), kernel_block_dim.x ),
											round_up_div( img_view.height(), kernel_block_dim.y ) )
							 .set_block_dim( kernel_block_dim );
		tasks.add( grp.position_precompute( launch_info, img_view, camera ).launch_async( zero_stream ) );
		tasks.add( grp.mark_block_id( launch_info, img_view, float3{ 2, 2, 2 } ).launch_async( zero_stream ) );
		for ( auto res : tasks.wait() ) {
			res.unwrap();
		}

		// auto volume = Volume<Voxel>::from_compressed( in );
		// auto grid_dim = volume.dim();
		// cout << "volume grid dim: " << grid_dim << endl;
		// float grid_dim_max = float( max( grid_dim.x, max( grid_dim.y, grid_dim.z ) ) );
		// auto scale = volume.scale();
		// cout << "volume scale: " << scale << endl;
		// auto block_dim = volume.block_dim();
		// cout << "volume block dim: " << block_dim << endl;
		// auto padding = volume.padding();
		// cout << "volume block padding: " << padding << endl;
		// auto padding_f = float3{ float( padding.x ),
		// 						 float( padding.y ),
		// 						 float( padding.z ) };
		// auto block_dim_f = float3{ float( block_dim.width ),
		// 						   float( block_dim.height ),
		// 						   float( block_dim.depth ) };
		// auto inner_scale = 1.f - 2.f * padding_f / block_dim_f;
		// auto blocks = volume.get_blocks();
		// stable_sort(
		//   blocks.begin(), blocks.end(),
		//   [&]( ArchivedVolumeBlock<Voxel> const &a, ArchivedVolumeBlock<Voxel> const &b ) {
		// 	  auto x = float3{ float( a.index().x ),
		// 					   float( a.index().y ),
		// 					   float( a.index().z ) };
		// 	  auto y = float3{ float( b.index().x ),
		// 					   float( b.index().y ),
		// 					   float( b.index().z ) };
		// 	  auto px = ( x + .5f ) / grid_dim_max * scale * 2.f - 1.f;
		// 	  auto py = ( y + .5f ) / grid_dim_max * scale * 2.f - 1.f;
		// 	  return length( px - camera.p ) < length( py - camera.p );
		//   } );

		// cuda::Array3D<Voxel> block_arr[ 2 ] = { device.alloc_arraynd<Voxel, 3>( block_dim ),
		// 										device.alloc_arraynd<Voxel, 3>( block_dim ) };
		// cuda::Stream swap[ 2 ];
		// vector<cuda::GlobalMemory> block_swap;
		// if ( direct_gpu_memory ) {
		// 	vm::println( "hardware acceleration enabled" );
		// 	for ( int i = 0; i != 2; ++i ) {
		// 		block_swap.emplace_back( device.alloc_global( block_dim.size() * 2 ) );
		// 	}
		// }

		// int curr_swap = 0;
		// tasks.wait();
		// int done_blocks = 0;

		// for ( auto &arch : blocks ) {
		// 	vm::println( "rendering block: {}, {} out of {}", arch.index(), ++done_blocks, blocks.size() );
		// 	auto &block = block_swap[ curr_swap ];
		// 	auto &arr = block_arr[ curr_swap ];
		// 	auto &stream = swap[ curr_swap ];

		// 	thread_local vector<unsigned char> host_buffer;

		// 	vm::Option<cufx::MemoryView1D<unsigned char>> dst_1d;
		// 	if ( direct_gpu_memory ) {
		// 		dst_1d = block.view_1d<unsigned char>( block.size() );
		// 	} else {
		// 		host_buffer.resize( block_dim.size() * 2 );
		// 		dst_1d = cufx::MemoryView1D<unsigned char>( host_buffer.data(), host_buffer.size() );
		// 	}
		// 	auto dim = arch.unarchive_into( dst_1d.value() );

		// 	auto view_info = cufx::MemoryView2DInfo{}
		// 					   .set_stride( dim.width * sizeof( char ) )
		// 					   .set_width( dim.width )
		// 					   .set_height( dim.height );
		// 	vm::Option<cufx::MemoryView3D<unsigned char>> src_3d;
		// 	if ( direct_gpu_memory ) {
		// 		src_3d = block.view_3d<unsigned char>( view_info, dim );
		// 	} else {
		// 		src_3d = cufx::MemoryView3D<unsigned char>( host_buffer.data(), view_info, dim );
		// 	}
		// 	cufx::memory_transfer( arr, src_3d.value() )
		// 	  .launch_async( stream );
		// 	auto idx = float3{ float( arch.index().x ),
		// 					   float( arch.index().y ),
		// 					   float( arch.index().z ) };
		// 	// auto box = Box3D{}
		// 	// 			 .set_min( idx / grid_dim_max * scale * 2.f - 1.f )
		// 	// 			 .set_max( ( idx + 1.f ) / grid_dim_max * scale * 2.f - 1.f );
		// 	// cout << box << endl;
		// 	swap[ 1 - curr_swap ].wait().unwrap();
		// 	bind_texture( block_arr[ curr_swap ] );
		// 	auto opts = RenderOptions{}
		// 				  .set_camera( camera )
		// 				  //   .set_box( box )
		// 				  .set_inner_scale( inner_scale )
		// 				  .set_block_index( idx / grid_dim_max );
		// 	grp.position_precompute( launch_info, img_view, opts )
		// 	  .launch_async( stream );
		// 	curr_swap = 1 - curr_swap;
		// }
		// swap[ 0 ].wait().unwrap();
		// swap[ 1 ].wait().unwrap();

		cout << "render finished" << endl;

		img_view.copy_from_device().launch();
		image.dump( out );
		cout << "written image " << out << endl;
		return 0;
	};

	switch ( auto nx = opts[ "s" ].as<std::size_t>() ) {
	case 1: return nx_msaa_impl( 1, Pixel<1>{}, render_group_1x );
	case 2: return nx_msaa_impl( 2, Pixel<2>{}, render_group_2x );
	case 4: return nx_msaa_impl( 4, Pixel<4>{}, render_group_4x );
	case 8: return nx_msaa_impl( 8, Pixel<8>{}, render_group_8x );
	case 16: return nx_msaa_impl( 16, Pixel<16>{}, render_group_16x );
	default: cout << "invalid sample count: " << nx << endl;
	}
}
