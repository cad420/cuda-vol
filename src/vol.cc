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
	cuda::PendingTasks tasks;

	auto devices = cuda::Device::scan();
	auto device = devices[ 0 ];

	auto nx_msaa_impl = [&]( int nx, auto pixel, auto &kernel ) {
		using Pixel = decltype( pixel );
		cout << "sample count: " << nx << "x" << endl;

		cuda::Image<Pixel> image( img_size, img_size );
		auto device_swap = device.alloc_image_swap( image );
		auto img_view = image.view().with_device_memory( device_swap.second );
		tasks.add( img_view.copy_to_device().launch_async() );

		auto volume = Volume<Voxel>::from_compressed( in );
		auto grid_dim = volume.dim();
		cout << "volume grid dim: " << grid_dim << endl;
		float grid_dim_max = float( max( grid_dim.x, max( grid_dim.y, grid_dim.z ) ) );
		auto scale = volume.scale();
		cout << "volume scale: " << scale << endl;
		auto block_dim = volume.block_dim();
		cout << "volume block dim: " << block_dim << endl;
		auto padding = volume.padding();
		cout << "volume block padding: " << padding << endl;
		auto padding_f = float3{ float( padding.x ),
								 float( padding.y ),
								 float( padding.z ) };
		auto block_dim_f = float3{ float( block_dim.width ),
								   float( block_dim.height ),
								   float( block_dim.depth ) };
		auto inner_scale = 1.f - 2.f * padding_f / block_dim_f;
		auto blocks = volume.get_blocks();
		stable_sort(
		  blocks.begin(), blocks.end(),
		  [&]( ArchivedVolumeBlock<Voxel> const &a, ArchivedVolumeBlock<Voxel> const &b ) {
			  auto x = float3{ float( a.index().x ),
							   float( a.index().y ),
							   float( a.index().z ) };
			  auto y = float3{ float( b.index().x ),
							   float( b.index().y ),
							   float( b.index().z ) };
			  auto px = ( x + .5f ) / grid_dim_max * scale * 2.f - 1.f;
			  auto py = ( y + .5f ) / grid_dim_max * scale * 2.f - 1.f;
			  return length( px - camera.p ) < length( py - camera.p );
		  } );

		auto kernel_block_dim = dim3( 32, 32 );
		auto launch_info = cuda::KernelLaunchInfo{}
							 .set_device( devices[ 0 ] )
							 .set_grid_dim( round_up_div( img_view.width(), kernel_block_dim.x ),
											round_up_div( img_view.height(), kernel_block_dim.y ) )
							 .set_block_dim( kernel_block_dim );
		cuda::Array3D<Voxel> block_arr[ 2 ] = { device.alloc_arraynd<Voxel, 3>( block_dim ),
												device.alloc_arraynd<Voxel, 3>( block_dim ) };
		cuda::Stream swap[ 2 ];
		vector<cuda::GlobalMemory> block_swap = { device.alloc_global( block_dim.size() ), 
												  device.alloc_global( block_dim.size() ) };
		int curr_swap = 0;
		tasks.wait();
		for ( auto &arch : blocks ) {
			cout << "rendering block: " << arch.index() << endl;
			auto &block = block_swap[ curr_swap ];
			auto &arr = block_arr[ curr_swap ];
			auto &stream = swap[ curr_swap ];

			// if (arch.index().x == 0 && arch.index().y == 0 && arch.index().z == 0) break;

			auto dim = arch.unarchive_into( block.view_1d<unsigned char>( block.size() ) );

			// vector<unsigned char> vec(block_dim.size() * 2);
			// auto dim = arch.unarchive_into( cufx::MemoryView1D<unsigned char>(vec.data(), vec.size()) );
			
			// for (int i = 0; i != 128; ++i) {
			// 	vm::print("{} ", int(vec[i]));
			// }

			// vm::println("{}", (void*)vec.data());
			// return 0;

			auto view_info = cufx::MemoryView2DInfo{}
						.set_stride( dim.width * sizeof( char ) )
						.set_width( dim.width )
						.set_height( dim.height );
			auto view = block.view_3d<unsigned char>( view_info, dim );
			cufx::memory_transfer( arr, view )
			  .launch_async( stream );
			auto idx = float3{ float( arch.index().x ),
							   float( arch.index().y ),
							   float( arch.index().z ) };
			auto box = Box3D{}
						 .set_min( idx / grid_dim_max * scale * 2.f - 1.f )
						 .set_max( ( idx + 1.f ) / grid_dim_max * scale * 2.f - 1.f );
			cout << box << endl;
			swap[ 1 - curr_swap ].wait().unwrap();
			bind_texture( block_arr[ curr_swap ] );
			auto opts = RenderOptions{}
						.set_camera( camera )
						.set_box( box )
						.set_inner_scale( inner_scale )
						.set_block_index( idx / grid_dim_max );
			kernel( launch_info, img_view, opts )
			  .launch_async( stream );
			curr_swap = 1 - curr_swap;
		}
		swap[ 0 ].wait().unwrap();
		swap[ 1 ].wait().unwrap();

		cout << "render finished" << endl;

		img_view.copy_from_device().launch();
		image.dump( out );
		cout << "written image " << out << endl;
		return 0;
	};

	switch ( auto nx = opts[ "s" ].as<std::size_t>() ) {
	case 1: return nx_msaa_impl( 1, Pixel<1>{}, render_kernel );
	case 2: return nx_msaa_impl( 2, Pixel<2>{}, render_kernel_2x );
	case 4: return nx_msaa_impl( 4, Pixel<4>{}, render_kernel_4x );
	case 8: return nx_msaa_impl( 8, Pixel<8>{}, render_kernel_8x );
	case 16: return nx_msaa_impl( 16, Pixel<16>{}, render_kernel_16x );
	default: cout << "invalid sample count: " << nx << endl;
	}
}
