#include <iostream>
#include <cstdlib>
#include <string>
#include <cxxopts.hpp>

#include <compute_kernel.hpp>
#include <utils/volume.hpp>

using namespace std;
using namespace vol;

int round_up_div( int a, int b )
{
	return a % b == 0 ? a / b : a / b + 1;
}

int main( int argc, char **argv )
{
	cxxopts::Options options( "vol", "Cuda based offline volume renderer" );
	options.add_options()( "i,input", "input lvd file name", cxxopts::value<string>() )(
	  "h,help", "show this help message" )(
	  "o,output", "place the output image into <file>", cxxopts::value<string>()->default_value( "a.png" ) )(
	  "x", "block_dim.x", cxxopts::value<unsigned>() )(
	  "y", "block_dim.y", cxxopts::value<unsigned>() )(
	  "z", "block_dim.z", cxxopts::value<unsigned>() );

	auto opts = options.parse( argc, argv );
	if ( opts.count( "h" ) ) {
		cout << options.help() << endl;
		exit( 0 );
	}
	auto in = opts[ "i" ].as<string>();
	auto out = opts[ "o" ].as<string>();

	auto block_dim = cuda::Extent{}
					   .set_width( opts[ "x" ].as<unsigned>() )
					   .set_height( opts[ "y" ].as<unsigned>() )
					   .set_depth( opts[ "z" ].as<unsigned>() );
	auto volume = Volume<Voxel>::from_raw( in, block_dim );
	std::cout << "total blocks: " << volume.block_count() << std::endl;
	auto block = volume.get_block( 0 );

	auto block_arr = cuda::Array3D<Voxel>( block_dim );
	auto res = cuda::memory_transfer( block_arr, block.view() ).launch();
	cout << res << endl;
	if ( res == cuda::Result::Err ) {
		auto err = cudaGetLastError();
		std::cout << cudaGetErrorName( err ) << std::endl;
	}
	bind_texture( block_arr );

	cuda::Image<Pixel> image( 512, 512 );
	auto device_swap = image.create_device_buffer();
	// cuda::GlobalMemory mem( 512 * 512 * sizeof( Pixel ) );
	// auto view_info = cuda::MemoryView2DInfo{}
	// 				   .set_stride( 256 * sizeof( Pixel ) )
	// 				   .set_width( 256 )
	// 				   .set_height( 400 );
	auto view =
	  image.view( cuda::Rect{}
					.set_x1( 512 )
					.set_y1( 512 ) )
		.with_device_memory( device_swap.second );

	auto block_size = dim3( 16, 16 );
	auto launch_info = cuda::KernelLaunchInfo{}
						 .set_grid_dim( round_up_div( view.width(), block_size.x ),
										round_up_div( view.height(), block_size.y ) )
						 .set_block_dim( block_size );
	res = render_kernel( launch_info, view ).launch();
	cout << res << endl;
	if ( res == cuda::Result::Err ) {
		auto err = cudaGetLastError();
		std::cout << cudaGetErrorName( err ) << std::endl;
	}

	res = view.copy_from_device().launch();
	cout << res << endl;
	if ( res == cuda::Result::Err ) {
		auto err = cudaGetLastError();
		std::cout << cudaGetErrorName( err ) << std::endl;
	}

	image.dump( out );
}
