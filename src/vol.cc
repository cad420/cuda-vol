#include <iostream>
#include <cstdlib>
#include <string>
#include <cxxopts.hpp>

#include <compute_kernel.hpp>
#include <utils/volume.hpp>

using namespace std;
using namespace vol;

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

	auto block_dim = VolumeBlockDim{}
					   .set_width( opts[ "x" ].as<unsigned>() )
					   .set_height( opts[ "y" ].as<unsigned>() )
					   .set_depth( opts[ "z" ].as<unsigned>() );
	auto volume = Volume<char>::from_raw( in, block_dim );

	std::cout << "total blocks: " << volume.block_count() << std::endl;
	auto block = volume.get_block( 0 );

	cuda::Image<Pixel> image( 512, 512 );
	cuda::GlobalMemory mem( 512 * 512 * sizeof( Pixel ) );
	auto view =
	  image.view( cuda::Rect{}.set_x1( 256 ).set_y1( 400 ) ).with_global_memory( mem );

	auto launch_info = cuda::KernelLaunchInfo{}
						 .set_grid_dim( 1 )
						 .set_block_dim( 1 );
	auto res = compute_kernel( launch_info, view ).launch();
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

	image.dump( "a.png" );
}
