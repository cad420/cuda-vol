#include <iostream>
#include <cstdlib>
#include <string>
#include <cxxopts.hpp>

#include <compute_kernel.hpp>

using namespace std;
using namespace vol;

int main( int argc, char **argv )
{
	cxxopts::Options options( "vol", "Cuda based offline volume renderer" );
	options.add_options()( "i,input", "input lvd file name" )(
	  "h,help", "show this help message" )(
	  "o,output", "place the output image into <file>",
	  cxxopts::value<string>()->default_value( "a.png" ) );
	auto opts = options.parse( argc, argv );
	if ( opts.count( "h" ) ) {
		cout << options.help() << endl;
		exit( 0 );
	}
	auto out = opts[ "o" ].as<string>();

	cuda::Image<Pixel> image( 512, 512 );
	cuda::GlobalMemory mem( 512 * 512 * sizeof( Pixel ) );
	auto view =
	  image.view( cuda::Rect{}.setX1( 256 ).setY1( 400 ) ).with_global_memory( mem );

	auto launch_info = cuda::KernelLaunchInfo{}
						 .setGridDim( 1 )
						 .setBlockDim( 1 );
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
