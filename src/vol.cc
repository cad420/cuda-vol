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

	auto launch_info = cuda::KernelLaunchInfo{}
						 .setGridDim( 1 )
						 .setBlockDim( 1 );
	auto res = compute_kernel.launch( launch_info );
	cout << res << endl;
}
