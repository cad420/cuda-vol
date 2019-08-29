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

	vector<std::future<cuda::Result>> tasks;

	cuda::Image<Pixel> image( 8192, 8192 );
	auto device_swap = image.create_device_buffer();
	auto view = image.view().with_device_memory( device_swap.second );
	tasks.emplace_back(
	  view.copy_to_device().launch_async() );

	auto block_dim = cuda::Extent{}
					   .set_width( opts[ "x" ].as<unsigned>() )
					   .set_height( opts[ "y" ].as<unsigned>() )
					   .set_depth( opts[ "z" ].as<unsigned>() );
	auto volume = Volume<Voxel>::from_raw( in, block_dim );
	auto dim = volume.dim();
	std::cout << "volume grid dim: " << dim.x << "," << dim.y << "," << dim.z << std::endl;
	auto block = volume.get_block( uint3{ 0, 0, 0 } );

	auto block_arr = cuda::Array3D<Voxel>( block_dim );
	bind_texture( block_arr );
	tasks.emplace_back(
	  cuda::memory_transfer( block_arr, block.view() ).launch_async() );

	auto block_size = dim3( 32, 32 );
	auto launch_info = cuda::KernelLaunchInfo{}
						 .set_grid_dim( round_up_div( view.width(), block_size.x ),
										round_up_div( view.height(), block_size.y ) )
						 .set_block_dim( block_size );

	for ( auto &task : tasks ) task.wait();
	tasks.clear();

	auto res = render_kernel( launch_info, view ).launch();
	cout << res << endl;
	if ( res == cuda::Result::Err ) {
		auto err = cudaGetLastError();
		std::cout << cudaGetErrorName( err ) << std::endl;
	}

	view.copy_from_device().launch();
	image.dump( out );
	cout << "written image " << out << endl;
}
