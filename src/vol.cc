#include <iostream>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cxxopts.hpp>

#include <cuda/device.hpp>
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
	  "d,dim", "output image dim", cxxopts::value<unsigned>()->default_value( "2048" ) )(
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
	auto img_size = opts[ "d" ].as<unsigned>();

	auto camera = Camera::Builder{}.build();
	cuda::PendingTasks tasks;

	auto devices = cuda::Device::scan();
	auto device = devices[ 0 ];

	cuda::Image<Pixel> image( img_size, img_size );
	auto device_swap = device.alloc_image_swap( image );
	auto view = image.view().with_device_memory( device_swap.second );
	tasks.add( view.copy_to_device().launch_async() );

	// auto block_dim = cuda::Extent{}
	// 				   .set_width( opts[ "x" ].as<unsigned>() )
	// 				   .set_height( opts[ "y" ].as<unsigned>() )
	// 				   .set_depth( opts[ "z" ].as<unsigned>() );
	auto volume = Volume<Voxel>::from_lvd( in );
	auto grid_dim = volume.dim();
	cout << "volume grid dim: " << grid_dim << endl;
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
	auto bump_pix = 0.f;
	auto bump = bump_pix / block_dim_f;
	auto inner_scale = 1.f - 2.f * ( padding_f + bump_pix ) /
							   ( block_dim_f + 2 * bump_pix );
	auto blocks = volume.get_blocks();
	stable_sort(
	  blocks.begin(), blocks.end(),
	  [&]( ArchievedVolumeBlock<Voxel> const &a, ArchievedVolumeBlock<Voxel> const &b ) {
		  auto x = float3{ float( a.index().x ),
						   float( a.index().y ),
						   float( a.index().z ) };
		  auto y = float3{ float( b.index().x ),
						   float( b.index().y ),
						   float( b.index().z ) };
		  return dot( x, camera.d ) < dot( y, camera.d );
	  } );

	auto kernel_block_dim = dim3( 32, 32 );
	auto launch_info = cuda::KernelLaunchInfo{}
						 .set_device( devices[ 0 ] )
						 .set_grid_dim( round_up_div( view.width(), kernel_block_dim.x ),
										round_up_div( view.height(), kernel_block_dim.y ) )
						 .set_block_dim( kernel_block_dim );
	float grid_dim_max = float( max( grid_dim.x, max( grid_dim.y, grid_dim.z ) ) );
	cuda::Array3D<Voxel> block_arr[ 2 ] = { device.alloc_arraynd<Voxel, 3>( block_dim ),
											device.alloc_arraynd<Voxel, 3>( block_dim ) };
	cuda::Stream swap[ 2 ];
	vector<shared_ptr<VolumeBlock<Voxel>>> block( 2 );
	int curr_swap = 0;
	tasks.wait();
	for ( auto &arch : blocks ) {
		cout << "rendering block: " << arch.index() << endl;
		block[ curr_swap ] = std::make_shared<VolumeBlock<Voxel>>( arch.unarchieve() );
		cuda::memory_transfer( block_arr[ curr_swap ], block[ curr_swap ]->view() )
		  .launch_async( swap[ curr_swap ] );
		auto idx = float3{ float( arch.index().x ),
						   float( arch.index().y ),
						   float( arch.index().z ) };
		auto box = Box3D{}
					 .set_min( ( idx - bump ) / grid_dim_max * 2.f - 1.f )
					 .set_max( ( idx + 1.f + bump ) / grid_dim_max * 2.f - 1.f );
		cout << box << endl;
		swap[ 1 - curr_swap ].wait().unwrap();
		bind_texture( block_arr[ curr_swap ] );
		render_kernel( launch_info, view, camera, box, inner_scale ).launch_async( swap[ curr_swap ] );
		curr_swap = 1 - curr_swap;
	}
	swap[ 0 ].wait().unwrap();
	swap[ 1 ].wait().unwrap();

	cout << "render finished" << endl;

	view.copy_from_device().launch();
	image.dump( out );
	cout << "written image " << out << endl;
}
