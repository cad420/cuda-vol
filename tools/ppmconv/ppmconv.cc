#include <iostream>
#include <string>
#include <cstdlib>
#include <cxxopts.hpp>
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

using namespace std;

enum class ImageType : int
{
	Unknown,
	Png,
	Bmp,
	Jpg
};

int main( int argc, char **argv )
{
	cxxopts::Options options( "ppmconv", "PPM image converter" );
	options.add_options()( "i,input", "input ppm file name",
						   cxxopts::value<string>() )(
	  "o,output", "output image file name",
	  cxxopts::value<string>()->default_value( "a.png" ) );
	( "h,help", "show this help message" );
	auto opts = options.parse( argc, argv );
	if ( opts.count( "h" ) ) {
		cout << options.help() << endl;
		exit( 0 );
	}
	auto in = opts[ "i" ].as<string>();
	auto out = opts[ "o" ].as<string>();

	int width, height, channels;
	auto image = stbi_load( in.c_str(),
							&width, &height,
							&channels, STBI_rgb );

	ImageType type = ImageType::Unknown;

	auto pos = out.find_last_of( '.' );
	if ( pos != out.npos ) {
		auto suf = out.substr( pos + 1 );
		if ( suf == "png" ) {
			type = ImageType::Png;
		}
		if ( suf == "jpg" || suf == "jpeg" ) {
			type = ImageType::Jpg;
		}
		if ( suf == "bmp" ) {
			type = ImageType::Bmp;
		}
	}

	switch ( type ) {
	case ImageType::Png: stbi_write_png( out.c_str(), width, height, channels, image, width * channels ); break;
	case ImageType::Bmp: stbi_write_bmp( out.c_str(), width, height, channels, image ); break;
	case ImageType::Jpg: stbi_write_jpg( out.c_str(), width, height, channels, image, width * channels ); break;
	default: {
		cout << "unknown image type" << std::endl;
		exit( 1 );
	}
	}
}
