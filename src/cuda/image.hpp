#pragma once

#include <iostream>
#include <stb/stb_image_write.h>

#include "memory.hpp"
#include <utils/attribute.hpp>

namespace vol
{
namespace cuda
{
struct Rect
{
	VOL_DEFINE_ATTRIBUTE( uint, x0 ) = 0;
	VOL_DEFINE_ATTRIBUTE( uint, y0 ) = 0;
	VOL_DEFINE_ATTRIBUTE( uint, x1 ) = 0;
	VOL_DEFINE_ATTRIBUTE( uint, y1 ) = 0;

public:
	uint width() const { return x1 - x0; }
	uint height() const { return y1 - y0; }
};

template <typename Pixel>
struct Image;

template <typename Pixel>
struct ImageView final
{
	__host__ Pixel &at_host( uint x, uint y ) const
	{
		return host_ptr[ x + y * stride ];
	}
	__device__ Pixel &at_device( uint x, uint y ) const
	{
		return device_mem.data<Pixel>()[ x + y * w ];
	}
	__host__ __device__ uint width() const { return w; }
	__host__ __device__ uint height() const { return h; }

public:
	ImageView with_global_memory( GlobalMemory const &memory ) const
	{
		auto _ = *this;
		_.device_mem = memory.view();
		return _;
	}
	Task copy_from_device() const
	{
		Task task;
		auto device_ptr = device_mem.data<>();
		for ( uint i = 0; i < h; ++i ) {
			auto host_ptr_line = &at_host( 0, i );
			auto device_ptr_line = &at_device( 0, i );
			auto device_begin =
			  reinterpret_cast<char *>( device_ptr_line ) - device_ptr;
			task.chain(
			  device_mem.copy_to( host_ptr_line, device_begin, sizeof( Pixel ) * w ) );
		}
		return task;
	}
	Task copy_to_device() const
	{
		Task task;
		auto device_ptr = device_mem.data<>();
		for ( uint i = 0; i < h; ++i ) {
			auto host_ptr_line = &at_host( 0, i );
			auto device_ptr_line = &at_device( 0, i );
			auto device_begin =
			  reinterpret_cast<char *>( device_ptr_line ) - device_ptr;
			task.chain(
			  device_mem.copy_from( host_ptr_line, device_begin, sizeof( Pixel ) * w ) );
		}
		return task;
	}

private:
	ImageView( uint w, uint h, uint stride, Pixel *host_ptr ) :
	  w( w ),
	  h( h ),
	  stride( stride ),
	  host_ptr( host_ptr ) {}

private:
	uint w, h, stride;
	Pixel *host_ptr;
	MemoryView device_mem = MemoryView::null();
	friend struct Image<Pixel>;
};

template <typename Pixel>
struct Image final : NoCopy
{
	Image( uint width, uint height ) :
	  width( width ),
	  height( height ),
	  pixels( new Pixel[ width * height ] ) {}

	Image( Image &&_ ) :
	  width( _.width ),
	  height( _.height ),
	  pixels( _.pixels )
	{
		_.pixels = nullptr;
	}

	Image &operator=( Image &&_ )
	{
		if ( pixels ) delete pixels;
		width = _.width;
		height = _.height;
		pixels = _.pixels;
		_.pixels = nullptr;
		return *this;
	}

	~Image()
	{
		if ( pixels ) delete pixels;
	}

public:
	Pixel &at( uint x, uint y ) const { return pixels[ x + y * width ]; }

	ImageView<Pixel> view( Rect const &region ) const
	{
		auto ptr_region = &at( region.x0, region.y0 );
		return ImageView<Pixel>( region.width(), region.height(), width, ptr_region );
	}
	ImageView<Pixel> view() const
	{
		return view( Rect{}.set_x0( 0 ).set_y0( 0 ).set_x1( width ).set_y1( height ) );
	}

	Image &dump( std::string const &file_name )
	{
		std::string _;
		_.resize( width * height * sizeof( char ) * 4 );
		auto buffer = const_cast<char *>( _.data() );
		for ( int i = 0; i != height; ++i ) {
			auto line_ptr = buffer + ( width * 4 ) * i;
			for ( int j = 0; j != width; ++j ) {
				auto pixel_ptr = line_ptr + 4 * j;
				at( j, i ).write_to( pixel_ptr );
			}
		}
		stbi_write_png( file_name.c_str(), width, height, 4, buffer, width * 4 );
		return *this;
	}

private:
	uint width, height;
	Pixel *pixels;
};
}  // namespace cuda

}  // namespace vol
