#pragma once

#include <iostream>
#include <stb/stb_image_write.h>

#include "memory.hpp"
#include "transfer.hpp"
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
	__host__ Pixel &at_host( uint x, uint y ) const { return host_mem.at( x, y ); }
	__device__ Pixel &at_device( uint x, uint y ) const { return device_mem.at( x, y ); }
	__host__ __device__ uint width() const { return host_mem.width(); }
	__host__ __device__ uint height() const { return host_mem.height(); }

public:
	ImageView with_device_memory( MemoryView2D<Pixel> const &memory ) const
	{
		auto _ = *this;
		if ( memory.location() != MemoryLocation::Device ) {
			throw std::runtime_error( "invalid device memory view" );
		}
		_.device_mem = memory;
		return _;
	}
	Task copy_from_device() const
	{
		return memory_transfer( host_mem, device_mem );
	}
	Task copy_to_device() const
	{
		return memory_transfer( device_mem, host_mem );
	}

private:
	ImageView( MemoryView2D<Pixel> const &mem ) :
	  host_mem( mem ) {}

private:
	MemoryView2D<Pixel> host_mem;
	MemoryView2D<Pixel> device_mem;
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
		auto ptr_region = reinterpret_cast<char *>( &at( region.x0, region.y0 ) );
		auto ptr_region_ln1 = reinterpret_cast<char *>( &at( region.x0, region.y0 + 1 ) );
		auto view = MemoryView2DInfo{}
					  .set_stride( ptr_region_ln1 - ptr_region )
					  .set_width( region.width() )
					  .set_height( region.height() );
		auto mem = MemoryView2D<Pixel>( ptr_region, view );
		return ImageView<Pixel>( mem );
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
