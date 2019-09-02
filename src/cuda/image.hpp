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
	VOL_DEFINE_ATTRIBUTE( std::size_t, x0 ) = 0;
	VOL_DEFINE_ATTRIBUTE( std::size_t, y0 ) = 0;
	VOL_DEFINE_ATTRIBUTE( std::size_t, x1 ) = 0;
	VOL_DEFINE_ATTRIBUTE( std::size_t, y1 ) = 0;

public:
	std::size_t width() const { return x1 - x0; }
	std::size_t height() const { return y1 - y0; }
};

template <typename Pixel>
struct Image;

template <typename Pixel = uchar4>
struct ImageView final
{
	__host__ Pixel &at_host( std::size_t x, std::size_t y ) const { return host_mem.at( x, y ); }
	__device__ Pixel &at_device( std::size_t x, std::size_t y ) const { return device_mem.at( x, y ); }
	__host__ __device__ std::size_t width() const { return host_mem.width(); }
	__host__ __device__ std::size_t height() const { return host_mem.height(); }

public:
	ImageView with_device_memory( MemoryView2D<Pixel> const &memory ) const
	{
		auto _ = *this;
		if ( memory.device_id().is_host() ) {
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

template <typename Pixel = uchar4>
struct Image final : NoCopy
{
	Image( std::size_t width, std::size_t height ) :
	  width( width ),
	  height( height ),
	  pixels( new Pixel[ width * height ]() ) {}  // init to zero

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
	Pixel &at( std::size_t x, std::size_t y ) const { return pixels[ x + y * width ]; }

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

	void dump( std::string const &file_name ) const
	{
		std::string _;
		_.resize( width * height * sizeof( char ) * 4 );
		auto buffer = const_cast<char *>( _.data() );
		for ( int i = 0; i != height; ++i ) {
			auto line_ptr = buffer + ( width * 4 ) * i;
			for ( int j = 0; j != width; ++j ) {
				auto pixel_ptr = line_ptr + 4 * j;
				at( j, i ).write_to( reinterpret_cast<unsigned char *>( pixel_ptr ) );
			}
		}
		stbi_write_png( file_name.c_str(), width, height, 4, buffer, width * 4 );
	}

private:
	std::size_t width, height;
	Pixel *pixels;
};

template <>
inline void Image<>::dump( std::string const &file_name ) const
{
	stbi_write_png( file_name.c_str(), width, height, 4,
					reinterpret_cast<unsigned char *>( pixels ), width * 4 );
}

}  // namespace cuda

}  // namespace vol
