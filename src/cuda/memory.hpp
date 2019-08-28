#pragma once

#include "stream.hpp"
#include "misc.hpp"

#include <utils/attribute.hpp>

namespace vol
{
namespace cuda
{
struct GlobalMemory;

enum class MemoryLocation : int
{
	Host,
	Device
};

namespace _
{
template <typename T, std::size_t N>
struct MemoryViewND
{
	__host__ __device__ T *data() const { return reinterpret_cast<T *>( _.ptr ); }
	__host__ __device__ explicit operator bool() const { return _.ptr; }
	cudaPitchedPtr get() const { return _; }
	MemoryLocation location() const
	{
		return is_device ? MemoryLocation::Device : MemoryLocation::Host;
	}

protected:
	cudaPitchedPtr _ = { 0 };
	bool is_device = false;
	friend struct cuda::GlobalMemory;
};

}  // namespace _

template <typename T, std::size_t N>
struct MemoryViewND;

struct MemoryView2DInfo
{
	VOL_DEFINE_ATTRIBUTE( std::size_t, offset ) = 0;
	VOL_DEFINE_ATTRIBUTE( std::size_t, stride ) = 0;
	VOL_DEFINE_ATTRIBUTE( std::size_t, width ) = 0;
	VOL_DEFINE_ATTRIBUTE( std::size_t, height ) = 0;
};

template <typename T>
struct MemoryViewND<T, 1> : _::MemoryViewND<T, 1>
{
	__host__ __device__ T &at( std::size_t x ) const
	{
		return reinterpret_cast<T *>( this->_.ptr )[ x ];
	}
	__host__ __device__ std::size_t size() { return this->_.xsize; }

public:
	MemoryViewND() = default;
	MemoryViewND( char *ptr, std::size_t len )
	{
		this->_ = make_cudaPitchedPtr( ptr, 0, len, 0 );
	}
};

template <typename T>
struct MemoryViewND<T, 2> : _::MemoryViewND<T, 2>
{
	__host__ __device__ T &at( std::size_t x, std::size_t y ) const
	{
		auto ptr = reinterpret_cast<char *>( this->_.ptr );
		auto line = reinterpret_cast<T *>( ptr + y * this->_.pitch );
		return line[ x ];
	}
	__host__ __device__ std::size_t width() const { return this->_.xsize; }
	__host__ __device__ std::size_t height() const { return this->_.ysize; }

public:
	MemoryViewND() = default;
	MemoryViewND( char *ptr, MemoryView2DInfo const &info )
	{
		this->_ = make_cudaPitchedPtr( ptr + info.offset, info.stride,
									   info.width, info.height );
	}
};

template <typename T>
struct MemoryViewND<T, 3> : _::MemoryViewND<T, 3>
{
	// __host__ __device__ T &at( std::size_t x, std::size_t y ) const
	// {
	// 	auto ptr = reinterpret_cast<char *>( this->_.ptr );
	// 	auto line = reinterpret_cast<T *>( ptr + y * this->_.pitch );
	// 	return line[ x ];
	// }
	__host__ __device__ cudaExtent extent() const { return dim.get(); }

public:
	MemoryViewND() = default;
	MemoryViewND( char *ptr, MemoryView2DInfo const &info, cuda::Extent dim ) :
	  dim( dim )
	{
		this->_ = make_cudaPitchedPtr( ptr + info.offset, info.stride,
									   info.width, info.height );
	}

private:
	cuda::Extent dim;
};

struct GlobalMemory
{
private:
	struct Inner : NoCopy, NoMove
	{
		~Inner() { cudaFree( _ ); }

		char *_;
		std::size_t size;
	};

public:
	GlobalMemory( std::size_t size ) { cudaMalloc( &_->_, _->size = size ); }

	std::size_t size() const { return _->size; }

public:
	template <typename T>
	MemoryViewND<T, 2> view_2d( MemoryView2DInfo const &info ) const
	{
		auto mem = MemoryViewND<T, 2>( _->_, info );
		static_cast<_::MemoryViewND<T, 2> &>( mem ).is_device = true;
		return mem;
	}

private:
	std::shared_ptr<Inner> _ = std::make_shared<Inner>();
};

template <typename T>
using MemoryView1D = MemoryViewND<T, 1>;
template <typename T>
using MemoryView2D = MemoryViewND<T, 2>;
template <typename T>
using MemoryView3D = MemoryViewND<T, 3>;

}  // namespace cuda

}  // namespace vol
