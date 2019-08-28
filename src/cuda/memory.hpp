#pragma once

#include "stream.hpp"

namespace vol
{
namespace cuda
{
struct GlobalMemory;

struct MemoryView
{
	template <typename E = char>
	__host__ __device__ E *data() const
	{
		return reinterpret_cast<E *>( _ );
	}
	__host__ __device__ explicit operator bool() const { return _; }
	Task copy_to( void *dst, std::size_t begin, std::size_t len,
				  cudaMemcpyKind kind = cudaMemcpyDeviceToHost ) const
	{
		return Task( [=]( cudaStream_t stream ) {
			cudaMemcpyAsync( dst, _ + begin, len, kind, stream );
		} );
	}
	Task copy_from( void *src, std::size_t begin, std::size_t len,
					cudaMemcpyKind kind = cudaMemcpyHostToDevice ) const
	{
		return Task( [=]( cudaStream_t stream ) {
			cudaMemcpyAsync( _ + begin, src, len, kind, stream );
		} );
	}

	static MemoryView null() { return MemoryView( nullptr, 0 ); }

private:
	MemoryView( char *_, std::size_t size ) :
	  _( _ ),
	  size( size ) {}

private:
	char *_;
	std::size_t size;
	friend struct GlobalMemory;
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
	MemoryView view() const { return MemoryView( _->_, _->size ); }

private:
	std::shared_ptr<Inner> _ = std::make_shared<Inner>();
};

}  // namespace cuda

}  // namespace vol
