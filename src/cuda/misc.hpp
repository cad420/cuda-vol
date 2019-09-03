#pragma once

#include <cuda_runtime.h>

#include <utils/attribute.hpp>
#include <utils/format.hpp>

namespace vol
{
namespace cuda
{
struct Extent
{
	VOL_DEFINE_ATTRIBUTE( std::size_t, width );
	VOL_DEFINE_ATTRIBUTE( std::size_t, height );
	VOL_DEFINE_ATTRIBUTE( std::size_t, depth );

public:
	std::size_t size() const { return width * height * depth; }
	cudaExtent get() const { return make_cudaExtent( width, height, depth ); }
};

}  // namespace cuda

#define VOL_CUDA_DEFINE_VECTOR1234_FMT( T )     \
    VOL_DEFINE_VECTOR1_FMT( T##1, x )           \
	VOL_DEFINE_VECTOR2_FMT( T##2, x, y )        \
	VOL_DEFINE_VECTOR3_FMT( T##3, x, y, z )     \
	VOL_DEFINE_VECTOR4_FMT( T##4, x, y, z, w )

VOL_CUDA_DEFINE_VECTOR1234_FMT( char )
VOL_CUDA_DEFINE_VECTOR1234_FMT( uchar )
VOL_CUDA_DEFINE_VECTOR1234_FMT( short )
VOL_CUDA_DEFINE_VECTOR1234_FMT( ushort )
VOL_CUDA_DEFINE_VECTOR1234_FMT( int )
VOL_CUDA_DEFINE_VECTOR1234_FMT( uint )
VOL_CUDA_DEFINE_VECTOR1234_FMT( long )
VOL_CUDA_DEFINE_VECTOR1234_FMT( ulong )
VOL_CUDA_DEFINE_VECTOR1234_FMT( longlong )
VOL_CUDA_DEFINE_VECTOR1234_FMT( ulonglong )
VOL_CUDA_DEFINE_VECTOR1234_FMT( float )
VOL_CUDA_DEFINE_VECTOR1234_FMT( double )
VOL_DEFINE_VECTOR3_FMT( dim3, x, y, z )

}  // namespace vol
