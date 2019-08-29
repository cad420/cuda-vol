#pragma once

#include <cuda_runtime.h>

#include <utils/attribute.hpp>

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

}

}
