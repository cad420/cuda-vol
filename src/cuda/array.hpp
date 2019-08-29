#pragma once

#include "stream.hpp"
#include "misc.hpp"

namespace vol
{
namespace cuda
{
namespace _
{
template <typename E, std::size_t N>
struct Array
{
private:
	struct Inner
	{
		~Inner() { cudaFreeArray( _ ); }

		cudaArray_t _;
	};

public:
	cudaArray_t get() const { return _->_; }
#ifdef __CUDACC__
	template <cudaTextureReadMode Mode>
	void bind_to_texture( texture<E, N, Mode> const &tex ) const
	{
		auto desc = cudaCreateChannelDesc<E>();
		cudaBindTextureToArray( tex, _->_, desc );
	}
#endif

protected:
	std::shared_ptr<Inner> _ = std::make_shared<Inner>();
};

}  // namespace _

template <typename E, std::size_t N>
struct ArrayND;

template <typename E>
struct ArrayND<E, 1> : _::Array<E, 1>
{
	ArrayND( std::size_t len ) :
	  w( len )
	{
		auto desc = cudaCreateChannelDesc<E>();
		cudaMallocArray( &this->_->_, &desc, w, 1 );
	}

	std::size_t size() const { return w; }

private:
	std::size_t w;
};

template <typename E>
struct ArrayND<E, 2> : _::Array<E, 2>
{
	ArrayND( std::size_t w, std::size_t h ) :
	  w( w ),
	  h( h )
	{
		auto desc = cudaCreateChannelDesc<E>();
		cudaMallocArray( &this->_->_, &desc, w, h );
	}

	std::size_t width() const { return w; }
	std::size_t height() const { return h; }

private:
	std::size_t w, h;
};

template <typename E>
struct ArrayND<E, 3> : _::Array<E, 3>
{
	ArrayND( Extent const &extent ) :
	  dim( extent )
	{
		auto desc = cudaCreateChannelDesc<E>();
		cudaMalloc3DArray( &this->_->_, &desc, dim.get() );
	}

	Extent extent() const { return dim; }

private:
	Extent dim;
};

template <typename E>
using Array1D = ArrayND<E, 1>;
template <typename E>
using Array2D = ArrayND<E, 2>;
template <typename E>
using Array3D = ArrayND<E, 3>;

}  // namespace cuda

}  // namespace vol
