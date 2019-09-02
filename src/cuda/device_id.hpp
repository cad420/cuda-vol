#pragma once

#include <cuda_runtime.h>

#include <utils/concepts.hpp>

namespace vol
{
namespace cuda
{
struct DeviceId
{
private:
	struct Lock : NoCopy, NoHeap
	{
		Lock( int _ ) :
		  _( _ ) {}
		Lock( Lock &&_ ) :
		  _( _._ )
		{
			_._ = -1;
		}
		Lock &operator=( Lock && ) = delete;
		~Lock()
		{
			if ( _ >= 0 ) { cudaSetDevice( _ ); }
		}

	private:
		int _;
	};

public:
	int id() const { return _; }
	bool is_host() const { return _ < 0; }
	bool is_device() const { return _ >= 0; }
	Lock lock() const
	{
		int d = -1;
		if ( _ >= 0 ) {
			cudaGetDevice( &d );
			cudaSetDevice( _ );
		}
		return Lock( d );
	}

	explicit DeviceId( int _ = 0 ) :
	  _( _ ) {}

private:
	int _;
};

}  // namespace cuda

}  // namespace vol