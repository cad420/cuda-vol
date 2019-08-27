#pragma once

#include <future>
#include <cstdint>
#include <memory>
#include <iostream>
#include <functional>
#include <cuda_runtime.h>

#include "concepts.hpp"

namespace vol
{
namespace cuda
{
enum class Poll : uint32_t
{
	Pending = 0,
	Done = 1,
	Error = 2
};

inline std::ostream &operator<<( std::ostream &os, Poll stat )
{
	switch ( stat ) {
	case Poll::Pending: return os << "Pending";
	case Poll::Done: return os << "Done";
	case Poll::Error: return os << "Error";
	default: throw std::runtime_error( "invalid internal state: Poll" );
	}
}

inline Poll from_cuda_poll_result( cudaError_t ret )
{
	switch ( ret ) {
	case cudaSuccess:
		return Poll::Done;
	case cudaErrorNotReady:
		return Poll::Pending;
	default:
		return Poll::Error;
	}
}

struct Event
{
private:
	struct Inner : NoCopy, NoMove
	{
		~Inner() { cudaEventDestroy( _ ); }

		cudaEvent_t _;
	};

public:
	Event( bool enable_timing = false )
	{
		unsigned flags = cudaEventBlockingSync;
		if ( !enable_timing ) flags |= cudaEventDisableTiming;
		cudaEventCreateWithFlags( &_->_, flags );
	}

	void record() const { cudaEventRecord( _->_ ); }
	Poll poll() const { return from_cuda_poll_result( cudaEventQuery( _->_ ) ); }
	bool wait() const { return cudaEventSynchronize( _->_ ) == cudaSuccess; }

private:
	std::shared_ptr<Inner> _ = std::make_shared<Inner>();
};

struct Stream
{
private:
	struct Inner : NoCopy, NoMove
	{
		~Inner()
		{
			if ( _ != 0 ) cudaStreamDestroy( _ );
		}

		cudaStream_t _ = 0;
		std::mutex mtx;
	};

	Stream( std::nullptr_t ) {}

public:
	struct Lock : NoCopy, NoHeap
	{
		Lock( Inner &stream ) :
		  stream( stream ),
		  _( stream.mtx )
		{
		}

		cudaStream_t get() const { return stream._; }

	private:
		Inner &stream;
		std::unique_lock<std::mutex> _;
	};

public:
	Stream() { cudaStreamCreate( &_->_ ); }

	Poll poll() const { return from_cuda_poll_result( cudaStreamQuery( _->_ ) ); }
	bool wait() const { return cudaStreamSynchronize( _->_ ) == cudaSuccess; }
	Lock lock() const { return Lock( *_ ); }

public:
	static Stream null() { return Stream( nullptr ); }

private:
	std::shared_ptr<Inner> _ = std::make_shared<Inner>();
};

enum class Result : uint32_t
{
	Ok = 0,
	Err = 1,
};

inline std::ostream &operator<<( std::ostream &os, Result stat )
{
	switch ( stat ) {
	case Result::Ok: return os << "Ok";
	case Result::Err: return os << "Err";
	default: throw std::runtime_error( "invalid internal state: Result" );
	}
}

struct Task : NoCopy
{
	Task( std::function<void( cudaStream_t )> &&_ ) :
	  _( std::move( _ ) ) {}

	std::future<Result> launch_async( Stream const &stream = Stream() ) &&
	{
		Event event;
		{
			auto lock = stream.lock();
			_( lock.get() );
			event.record();
		}
		return std::async( std::launch::deferred, [=]() -> Result {
			if ( event.wait() ) {
				return Result::Ok;
			} else {
				return Result::Err;
			}
		} );
	}

	Result launch( Stream const &stream = Stream() ) &&
	{
		auto future = std::move( *this ).launch_async( stream );
		future.wait();
		return future.get();
	}

private:
	std::function<void( cudaStream_t )> _;
};

struct KernelLaunchInfo
{
	KernelLaunchInfo &setGridDim( dim3 const &_ )
	{
		grid_dim = _;
		return *this;
	}
	KernelLaunchInfo &setBlockDim( dim3 const &_ )
	{
		block_dim = _;
		return *this;
	}
	KernelLaunchInfo &setShmPerBlockBytes( std::size_t _ )
	{
		shm_per_block_bytes = _;
		return *this;
	}

public:
	dim3 grid_dim;
	dim3 block_dim;
	std::size_t shm_per_block_bytes = 0;
};

template <typename F>
struct Kernel;

template <typename Ret, typename... Args>
struct Kernel<Ret( Args... )>
{
private:
	using Launcher = void( KernelLaunchInfo const &, Args... args, cudaStream_t );

public:
	Kernel( Launcher *_ ) :
	  _( _ ) {}

	Task operator()( KernelLaunchInfo const &info, Args... args )
	{
		return Task( [=]( cudaStream_t stream ) { _( info, args..., stream ); } );
	}

private:
	Launcher *_;
};

template <typename F>
struct Functionlify;

template <typename Ret, typename... Args>
struct Functionlify<Ret( Args... )>
{
	using type = Ret( Args... );
};

template <typename Ret, typename... Args>
struct Functionlify<Ret ( * )( Args... )> : Functionlify<Ret( Args... )>
{
};

template <typename Ret, typename... Args>
struct Functionlify<Ret ( *const )( Args... )> : Functionlify<Ret( Args... )>
{
};

/* clang-format off */
#define VOL_DEFINE_CUDA_KERNEL( name, impl )                                       \
	namespace                                                                      \
	{                                                                              \
	template <typename F>                                                          \
	struct __Kernel_Impl_##impl;                                                   \
	template <typename Ret, typename... Args>                                      \
	struct __Kernel_Impl_##impl<Ret( Args... )>                                    \
	{                                                                              \
		static void launch( vol::cuda::KernelLaunchInfo const &info, Args... args, \
							cudaStream_t               stream )                    \
		{                                                                          \
			impl<<<info.grid_dim, info.block_dim, info.shm_per_block_bytes,        \
				   stream>>>( args... );                                           \
		}                                                                          \
	};                                                                             \
	}                                                                              \
	::vol::cuda::Kernel<                                                           \
	  typename ::vol::cuda::Functionlify<decltype( impl )>::type>                  \
	  name( __Kernel_Impl_##impl<                                                  \
			typename ::vol::cuda::Functionlify<decltype( impl )>::type>::launch )
/* clang-format on */

}  // namespace cuda

}  // namespace vol
