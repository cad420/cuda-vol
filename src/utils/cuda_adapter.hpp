#pragma once

#include <future>
#include <cstdint>
#include <memory>
#include <iostream>
#include <cuda_runtime.h>

namespace vol
{
namespace cuda
{
enum class StreamStatus : uint32_t
{
	Pending = 0,
	Done = 1,
	Error = 2
};

inline std::ostream &operator<<( std::ostream &os, StreamStatus stat )
{
	switch ( stat ) {
	case StreamStatus::Pending: return os << "Pending";
	case StreamStatus::Done: return os << "Done";
	case StreamStatus::Error: return os << "Error";
	default: throw std::runtime_error( "invalid internal state: StreamStatus" );
	}
}

struct Stream
{
private:
	struct Inner
	{
		~Inner()
		{
			if ( _ != 0 ) cudaStreamDestroy( _ );
		}

		cudaStream_t _ = 0;
	};

	Stream( std::nullptr_t ) {}

public:
	Stream() { cudaStreamCreate( &_->_ ); }

	StreamStatus poll() const
	{
		switch ( cudaStreamQuery( _->_ ) ) {
		case cudaSuccess:
			return StreamStatus::Done;
		case cudaErrorNotReady:
			return StreamStatus::Pending;
		default:
			return StreamStatus::Error;
		}
	}

	bool wait() const { return cudaStreamSynchronize( _->_ ) == cudaSuccess; }

	cudaStream_t get() const { return _->_; }

public:
	static Stream null() { return Stream( nullptr ); }

private:
	std::shared_ptr<Inner> _ = std::make_shared<Inner>();
};

enum class ExecutionStatus : uint32_t
{
	Ok = 0,
	Err = 1,
};
inline std::ostream &operator<<( std::ostream &os, ExecutionStatus stat )
{
	switch ( stat ) {
	case ExecutionStatus::Ok: return os << "Ok";
	case ExecutionStatus::Err: return os << "Err";
	default: throw std::runtime_error( "invalid internal state: ExecutionStatus" );
	}
}

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
	using Launcher = void( KernelLaunchInfo const &, Args... args, Stream const & );

public:
	Kernel( Launcher *_ ) :
	  _( _ ) {}

	std::future<ExecutionStatus> launch_async( KernelLaunchInfo const &info,
											   Args &&... args,
											   Stream stream = Stream() )
	{
		_( info, args..., stream );
		return std::async( std::launch::deferred, [=]() -> ExecutionStatus {
			if ( stream.wait() ) {
				return ExecutionStatus::Ok;
			} else {
				return ExecutionStatus::Err;
			}
		} );
	}

	ExecutionStatus launch( KernelLaunchInfo const &info, Args &&... args )
	{
		auto stream = Stream::null();
		auto future = launch_async( info, std::forward<Args>( args )..., stream );
		future.wait();
		return future.get();
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
							::vol::cuda::Stream const &stream )                    \
		{                                                                          \
			impl<<<info.grid_dim, info.block_dim, info.shm_per_block_bytes,        \
				   stream.get()>>>( args... );                                     \
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
