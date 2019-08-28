#pragma once

#include "stream.hpp"

namespace vol
{
namespace cuda
{
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

// #define VOL_CUDA_DEVICE __device__
// #define VOL_CUDA_DEVICE __device__

}  // namespace cuda

}  // namespace vol
