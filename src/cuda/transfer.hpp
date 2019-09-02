#pragma once

#include "stream.hpp"
#include "memory.hpp"
#include "array.hpp"

namespace vol
{
namespace cuda
{
namespace _
{
template <typename T, std::size_t N>
cudaMemcpyKind copy_type( MemoryViewND<T, N> const &dst, MemoryViewND<T, N> const &src )
{
	if ( dst.device_id().is_host() ) {
		if ( src.device_id().is_device() ) {
			return cudaMemcpyDeviceToHost;
		} else {
			return cudaMemcpyHostToHost;
		}
	} else {
		if ( src.device_id().is_device() ) {
			return cudaMemcpyDeviceToDevice;
		} else {
			return cudaMemcpyHostToDevice;
		}
	}
}

template <typename T, std::size_t N>
cudaMemcpyKind copy_type( MemoryViewND<T, N> const &src )
{
	return src.device_id().is_device() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
}

template <typename T, std::size_t N>
struct MemTrans;

template <typename T>
struct MemTrans<T, 1>
{
	static Task transfer( MemoryView1D<T> const &dst, MemoryView1D<T> const &src )
	{
		return Task( [=]( cudaStream_t _ ) {
			auto d = dst.get();
			auto s = src.get();
			cudaMemcpyAsync( d.ptr, s.ptr, d.xsize * sizeof( T ), copy_type( dst, src ), _ );
		} );
	}
};

template <typename T>
struct MemTrans<T, 2>
{
	static Task transfer( MemoryView2D<T> const &dst, MemoryView2D<T> const &src )
	{
		return Task( [=]( cudaStream_t _ ) {
			auto d = dst.get();
			auto s = src.get();
			cudaMemcpy2DAsync( d.ptr, d.pitch, s.ptr, s.pitch,
							   d.xsize * sizeof( T ), d.ysize, copy_type( dst, src ), _ );
		} );
	}
};

template <typename T>
struct MemTrans<T, 3>
{
	static Task transfer( MemoryView3D<T> const &dst, MemoryView3D<T> const &src )
	{
		return Task( [=]( cudaStream_t _ ) {
			auto d = dst.get();
			auto s = src.get();
			cudaMemcpy2DAsync( d.ptr, d.pitch, s.ptr, s.pitch,
							   d.xsize * sizeof( T ), d.ysize, copy_type( dst, src ), _ );
		} );
	}
};

template <typename T, std::size_t N>
struct ArrayTrans;

template <typename T>
struct ArrayTrans<T, 1>
{
	static Task transfer( Array1D<T> const &dst, MemoryView1D<T> const &src )
	{
		return Task( [=]( cudaStream_t _ ) {
			auto s = src.get();
			cudaMemcpyToArrayAsync( dst.get(), 0, 0, s.ptr,
									dst.size() * sizeof( T ), copy_type( src ), _ );
		} );
	}
};

template <typename T>
struct ArrayTrans<T, 2>
{
	static Task transfer( Array2D<T> const &dst, MemoryView2D<T> const &src )
	{
		return Task( [=]( cudaStream_t _ ) {
			auto s = src.get();
			cudaMemcpy2DToArrayAsync( dst.get(), 0, 0, s.ptr, s.pitch,
									  dst.width() * sizeof( T ), dst.height(), copy_type( src ), _ );
		} );
	}
};

template <typename T>
struct ArrayTrans<T, 3>
{
	static Task transfer( Array3D<T> const &dst, MemoryView3D<T> const &src )
	{
		return Task( [=]( cudaStream_t _ ) {
			cudaMemcpy3DParms params = { 0 };
			auto &srcPtr = params.srcPtr = src.get();
			params.dstArray = dst.get();
			params.extent = dst.extent().get();
			params.kind = copy_type( src );
			// using namespace std;
			// cout << srcPtr.pitch << " " << srcPtr.xsize << " " << srcPtr.ysize << endl;
			// cout << dst.extent().width << " " << dst.extent().height << " " << dst.extent().depth
			// 	 << endl;

			cudaMemcpy3DAsync( &params, _ );
		} );
	}
};

}  // namespace _

template <typename T, std::size_t N>
Task memory_transfer( MemoryViewND<T, N> const &dst, MemoryViewND<T, N> const &src )
{
	auto dst_lock = dst.device_id().lock();
	auto src_lock = src.device_id().lock();
	return _::MemTrans<T, N>::transfer( dst, src );
}

template <typename T, std::size_t N>
Task memory_transfer( ArrayND<T, N> const &dst, MemoryViewND<T, N> const &src )
{
	auto dst_lock = dst.device_id().lock();
	auto src_lock = src.device_id().lock();
	return _::ArrayTrans<T, N>::transfer( dst, src );
}

}  // namespace cuda

}  // namespace vol
