#pragma once

#include <future>
#include <cstdint>
#include <memory>
#include <iostream>
#include <vector>
#include <functional>
#include <cuda_runtime.h>

#include <utils/concepts.hpp>

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
	Task() = default;
	Task( std::function<void( cudaStream_t )> &&_ ) :
	  _{ std::move( _ ) } {}

	std::future<Result> launch_async( Stream const &stream = Stream() ) &&
	{
		Event event;
		{
			auto lock = stream.lock();
			for ( auto &e : _ ) {
				e( lock.get() );
			}
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
	Task &chain( Task &&other )
	{
		for ( auto &e : other._ ) {
			_.emplace_back( std::move( e ) );
		}
		return *this;
	}

private:
	std::vector<std::function<void( cudaStream_t )>> _;
};

struct PendingTasks
{
	PendingTasks &add( std::future<Result> &&one )
	{
		_.emplace_back( std::move( one ) );
		return *this;
	}
	std::vector<Result> wait()
	{
		std::vector<Result> ret;
		for ( auto &e : _ ) {
			e.wait();
			ret.emplace_back( e.get() );
		}
		_.clear();
		return ret;
	}

private:
	std::vector<std::future<Result>> _;
};

}  // namespace cuda

}  // namespace vol
