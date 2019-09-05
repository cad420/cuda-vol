#pragma once

#include <future>
#include <cstdint>
#include <memory>
#include <iostream>
#include <vector>
#include <functional>
#include <chrono>
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

struct Result
{
	Result( cudaError_t _ = cudaSuccess ) :
	  _( _ ) {}

	bool ok() const { return _ == cudaSuccess; }
	bool err() const { return !ok(); }
	explicit operator bool() const { return ok(); }
	const char *message() const { return cudaGetErrorString( _ ); }
	void unwrap() const
	{
		if ( err() ) {
			std::cerr << "Result unwrap failed: " << message() << std::endl;
			std::abort();
		}
	}

private:
	cudaError_t _;
};

inline std::ostream &operator<<( std::ostream &os, Result stat )
{
	if ( stat.ok() ) {
		return os << "Ok";
	} else {
		return os << "Err: " << stat.message();
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
	Result wait() const { return cudaEventSynchronize( _->_ ); }

public:
	static std::chrono::microseconds elapsed( Event const &a, Event const &b )
	{
		float dt;
		cudaEventElapsedTime( &dt, a._->_, b._->_ );
		return std::chrono::microseconds( uint64_t( dt * 1000 ) );
	}

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
	Result wait() const { return cudaStreamSynchronize( _->_ ); }
	Lock lock() const { return Lock( *_ ); }

public:
	static Stream null() { return Stream( nullptr ); }

private:
	std::shared_ptr<Inner> _ = std::make_shared<Inner>();
};

struct Task : NoCopy
{
	Task() = default;
	Task( std::function<void( cudaStream_t )> &&_ ) :
	  _{ std::move( _ ) } {}

	std::future<Result> launch_async( Stream const &stream = Stream() ) &&
	{
		Event start, stop;
		{
			auto lock = stream.lock();
			start.record();
			for ( auto &e : _ ) e( lock.get() );
			stop.record();
		}
		return std::async( std::launch::deferred, [=] { return stop.wait(); } );
	}
	Result launch( Stream const &stream = Stream() ) &&
	{
		auto future = std::move( *this ).launch_async( stream );
		future.wait();
		return future.get();
	}
	Task &chain( Task &&other )
	{
		for ( auto &e : other._ ) _.emplace_back( std::move( e ) );
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
