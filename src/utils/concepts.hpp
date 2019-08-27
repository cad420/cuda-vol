#pragma once

#include <utility>

namespace vol
{
struct NoMove
{
	NoMove() = default;

	NoMove( NoMove const & ) = default;
	NoMove &operator=( NoMove const & ) = default;
	NoMove( NoMove && ) = delete;
	NoMove &operator=( NoMove && ) = delete;
};

struct NoCopy
{
	NoCopy() = default;

	NoCopy( NoCopy && ) = default;
	NoCopy &operator=( NoCopy && ) = default;
	NoCopy( NoCopy const & ) = delete;
	NoCopy &operator=( NoCopy const & ) = delete;
};

struct ExplicitCopy
{
	ExplicitCopy() = default;

	ExplicitCopy( ExplicitCopy && ) = default;
	ExplicitCopy &operator=( ExplicitCopy && ) = default;

protected:
	explicit ExplicitCopy( ExplicitCopy const & ) = default;

private:
	ExplicitCopy &operator=( ExplicitCopy const & ) = default;
};

struct NoHeap
{
private:
	static void *operator new( std::size_t );
	static void *operator new[]( std::size_t );
};

struct Dynamic
{
	virtual ~Dynamic() = default;
};

}  // namespace vol
