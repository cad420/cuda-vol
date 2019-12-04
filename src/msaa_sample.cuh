#pragma once

#include <VMUtils/modules.hpp>

VM_BEGIN_MODULE(vol)

#define MS_SAMPLE( x, y )                                \
	{                                                    \
		float( x / 10.f / .8f ), float( y / 10.f / .8f ) \
	}

template <std::size_t N>
__device__ float2 msaa_sample( int idx );

template <>
__device__ float2 msaa_sample<1>( int )
{
	return float2{ 0, 0 };
}

template <>
__device__ float2 msaa_sample<2>( int i )
{
	static float2 _[] = { MS_SAMPLE( 4, 4 ),
						  MS_SAMPLE( -4, -4 ) };
	return _[ i ];
}

template <>
__device__ float2 msaa_sample<4>( int i )
{
	static float2 _[] = { MS_SAMPLE( -2, -6 ),
						  MS_SAMPLE( 6, -2 ),
						  MS_SAMPLE( -6, 2 ),
						  MS_SAMPLE( 2, 6 ) };
	return _[ i ];
}

template <>
__device__ float2 msaa_sample<8>( int i )
{
	static float2 _[] = { MS_SAMPLE( 1, -3 ),
						  MS_SAMPLE( -1, 3 ),
						  MS_SAMPLE( 5, 1 ),
						  MS_SAMPLE( -3, -5 ),
						  MS_SAMPLE( -5, 5 ),
						  MS_SAMPLE( -7, -1 ),
						  MS_SAMPLE( 3, 7 ),
						  MS_SAMPLE( 7, -7 ) };
	return _[ i ];
}

template <>
__device__ float2 msaa_sample<16>( int i )
{
	static float2 _[] = { MS_SAMPLE( 1, 1 ),
						  MS_SAMPLE( -1, -3 ),
						  MS_SAMPLE( -3, 2 ),
						  MS_SAMPLE( 4, -1 ),
						  MS_SAMPLE( -5, -2 ),
						  MS_SAMPLE( 2, 5 ),
						  MS_SAMPLE( 5, 3 ),
						  MS_SAMPLE( 3, -5 ),
						  MS_SAMPLE( -2, 6 ),
						  MS_SAMPLE( 0, -7 ),
						  MS_SAMPLE( -4, -6 ),
						  MS_SAMPLE( -6, 4 ),
						  MS_SAMPLE( -8, 0 ),
						  MS_SAMPLE( 7, -4 ),
						  MS_SAMPLE( 6, 7 ),
						  MS_SAMPLE( -7, -8 ) };
	return _[ i ];
}

VM_END_MODULE()
