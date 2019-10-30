#pragma once

#include <VMUtils/json_binding.hpp>

#define VOL_JSON_CUDA_VEC_IMPL1( Vec )                       \
	inline void to_json( nlohmann::json &j, const Vec &v )   \
	{                                                        \
		j = nlohmann::json{ v.x };                           \
	}                                                        \
	inline void from_json( const nlohmann::json &j, Vec &v ) \
	{                                                        \
		v.x = j[ 0 ].get<decltype( v.x )>();                 \
	}

#define VOL_JSON_CUDA_VEC_IMPL2( Vec )                       \
	inline void to_json( nlohmann::json &j, const Vec &v )   \
	{                                                        \
		j = nlohmann::json{ v.x, v.y };                      \
	}                                                        \
	inline void from_json( const nlohmann::json &j, Vec &v ) \
	{                                                        \
		v.x = j[ 0 ].get<decltype( v.x )>();                 \
		v.y = j[ 1 ].get<decltype( v.y )>();                 \
	}

#define VOL_JSON_CUDA_VEC_IMPL3( Vec )                       \
	inline void to_json( nlohmann::json &j, const Vec &v )   \
	{                                                        \
		j = nlohmann::json{ v.x, v.y, v.z };                 \
	}                                                        \
	inline void from_json( const nlohmann::json &j, Vec &v ) \
	{                                                        \
		v.x = j[ 0 ].get<decltype( v.x )>();                 \
		v.y = j[ 1 ].get<decltype( v.y )>();                 \
		v.z = j[ 2 ].get<decltype( v.z )>();                 \
	}

#define VOL_JSON_CUDA_VEC_IMPL4( Vec )                       \
	inline void to_json( nlohmann::json &j, const Vec &v )   \
	{                                                        \
		j = nlohmann::json{ v.x, v.y, v.z, v.w };            \
	}                                                        \
	inline void from_json( const nlohmann::json &j, Vec &v ) \
	{                                                        \
		v.x = j[ 0 ].get<decltype( v.x )>();                 \
		v.y = j[ 1 ].get<decltype( v.y )>();                 \
		v.z = j[ 2 ].get<decltype( v.z )>();                 \
		v.w = j[ 3 ].get<decltype( v.w )>();                 \
	}

#define VOL_JSON_CUDA_VEC_IMPLN( Vec ) \
	VOL_JSON_CUDA_VEC_IMPL1( Vec##1 )  \
	VOL_JSON_CUDA_VEC_IMPL2( Vec##2 )  \
	VOL_JSON_CUDA_VEC_IMPL3( Vec##3 )  \
	VOL_JSON_CUDA_VEC_IMPL4( Vec##4 )

VOL_JSON_CUDA_VEC_IMPLN( float )
VOL_JSON_CUDA_VEC_IMPLN( double )
VOL_JSON_CUDA_VEC_IMPLN( int )
VOL_JSON_CUDA_VEC_IMPLN( uint )
VOL_JSON_CUDA_VEC_IMPLN( long )
VOL_JSON_CUDA_VEC_IMPLN( ulong )
VOL_JSON_CUDA_VEC_IMPLN( longlong )
VOL_JSON_CUDA_VEC_IMPLN( ulonglong )
VOL_JSON_CUDA_VEC_IMPLN( short )
VOL_JSON_CUDA_VEC_IMPLN( ushort )
VOL_JSON_CUDA_VEC_IMPLN( char )
VOL_JSON_CUDA_VEC_IMPLN( uchar )

#undef VOL_JSON_CUDA_VEC_IMPLN
#undef VOL_JSON_CUDA_VEC_IMPL1
#undef VOL_JSON_CUDA_VEC_IMPL2
#undef VOL_JSON_CUDA_VEC_IMPL3
#undef VOL_JSON_CUDA_VEC_IMPL4
