#pragma once

#include <utility>

#define VOL_DEFINE_ATTRIBUTE( type, name )                    \
public:                                                       \
	template <typename... Args>                               \
	auto set_##name( Args &&... args )->decltype( ( *this ) ) \
	{                                                         \
		name = type( std::forward<Args>( args )... );         \
		return *this;                                         \
	}                                                         \
                                                              \
public:                                                       \
	type name

// // auto set_##name( type const &_ )->decltype( ( *this ) ) \
	// {                                                       \
	// 	name = _;                                           \
	// 	return *this;                                       \
	// }                                                       \
	// auto set_##name( type &&_ )->decltype( ( *this ) )      \
	// {                                                       \
	// 	name = std::move( _ );                              \
	// 	return *this;                                       \
	// }                                                       \
    //                                                         \

