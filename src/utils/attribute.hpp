#pragma once

#include <utility>

#define VOL_DEFINE_ATTRIBUTE( type, name )                  \
public:                                                     \
	auto set_##name( type const &_ )->decltype( ( *this ) ) \
	{                                                       \
		name = _;                                           \
		return *this;                                       \
	}                                                       \
	auto set_##name( type &&_ )->decltype( ( *this ) )      \
	{                                                       \
		name = std::move( _ );                              \
		return *this;                                       \
	}                                                       \
                                                            \
public:                                                     \
	type name
