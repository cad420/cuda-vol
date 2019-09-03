#pragma once

#include <iostream>

#define VOL_DEFINE_VECTOR1_FMT(T, x)                                \
    inline std::ostream &operator <<(std::ostream &os, T const &_)  \
    {                                                               \
        return os << "(" << _.x << ")";                             \
    }

#define VOL_DEFINE_VECTOR2_FMT(T, x, y)                             \
    inline std::ostream &operator <<(std::ostream &os, T const &_)  \
    {                                                               \
        return os << "(" << _.x << ", " << _.y << ")";              \
    }

#define VOL_DEFINE_VECTOR3_FMT(T, x, y, z)                          \
    inline std::ostream &operator <<(std::ostream &os, T const &_)  \
    {                                                               \
        return os << "(" << _.x << ", " << _.y << ", "              \
                  << _.z << ")";                                    \
    }

#define VOL_DEFINE_VECTOR34_FMT(T, x, y, z, w)                      \
    inline std::ostream &operator <<(std::ostream &os, T const &_)  \
    {                                                               \
        return os << "(" << _.x << ", " << _.y << ", "              \
                  << _.z << ", " << _.w << ")";                     \
    }
