#pragma once

#include <iostream>

namespace vol
{

struct Image
{
    Image &dump(std::ostream &os)
    {
        return *this;
    }
};

} // namespace vol