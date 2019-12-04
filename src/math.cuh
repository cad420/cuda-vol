#pragma once

#include <limits>
#include <cuda_runtime.h>
#include <nv/helper_math.h>

VM_BEGIN_MODULE( vol )

struct Box3D
{
	float3 min, max;

public:
	__host__ __device__ float3 center() const { return ( min + max ) / 2; }
};

struct Ray3D
{
	float3 o, d;

public:
	__host__ __device__ bool intersect( Box3D const &box, float &tnear, float &tfar )
	{
		float3 invr = float3{ 1., 1., 1. } / d;
		float3 tbot = invr * ( box.min - o );
		float3 ttop = invr * ( box.max - o );

		float3 tmin = fminf( ttop, tbot );
		float3 tmax = fmaxf( ttop, tbot );

		// if ( isinf( tmin.x ) ) tmin.x = -std::numeric_limits<float>::infinity();
		// if ( isinf( tmin.y ) ) tmin.y = -std::numeric_limits<float>::infinity();
		// if ( isinf( tmin.z ) ) tmin.z = -std::numeric_limits<float>::infinity();

		// if ( isinf( tmax.x ) ) tmax.x = std::numeric_limits<float>::infinity();
		// if ( isinf( tmax.y ) ) tmax.y = std::numeric_limits<float>::infinity();
		// if ( isinf( tmax.z ) ) tmax.z = std::numeric_limits<float>::infinity();

		tnear = fmaxf( fmaxf( tmin.x, tmin.y ), tmin.z );
		tfar = fminf( fminf( tmax.x, tmax.y ), tmax.z );

		return tfar > tnear;
	}
};

VM_END_MODULE()
