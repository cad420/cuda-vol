cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 14)
project(vol)

option(VOL_BUILD_RENDERER "set on to build renderer" ON)
option(VOL_BUILD_TOOLS "set on to build tools" ON)

find_package(Git)
execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive)
# install external modules
include(scripts/external.cmake)
vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/cuda-fx
  GIT_TAG        master
)
cmake_policy(SET CMP0077 NEW)
set(VOCOMP_BUILD_NVCODEC ON)
set(VOCOMP_BUILD_REFINER OFF)
set(VOCOMP_BUILD_EXTRACTOR ON)
set(VOCOMP_BUILD_TOOLS OFF)
vm_external_module(
  GIT_REPOSITORY https://github.com/cad420/voxel-compression
  GIT_TAG        master
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")

include_directories(
  include
  src
)

if (VOL_BUILD_RENDERER)

  find_package(CUDA REQUIRED)
  # set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
  # set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_NVCC_FLAGS ${CMAKE_NVCC_FLAGS}
	-std=c++14
	#  -v  
	--expt-extended-lambda
	--use_fast_math
  --keep-device-functions
  --expt-relaxed-constexpr
	-keep
	-maxrregcount
	32
	# -w
	-lineinfo
	-Xcudafe --display_error_number
  )

  file(GLOB_RECURSE SOURCES
    src/*.cc
    src/*.cu
  )

  cuda_add_executable(vol
    ${SOURCES}
  )
  add_definitions(
	  -DKOI_NO_LIBUV_REACTOR
  )
  set_property(
    TARGET vol
    PROPERTY CUDA_SEPARABLE_COMPILATION ON
  )
  vm_target_dependency(vol cudafx PRIVATE)
  vm_target_dependency(vol voxel_extract PRIVATE)
  vm_target_dependency(vol video_decompress PRIVATE)
  target_link_libraries(vol ${CUDA_CUDA_LIBRARY})

endif()

if (VOL_BUILD_TOOLS)

  add_subdirectory(tools)

endif()
