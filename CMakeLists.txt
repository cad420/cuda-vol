cmake_minimum_required(VERSION 2.7)
set(CMAKE_CXX_STANDARD 11)
project(vol)

option(VOL_BUILD_RENDERER "set on to build renderer" ON)
option(VOL_BUILD_TOOLS "set on to build tools" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")

include_directories(
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/src"
)

if (VOL_BUILD_RENDERER)

  find_package(CUDA REQUIRED)
  set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
  set(CUDA_SEPARABLE_COMPILATION ON)
  set(CUDA_NVCC_FLAGS ${CMAKE_NVCC_FLAGS}
	-std=c++11
	#  -v  
	--expt-extended-lambda
	--use_fast_math
	--keep-device-functions
	-keep
	-maxrregcount
	32
	# -w
	-lineinfo
	-Xcudafe --display_error_number
  )

  file(GLOB_RECURSE SOURCES
	"${CMAKE_SOURCE_DIR}/src/*.cc"
	"${CMAKE_SOURCE_DIR}/src/*.cu"
  )

  cuda_add_executable(vol
	${SOURCES}
  )

  if (CMAKE_CXX_COMPILER MATCHES MSVC)

  else()
    target_link_libraries(vol pthread)
  endif()

endif()

if (VOL_BUILD_TOOLS)

  add_subdirectory(tools)

endif()
