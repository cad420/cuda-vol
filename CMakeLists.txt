cmake_minimum_required(VERSION 2.7)
set(CMAKE_CXX_STANDARD 11)
project(vol)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")

include_directories(
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_SOURCE_DIR}/src"
)

file(GLOB_RECURSE SOURCES
  "${CMAKE_SOURCE_DIR}/src/*.cc"
  "${CMAKE_SOURCE_DIR}/src/*.cu"
)

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

cuda_add_executable(vol
  ${SOURCES}
)

if ("${CMAKE_CXX_COMPILER}" MATCHES "MSVC")
  set(LINK_LIBRARIES)
else()
  set(LINK_LIBRARIES pthread)
endif()

target_link_libraries(vol ${LINK_LIBRARIES})
