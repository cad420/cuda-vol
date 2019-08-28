file(GLOB_RECURSE PPMCONV_SOURCES "./*.cc")

add_executable(ppmconv ${PPMCONV_SOURCES})
target_link_libraries(ppmconv)
# target_include_directories(ppmconv PRIVATE "./")
