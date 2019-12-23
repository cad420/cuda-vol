file(GLOB_RECURSE PPMCONV_SOURCES "./*.cc")

add_executable(ppmconv ${PPMCONV_SOURCES})
vm_target_dependency(ppmconv stbi PRIVATE)
