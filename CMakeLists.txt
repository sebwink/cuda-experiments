cmake_minimum_required(VERSION 3.19)

project(cuda LANGUAGES CXX CUDA VERSION 0.0.42)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CPM_DOWNLOAD_VERSION 0.32.0)

include(FindCUDAToolkit)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(get_cpm)
include(multiple_cppfiles_one_executable)
include(one_cppfile_one_executable)

CPMAddPackage(
  NAME CUDA_HELPER_HEADERS
  GIT_REPOSITORY https://github.com/NVIDIA/cuda-samples
  VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}
  DOWNLOAD_ONLY YES
)

set(CUDA_HELPER_INCLUDE_PATH ${CUDA_HELPER_HEADERS_SOURCE_DIR}/Common)
set(CUDA_COMMON_INCLUDES ${CUDA_HELPER_INCLUDE_PATH} ${CMAKE_SOURCE_DIR}/lib/include)

add_subdirectory(src)
