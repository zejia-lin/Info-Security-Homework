cmake_minimum_required(VERSION 3.18.0 FATAL_ERROR)
project(LzjWatermrk LANGUAGES C CXX)

############################################################
# CUDA librarires
############################################################
find_package(CUDAToolkit REQUIRED)
if(NOT ${CUDAToolkit_FOUND})
    message(FATAL_ERROR "CUDA Toolkit not found")
endif()
message(STATUS "Found CUDA Toolkit(${CUDAToolkit_VERSION}) at ${CUDAToolkit_BIN_DIR}")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60)
  message(WARNING "CUDA Arch not defined")
endif()
message(STATUS "Using CUDA Arch ${CMAKE_CUDA_ARCHITECTURES}")

include_directories(${CUDAToolkit_INCLUDE_DIRS})
link_libraries(${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY})


############################################################
# Opencv libraries
############################################################
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${Opencv_LIBS})
message(STATUS "Found OpenCV(${OpenCV_VERSION}) at ${OpenCV_DIR}")


############################################################
# CMake build type
############################################################
enable_language(CUDA)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build with ${CMAKE_BUILD_TYPE}")


############################################################
# Targets
############################################################
add_executable(Embed test/embed.cu)
target_link_libraries(Embed ${OpenCV_LIBS})
target_compile_options(Embed PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --use_fast_math
                       -O3
                       >)

add_executable(Extract test/extract.cu)
target_link_libraries(Extract ${OpenCV_LIBS})
target_compile_options(Extract PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --use_fast_math
                       -O3
                       >)

add_executable(WmCLI src/cli.cu)
target_link_libraries(WmCLI ${OpenCV_LIBS})
target_compile_options(WmCLI PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --use_fast_math
                       -O3
                       >)


add_executable(Dvid demo/demo_video.cu)
target_link_libraries(Dvid ${OpenCV_LIBS})
target_compile_options(Dvid PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --use_fast_math
                       -O3
                       >)

