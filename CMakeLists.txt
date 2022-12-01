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
# Targets
############################################################

enable_language(CUDA)

add_executable(Embed test/embed.cu)
target_link_libraries(Embed ${OpenCV_LIBS})

add_executable(Helloworld test/helloworld.cu)
target_link_libraries(Helloworld ${OpenCV_LIBS})
