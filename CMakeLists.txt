cmake_minimum_required(VERSION 3.10)
project(PlateRecognize)

set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")

set(OpenCV_INCLUDE_DIRS D:/work/thirdparty/compiled/opencv452_s/include)
set(CUDA_INCLUDE "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\include")
set(CUDA_LIB "C:\\Program\ Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.0\\lib\\x64")

set(Torch_DIR D:/work/thirdparty/compiled/libtorch/share/cmake/Torch)
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch library found!")
    message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate Torch" \n)
endif()

include_directories(${CUDA_INCLUDE})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

link_directories(D:/work/thirdparty/compiled/opencv452_s/lib)
link_directories(${CUDA_LIB})
link_directories()


set(OpenCV_LIBS 
    opencv_world452
    ade
    libjpeg-turbo
    libopenjp2
    libpng
    libtiff
    libwebp
    zlib
    )

# set(Torch_LIBS
#     asmjit.lib
#     c10_cuda.lib
#     caffe2_detectron_ops_gpu.lib
#     caffe2_module_test_dynamic.lib
#     caffe2_nvrtc.lib
#     clog.lib
#     cpuinfo.lib
#     dnnl.lib
#     fbgemm.lib
#     gloo.lib
#     gloo_cuda.lib
#     libprotobuf.lib
#     libprotobuf-lite.lib
#     libprotoc.lib
#     mkldnn.lib
#     torch.lib
#     torch_cuda.lib
#     )

set(CUDA_LIBS
    cudart.lib
    cublas.lib
    cudnn.lib)

file(GLOB_RECURSE library_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cpp src/*.cc src/*.c)


add_library(plate_recognize SHARED ${library_src})

target_link_libraries(plate_recognize
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    ${CUDA_LIBS}
)

# add_library(plate_recognize_static STATIC ${library_src})
# SET_TARGET_PROPERTIES(plate_recognize_static PROPERTIES OUTPUT_NAME "plate_recognize")
# target_link_libraries(plate_recognize_static
#     onnxruntime
#     ${OpenCV_LIBS}
# )


file(GLOB_RECURSE sample_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} tools/*.cpp)
add_executable(plate_demo ${sample_src})