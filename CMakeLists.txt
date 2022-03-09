cmake_minimum_required(VERSION 3.10)
project(PlateRecognize)

set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd")
set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd")
set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT")

set(OpenCV_INCLUDE_DIRS D:/work/thirdparty/compiled/opencv452_s/include)

include_directories(D:/work/thirdparty/compiled/onnxruntime-win-x64-gpu-1.10.0/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

link_directories(D:/work/thirdparty/compiled/opencv452_s/lib)
link_directories(D:/work/thirdparty/compiled/onnxruntime-win-x64-gpu-1.10.0/lib)

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

file(GLOB_RECURSE library_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} src/*.cpp src/*.cc src/*.c)


add_library(plate_recognize SHARED ${library_src})

target_link_libraries(plate_recognize
    onnxruntime
    ${OpenCV_LIBS}
)

# add_library(plate_recognize_static STATIC ${library_src})
# SET_TARGET_PROPERTIES(plate_recognize_static PROPERTIES OUTPUT_NAME "plate_recognize")
# target_link_libraries(plate_recognize_static
#     onnxruntime
#     ${OpenCV_LIBS}
# )


file(GLOB_RECURSE sample_src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} tools/*.cpp)
add_executable(plate_demo ${sample_src})
target_link_libraries(plate_demo 
    plate_recognize
    onnxruntime
    ${OpenCV_LIBS}
)