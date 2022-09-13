set(CUDA_HOME "/home/yannick/repos/8/lib/UNIX/x64/9/cuda/cuda-11.1")
set(CUDA_TOOLKIT_ROOT_DIR "/home/yannick/repos/8/lib/UNIX/x64/9/cuda/cuda-11.1")
set(CMAKE_CUDA_COMPILER "/home/yannick/repos/8/lib/UNIX/x64/9/cuda/cuda-11.1/bin/nvcc")

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops LANGUAGES CXX CUDA)

set(OpenVINO_DIR "/home/yannick/intel/openvino_2022/runtime/cmake")

# find_package(Torch REQUIRED)
find_package(OpenCV 4.2.0 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(CUDA)
find_package(OpenVINO REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

include_directories(    ${OpenCV_INCLUDE_DIRS}
                        ${CUDA_INCLUDE_DIRS}
                        "/home/yannick/repos/8/lib/UNIX/x64/9/tensorrt/7.2.3.4_cuda-11.1/include"
                        "/home/yannick/repos/8/lib/UNIX/x64/9/boost/include"
                        "/home/yannick/intel/openvino_2022/runtime/include"
                        )

#    "/home/prynel/Documents/Yannick/repos/8/lib/UNIX/x64/7/boost_161/include"
                    
link_directories(${CMAKE_SOURCE_DIR}/lib)
link_directories(${CUDA_LIBRARY_DIRS})
link_directories("/home/yannick/repos/8/lib/UNIX/x64/9/tensorrt/7.2.3.4_cuda-11.1/lib")
link_directories("/home/yannick/repos/8/lib/UNIX/x64/9/boost/lib/debug")

add_executable( example-app
                src/example-app.cpp
                src/model_loaders/model_loader_tensorrt.cpp
                src/model_loaders/model_loader_openvino.cpp
                src/model_loaders/model_loader.cpp)# src/example-app.cpp src/model_loaders/model_loader_torch_script.cpp

target_link_libraries(example-app "${OpenCV_LIBS}" "${CUDA_LIBRARIES}" "${CUDA_LIBRARY}" nvinfer nvonnxparser nvparsers openvino::runtime boost_filesystem boost_system) # "${TORCH_LIBRARIES}"