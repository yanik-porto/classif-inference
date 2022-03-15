cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops LANGUAGES CXX CUDA)

set(InferenceEngine_DIR "/opt/intel/deployment_tools/inference_engine/share")
set(ngraph_DIR "/opt/intel/deployment_tools/ngraph/cmake")


# find_package(Torch REQUIRED)
find_package(OpenCV 4.2.0 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(CUDA)
find_package(ngraph REQUIRED)
find_package(InferenceEngine REQUIRED)


set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

include_directories(    ${OpenCV_INCLUDE_DIRS}
                        ${CUDA_INCLUDE_DIRS}
                        "/home/prynel/Documents/Yannick/repos/8/lib/UNIX/x64/7/tensorrt/7.2.3.4_cuda-11.1/include"
                        "/home/prynel/Documents/Yannick/repos/8/lib/UNIX/x64/7/boost_161/include"
                        ${InferenceEngine_INCLUDE_DIRS}
                    )

link_directories(${CMAKE_SOURCE_DIR}/lib)
link_directories(${CUDA_LIBRARY_DIRS})
link_directories("/home/prynel/Documents/Yannick/repos/8/lib/UNIX/x64/7/tensorrt/7.2.3.4_cuda-11.1/lib")
link_directories("/home/prynel/Documents/Yannick/repos/8/lib/UNIX/x64/7/boost_161/lib")

add_executable( example-app
                src/example-app.cpp
                src/model_loaders/model_loader_tensorrt.cpp
                src/model_loaders/model_loader_openvino.cpp
                src/model_loaders/model_loader.cpp)
                #src/example-app.cpp src/model_loaders/model_loader_torch_script.cpp

target_link_libraries(example-app "${OpenCV_LIBS}" "${CUDA_LIBRARIES}" "${CUDA_LIBRARY}" nvinfer nvonnxparser nvparsers ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES} boost_filesystem boost_system) #"${TORCH_LIBRARIES}"