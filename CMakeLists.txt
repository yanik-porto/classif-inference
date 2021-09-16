cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops LANGUAGES CXX CUDA)


# find_package(Torch REQUIRED)
find_package(OpenCV 4.2.0 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(CUDA)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

include_directories(    ${OpenCV_INCLUDE_DIRS}
                        ${CUDA_INCLUDE_DIRS}
                        "/home/prynel/Documents/Classification_yannick/InferenceRunner/externals/tensorrt/include"
                    )

link_directories(${CMAKE_SOURCE_DIR}/lib)
link_directories(${CUDA_LIBRARY_DIRS})
link_directories("/home/prynel/Documents/Classification_yannick/InferenceRunner/externals/tensorrt/lib")

add_executable( example-app
                src/example-app.cpp src/model_loaders/model_loader_tensorrt.cpp)
                #src/example-app.cpp src/model_loaders/model_loader_torch_script.cpp

target_link_libraries(example-app "${OpenCV_LIBS}" "${CUDA_LIBRARIES}" "${CUDA_LIBRARY}" nvinfer nvonnxparser nvparsers) #"${TORCH_LIBRARIES}"