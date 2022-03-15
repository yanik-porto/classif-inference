#include "model_loader_tensorrt.h"
#include "definitionstrt.h"

#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime.h>

struct ModelLoaderTensorRt::Inner {
    TRTUniquePtr<nvinfer1::ICudaEngine> engine;
    TRTUniquePtr<nvinfer1::IExecutionContext> context;
    int batchSize;
    Inner();
};

ModelLoaderTensorRt::Inner::Inner(){
    engine = TRTUniquePtr<nvinfer1::ICudaEngine>{nullptr};
    context = TRTUniquePtr<nvinfer1::IExecutionContext>{nullptr};
    batchSize = 1;
}

ModelLoaderTensorRt::ModelLoaderTensorRt() :
    ModelLoader(),
    _inner(new Inner())
{
}

ModelLoaderTensorRt::~ModelLoaderTensorRt()
{
}

void ModelLoaderTensorRt::Load(const std::string &modelPath)
{
    if (modelPath.substr(modelPath.find_last_of(".") + 1) == "onnx") {
        parseOnnxModel(modelPath);
    }
    else if (modelPath.substr(modelPath.find_last_of(".") + 1) == "engine") {
        parseEngineModel(modelPath);
    }
}

std::string ModelLoaderTensorRt::Execute(const std::string &imgPath)
{

    std::vector< nvinfer1::Dims > inputDims; // we expect only one input
    std::vector< nvinfer1::Dims > outputDims; // and one output
    std::vector< void * > buffers(_inner->engine->getNbBindings()); // buffers for input and ouput data
    for (size_t i = 0; i < _inner->engine->getNbBindings(); ++i)
    {
        auto bindingSize = getSizeByDim(_inner->engine->getBindingDimensions(i)) * _inner->batchSize * sizeof(float);
        cudaMalloc(&buffers[i], bindingSize);
        if (_inner->engine->bindingIsInput(i))
        {
            auto name = _inner->engine->getBindingName(i);
            std::cout << "input name : " << name << std::endl;
            inputDims.emplace_back(_inner->engine->getBindingDimensions(i));
        }
        else
        {
            auto name = _inner->engine->getBindingName(i);
            std::cout << "output name : " << name << std::endl;
            outputDims.emplace_back(_inner->engine->getBindingDimensions(i));
        }
    }
    if (inputDims.empty() || outputDims.empty())
    {
        std::cerr << "Expect at least one input and one output for networkn";
        return "unkown";
    }

    // preprocess input data
    this->PreprocessImage(imgPath, (float *)buffers[0], inputDims[0]);

    //inference
    auto start = std::chrono::system_clock::now();
    _inner->context->enqueue(_inner->batchSize, buffers.data(), 0, nullptr);
    std::cout << "inference time : " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count()
              << " ms" << std::endl;

    // post-process results
    std::string classFound = this->PostprocessResults((float *)buffers[1], outputDims[0], _inner->batchSize);

    for (void* buf : buffers)
    {
        cudaFree(buf);
    }

    return classFound;
}

void ModelLoaderTensorRt::PreprocessImage(const std::string &imgPath, float *gpuInput, const nvinfer1::Dims &dims)
{
    cv::Mat frame = cv::imread(imgPath); // TODO : check if RGB conversion needed
    if (frame.empty())
    {
        std::cerr << "Input image " << imgPath << " load failed\n";
    }
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    cv::cuda::GpuMat gpuFrame;
    gpuFrame.upload(frame);

    auto inputWidth = dims.d[2];
    auto inputHeight = dims.d[1];
    auto channels = dims.d[0];
    auto inputSize = cv::Size(inputWidth, inputHeight);

    // resize
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpuFrame, resized, inputSize, 0, 0, cv::INTER_NEAREST);

    // normalize
    cv::cuda::GpuMat fltImg;
    resized.convertTo(fltImg, CV_32FC3, 1.f / 255.f, cv::INTER_NEAREST);
    // cv::cuda::subtract(fltImg, cv::Scalar(0.4085f, 0.4228f, 0.3828f), fltImg, cv::noArray(), -1);
    cv::cuda::subtract(fltImg, cv::Scalar(0.485f, 0.456f, 0.406f), fltImg, cv::noArray(), -1);
    // cv::cuda::divide(fltImg, cv::Scalar(0.3675f, 0.3731f, 0.3788f), fltImg, 1, -1);
    cv::cuda::divide(fltImg, cv::Scalar(0.229f, 0.224f, 0.225f), fltImg, 1, -1);

    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(inputSize, CV_32FC1, gpuInput + i * inputWidth * inputHeight));
    }
    cv::cuda::split(fltImg, chw);
}

std::string ModelLoaderTensorRt::PostprocessResults(float *gpuOutput, const nvinfer1::Dims &dims, int batchSize)
{
    // copy results from GPU to CPU
    std::vector<float> cpuOutput(getSizeByDim(dims) * batchSize);
    cudaMemcpy(cpuOutput.data(), gpuOutput, cpuOutput.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // calculate softmax
    std::transform(cpuOutput.begin(), cpuOutput.end(), cpuOutput.begin(), [](float val) {return std::exp(val);});
    auto sum = std::accumulate(cpuOutput.begin(), cpuOutput.end(), 0.0);
    // find top classes predicted by the model
    std::vector<int> indices(getSizeByDim(dims) * batchSize);
    // generate sequence
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&cpuOutput](int i1, int i2) {return cpuOutput[i1] > cpuOutput[i2];});
    // print results
    int i = 0;
    while (cpuOutput[indices[i]] / sum > 0.005)
    {
        if (_classes.size() > indices[i])
        {
            std::cout << "class: " << _classes[indices[i]] << " | ";
        }
        std::cout << "confidence: " << 100 * cpuOutput[indices[i]] / sum << "% | index:" << indices[i] << std::endl;
        ++i;
    }

    return _classes[indices[0]];
}

size_t ModelLoaderTensorRt::getSizeByDim(const nvinfer1::Dims &dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}

void ModelLoaderTensorRt::parseOnnxModel(const std::string &modelPath)
{
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetwork()};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    // parse ONNX
    if (!parser->parseFromFile(modelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }

    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // we have only one image in batch
    builder->setMaxBatchSize(1);

    _inner->engine.reset(builder->buildEngineWithConfig(*network, *config));
    _inner->context.reset(_inner->engine->createExecutionContext());
}

void ModelLoaderTensorRt::parseEngineModel(const std::string &modelPath)
{
    std::ifstream engineFile(modelPath, std::ios::binary);
    if (!engineFile)
    {
        std::cerr << "Error opening engine file: " << modelPath << std::endl;
        return;
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        std::cerr << "Error loading engine file: " << modelPath << std::endl;
        return;
    }

    TRTUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(gLogger)};

    // int DLACore = 0;
    // if (DLACore != -1)
    // {
    //     runtime->setDLACore(DLACore);
    // }

    _inner->engine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
    _inner->context.reset(_inner->engine->createExecutionContext());
}

void ModelLoaderTensorRt::Logger::log(Severity severity, const char *msg)
{
    // remove this 'if' if you need more logged info
    if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR))
    {
        std::cout << msg << "n";
    }
}