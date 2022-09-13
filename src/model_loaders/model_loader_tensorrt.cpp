#include "model_loader_tensorrt.h"

#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cuda_runtime.h>

using namespace std::chrono;

uint32_t GetTensorRTVersion()
{
    return 1000 * NV_TENSORRT_MAJOR + 100 * NV_TENSORRT_MINOR + 10 * NV_TENSORRT_PATCH + NV_TENSORRT_BUILD;
}

struct ModelLoaderTensorRt::Inner {
    TRTUniquePtr<nvinfer1::ICudaEngine> engine;
    TRTUniquePtr<nvinfer1::IExecutionContext> context;
    int batchSize;
    int m_inputBindingIndex{}, m_outputBindingIndex{};
    float *m_pInputDataDevice{}, *m_pOutputDataDevice{};
    // Buffer d'entrée et de sortie
    float *m_pInputDataHost{}, *m_pOutputDataHost{};
    size_t m_numInput{}, m_numOutput{};
    nvinfer1::Dims m_inputDims{}, m_outputDims{};
    Inner();
};

ModelLoaderTensorRt::Inner::Inner(){
    engine = TRTUniquePtr<nvinfer1::ICudaEngine>{nullptr};
    context = TRTUniquePtr<nvinfer1::IExecutionContext>{nullptr};
    batchSize = 1;
    m_inputBindingIndex = -1;
    m_outputBindingIndex = -1;
    m_pInputDataDevice = nullptr;
    m_pOutputDataDevice = nullptr;
    m_pInputDataHost = nullptr;
    m_pOutputDataHost = nullptr;
    m_numInput = 0;
    m_numOutput = 0;
}

ModelLoaderTensorRt::ModelLoaderTensorRt() :
    ModelLoader(),
    _inner(new Inner())
{
    std::cout << "TensorRt version : " << GetTensorRTVersion() << std::endl;
}

ModelLoaderTensorRt::~ModelLoaderTensorRt()
{
    if (_inner->m_pInputDataDevice) {
        cudaFree(_inner->m_pInputDataDevice);
        _inner->m_pInputDataDevice = nullptr;
    }

    if (_inner->m_pOutputDataDevice) {
        cudaFree(_inner->m_pOutputDataDevice);
        _inner->m_pOutputDataDevice = nullptr;
    }
    if (_inner->m_pInputDataHost) {
        free(_inner->m_pInputDataHost);
        _inner->m_pInputDataHost = nullptr;
    }
    if (_inner->m_pOutputDataHost) {
        free(_inner->m_pOutputDataHost);
        _inner->m_pOutputDataHost = nullptr;
    }
}

void ModelLoaderTensorRt::parseOnnxModel(const std::string &modelPath)
{
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
    // parse ONNX
    if (!parser->parseFromFile(modelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }

    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(2ULL << 30);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    // we have only one image in batch
    builder->setMaxBatchSize(1);

    _inner->engine.reset(builder->buildEngineWithConfig(*network, *config));
    if (_inner->engine == nullptr)
    {
        std::cerr << "empty engine" << std::endl;
        return;
    }
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

void ModelLoaderTensorRt::Load(const std::string &modelPath)
{
    if (modelPath.substr(modelPath.find_last_of(".") + 1) == "onnx") {
        parseOnnxModel(modelPath);
    }
    else if (modelPath.substr(modelPath.find_last_of(".") + 1) == "engine") {
        parseEngineModel(modelPath);
    }

    std::vector<void *> buffers(_inner->engine->getNbBindings()); // buffers for input and ouput data
    for (size_t i = 0; i < _inner->engine->getNbBindings(); ++i)
    {
        auto bindingSize = getSizeByDim(_inner->engine->getBindingDimensions(i)) * _inner->batchSize * sizeof(float);
        cudaMalloc(&buffers[i], bindingSize);
        if (_inner->engine->bindingIsInput(i))
        {
            auto name = _inner->engine->getBindingName(i);
            std::cout << "input name : " << name << std::endl;
            _inner->m_inputDims = _inner->engine->getBindingDimensions(i);
            _inner->m_inputBindingIndex = i;
        }
        else
        {
            auto name = _inner->engine->getBindingName(i);
            std::cout << "output name : " << name << std::endl;
            _inner->m_outputDims = _inner->engine->getBindingDimensions(i);
            _inner->m_outputBindingIndex = i;
        }
    }
    if (_inner->m_inputDims.nbDims == 0 || _inner->m_outputDims.nbDims == 0)
    {
        std::cerr << "Expect at least one input and one output for networkn";
        return;
    }

    std::cout << "_inner->m_inputDims : ";
    for (size_t i = 0; i < _inner->m_inputDims.nbDims; i++) {
        std::cout << _inner->m_inputDims.d[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "_inner->m_outputDims : ";
    for (size_t i = 0; i < _inner->m_outputDims.nbDims; i++) {
        std::cout << _inner->m_outputDims.d[i] << " ";
    }
    std::cout << std::endl;

    _inner->m_numInput = getSizeByDim(this->_inner->m_inputDims);
    _inner->m_numOutput = getSizeByDim(this->_inner->m_outputDims);

    std::cout << "inputBufferSize : " << this->_inner->m_numInput * sizeof(float) << std::endl;
    std::cout << "outputBufferSize : " << this->_inner->m_numOutput * sizeof(float) << std::endl;
    _inner->m_pInputDataHost = (float *)malloc(this->_inner->m_numInput * sizeof(float)); // Buffer d'image d'entrée
    _inner->m_pOutputDataHost = (float *)malloc(this->_inner->m_numOutput * sizeof(float)); // Buffer de sortie
    cudaMalloc(&_inner->m_pInputDataDevice, _inner->m_numInput * sizeof(float));            // Allocation mémoire du buffer d'entrée
    cudaMalloc(&_inner->m_pOutputDataDevice, _inner->m_numOutput * sizeof(float));          // Allocation mémoire du buffer de sortie
}

void ModelLoaderTensorRt::PreprocessImage(const std::vector<std::string> &imgPaths, float *tensor, const nvinfer1::Dims &dims)
{
    if (dims.nbDims < 3)
    {
        std::cerr << "load model error input: dims.nbDims < 3 : dims.nbDims = " << dims.nbDims << std::endl;
        return;
    }
    const int inputN = dims.d[dims.nbDims - 4]; // Nombre d'images dans le batch
    const int inputH = dims.d[dims.nbDims - 2]; // Hauteur de l'image
    const int inputW = dims.d[dims.nbDims - 1]; // Largeur de l'image
    const int inputC = dims.d[dims.nbDims - 3]; // Nb cannaux de l'image

    if (inputN != imgPaths.size()) {
        std::cerr << "Error: number of input images does not match the batch size : " << inputN << " vs " << imgPaths.size() << std::endl;
        return;
    }

    std::vector<float> mean{0.485f, 0.456f, 0.406f};
    std::vector<float> std{0.229f, 0.224f, 0.225f};
    for (size_t iImg = 0; iImg < imgPaths.size(); iImg++) {
        auto frame = cv::imread(imgPaths[iImg]);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        // Resize image
        cv::Mat imgResized;

        cv::resize(frame, imgResized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
        std::vector<cv::Mat> img_vector;
        cv::split(imgResized, img_vector);

        // Chargement de l'image dans un buffer
        // Passage NWHC -> NCHW
        for (int n = 0; n < inputC; n++)
        {
            for (int i = 0; i < inputW * inputH; i++)
            {
                tensor[(iImg * inputC * inputW * inputH) + i + n * inputW * inputH] = ((img_vector.at(n).at<uchar>(i) / 255.f) - mean.at(n)) / std.at(n);
            }
        }
    }
}

std::string ModelLoaderTensorRt::Execute(const std::string &imgPath)
{
    PreprocessImage(std::vector<std::string>{imgPath}, _inner->m_pInputDataHost, _inner->m_inputDims);

    void *bindings[2];
    /* transfer to device */
    cudaMemcpy(_inner->m_pInputDataDevice, _inner->m_pInputDataHost, _inner->m_numInput * sizeof(float), cudaMemcpyHostToDevice);
    bindings[_inner->m_inputBindingIndex] = (void *)_inner->m_pInputDataDevice;
    bindings[_inner->m_outputBindingIndex] = (void *)_inner->m_pOutputDataDevice;
    /* execute engine */
    if (!_inner->context->execute(_inner->batchSize, bindings))
    {
        return "unknown";
    }
    cudaMemcpy(_inner->m_pOutputDataHost, _inner->m_pOutputDataDevice, _inner->m_numOutput * sizeof(float), cudaMemcpyDeviceToHost);

    float *tensor = _inner->m_pOutputDataHost;

    // Recherche de l'indice correspondant à la classe prédicte : Recherche de la valeur de sortie maximale
    size_t numel = getSizeByDim(_inner->m_outputDims);
    std::vector<size_t> indices(numel);
    for (int i = 0; i < static_cast<int>(numel); i++)
    {
        indices[i] = i;
    }
    sort(indices.begin(), indices.begin() + numel, [tensor](size_t idx1, size_t idx2)
         { return tensor[idx1] > tensor[idx2]; });

    std::cout << "class " << indices[0] << " with score " << tensor[indices[0]] << std::endl;
    return _classes[indices[0]];
}

std::vector<std::string> ModelLoaderTensorRt::Execute(const std::vector<std::string> &imgPaths)
{
    std::vector<std::string> out(imgPaths.size(), "unknown");

    PreprocessImage(imgPaths, _inner->m_pInputDataHost, _inner->m_inputDims);

    auto start = high_resolution_clock::now();

    void *bindings[2];
    /* transfer to device */
    cudaMemcpy(_inner->m_pInputDataDevice, _inner->m_pInputDataHost, _inner->m_numInput * sizeof(float), cudaMemcpyHostToDevice);
    bindings[_inner->m_inputBindingIndex] = (void *)_inner->m_pInputDataDevice;
    bindings[_inner->m_outputBindingIndex] = (void *)_inner->m_pOutputDataDevice;
    /* execute engine */
    if (!_inner->context->execute(_inner->batchSize, bindings))
    {
        return out;
    }
    cudaMemcpy(_inner->m_pOutputDataHost, _inner->m_pOutputDataDevice, _inner->m_numOutput * sizeof(float), cudaMemcpyDeviceToHost);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "inference time : " << duration.count() / 1000. << " ms" << std::endl;

    float *tensor = _inner->m_pOutputDataHost;

    // Recherche de l'indice correspondant à la classe prédicte : Recherche de la valeur de sortie maximale
    int nImgPerBatch = _inner->m_outputDims.d[0];
    int nResPerImg = _inner->m_outputDims.d[1];

    for (size_t iB = 0; iB < nImgPerBatch; iB++) {
        if (iB >= imgPaths.size()) {
            break;
        }

        std::vector<size_t> indices(nResPerImg);
        std::vector<float> resForImg(nResPerImg);
        for (size_t i = 0; i < static_cast<int>(nResPerImg); i++)
        {
            indices[i] = i;
            resForImg[i] = tensor[i + iB * nResPerImg];
        }
        sort(indices.begin(), indices.begin() + nResPerImg, [resForImg](size_t idx1, size_t idx2)
            { return resForImg[idx1] > resForImg[idx2]; });

        std::cout << "class " << indices[0] << " with score " << resForImg[indices[0]] << std::endl;
        out[iB] = _classes[indices[0]];
    }

    return out;
}

std::string ModelLoaderTensorRt::PostprocessResults(float *gpuOutput, const nvinfer1::Dims &dims, int batchSize)
{
    // copy results from GPU to CPU
    std::vector<float> cpuOutput(getSizeByDim(dims) * batchSize);
    cudaMemcpy(cpuOutput.data(), gpuOutput, cpuOutput.size() * sizeof(float), cudaMemcpyDeviceToHost);


    std::cout << "CPU Output : ";
    for (auto const out : cpuOutput)
    {
        std::cout << out << " ";
    }
    std::cout << std::endl;

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

void ModelLoaderTensorRt::Logger::log(Severity severity, const char *msg)
{
    // remove this 'if' if you need more logged info
    if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR))
    {
        std::cout << msg << "n";
    }
}

size_t ModelLoaderTensorRt::getSizeByDim(const nvinfer1::Dims &dims)
{
    if (dims.nbDims == 0)
        return 0;

    size_t size = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        size *= static_cast<size_t>(dims.d[i]);
    }
    return size;
}