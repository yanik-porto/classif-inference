#include "model_loader_openvino.h"
#include "classification_results.h"

#include <inference_engine.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

struct ModelLoaderOpenVino::Inner
{
    InferenceEngine::Core _core;
    InferenceEngine::CNNNetwork _network;
    InferenceEngine::ExecutableNetwork _executable_network;
    InferenceEngine::InferRequest _infer_request;
    std::string _input_name;
    std::string _output_name;
};

ModelLoaderOpenVino::ModelLoaderOpenVino() :
    _inner(new Inner())
{
}

ModelLoaderOpenVino::~ModelLoaderOpenVino()
{
}

void ModelLoaderOpenVino::Load(const std::string &modelPath)
{
    auto network = _inner->_core.ReadNetwork(modelPath);

    /** Take information about all topology inputs **/
    InferenceEngine::InputsDataMap input_info = network.getInputsInfo();
    _inner->_input_name = input_info.begin()->first;

    /** Iterate over all input info**/
    for (auto &item : input_info)
    {
        auto input_data = item.second;
        // Add your input configuration steps here
        // input_data->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
        input_data->setPrecision(InferenceEngine::Precision::U8);
        input_data->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);
    }

    /** Take information about all topology outputs **/
    InferenceEngine::OutputsDataMap output_info = network.getOutputsInfo();
    _inner->_output_name = output_info.begin()->first;
    /** Iterate over all output info**/
    for (auto &item : output_info)
    {
        auto output_data = item.second;
        // Add your output configuration steps here
        output_data->setPrecision(InferenceEngine::Precision::FP32);
    }

    _inner->_executable_network = _inner->_core.LoadNetwork(network, "CPU");
    _inner->_infer_request = _inner->_executable_network.CreateInferRequest();
}

InferenceEngine::Blob::Ptr wrapMat2Blob(const cv::Mat &mat)
{
    size_t channels = mat.channels();
    size_t height = mat.size().height;
    size_t width = mat.size().width;

    size_t strideH = mat.step.buf[0];
    size_t strideW = mat.step.buf[1];

    bool is_dense = strideW == channels && strideH == channels * width;

    if (!is_dense) {
        IE_THROW() << "Doesn't support conversion from not dense cv::Mat";
    }

    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::U8, {1, channels, height, width}, InferenceEngine::Layout::NHWC);

    return InferenceEngine::make_shared_blob<uint8_t>(tDesc, mat.data);
}

void ModelLoaderOpenVino::Execute(const std::string &imgPath)
{
    cv::Mat frame = cv::imread(imgPath);
    if (frame.empty())
    {
        std::cerr << "Input image " << imgPath << " load failed\n";
    }

    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    cv::resize(frame, frame, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);
    InferenceEngine::Blob::Ptr input = wrapMat2Blob(frame);

    _inner->_infer_request.SetBlob(_inner->_input_name, input);
    std::cout << "Execution de l'inference ... " << std::endl;
    _inner->_infer_request.Infer();
    std::cout << "... OK !" << std::endl;

    InferenceEngine::Blob::Ptr output = _inner->_infer_request.GetBlob(_inner->_output_name);

    // Print classification results
    ClassificationResult classificationResult(output, {imgPath}, 1, 5, _classes);
    classificationResult.print();
}