#include "model_loader_openvino.h"
#include "classification_results.h"

#include <inference_engine.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>

struct ModelLoaderOpenVino::Inner
{
    InferenceEngine::InferRequest _infer_request;
    std::string _input_name;
    std::string _output_name;
    std::vector<char> weights;
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
    InferenceEngine::Core core;

    auto fileToBuffer = [](const std::string& filePath) {
        std::ifstream file(filePath, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            return std::vector<char>();
        }
        return buffer;
    };

    // find weight path from model path
    std::string weightPath = modelPath.substr(0, modelPath.find_last_of(".") + 1) + "bin";

    // read model and weight file
    std::vector<char> model = fileToBuffer(modelPath);
    _inner->weights = fileToBuffer(weightPath);

    // read network from buffers
    std::string strModel(model.begin(), model.end());
    InferenceEngine::CNNNetwork network = core.ReadNetwork(strModel, InferenceEngine::make_shared_blob<uint8_t>({InferenceEngine::Precision::U8, {_inner->weights.size()}, InferenceEngine::C}, (uint8_t *)_inner->weights.data()));

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

    InferenceEngine::ExecutableNetwork executable_network;
    executable_network = core.LoadNetwork(network, "CPU");
    _inner->_infer_request = executable_network.CreateInferRequest();
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

std::string ModelLoaderOpenVino::Execute(const std::string &imgPath)
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

    return "unknown"; // TODO : get best res
std::vector<std::string> ModelLoaderOpenVino::Execute(const std::vector<std::string> &imgPaths)
{
    // TODO : Multi image per batch
    std::vector<std::string> results;

    for (auto &img : imgPaths) {
        std::string res = Execute(img);
        results.push_back(res);
    }
    return results;
}