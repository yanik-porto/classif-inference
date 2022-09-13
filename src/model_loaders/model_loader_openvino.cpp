#include "model_loader_openvino.h"
#include "classification_results.h"

#include <inference_engine.hpp>
#include <openvino/openvino.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <iterator>

struct ModelLoaderOpenVino::Inner
{
    ov::InferRequest _infer_request;
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
    std::vector<char> modelBuffer = fileToBuffer(modelPath);
    std::vector<char> weights = fileToBuffer(weightPath);

    // read network from buffers
    std::string strModel(modelBuffer.begin(), modelBuffer.end());

    // fill weights tensor
    ov::Tensor tensor (ov::element::Type_t::u8, ov::Shape({weights.size()}));
    auto tensorData = tensor.data<uint8_t>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        tensorData[i] = weights[i];
    }

    // read model
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(strModel, tensor);

    // preprocessing info
    ov::preprocess::PrePostProcessor ppp(model);
    ov::preprocess::InputInfo &input = ppp.input(0);
    // layout and precision conversion is inserted automatically,
    // because tensor format != model input format
    input.tensor().set_layout("NHWC").set_element_type(ov::element::u8);
    input.model().set_layout("NCHW");

    // create infer request
    model = ppp.build();
    ov::CompiledModel compiledModel = core.compile_model(model, "CPU");
    _inner->_infer_request = compiledModel.create_infer_request();

    // Display input info
    ov::Tensor inputTensor = _inner->_infer_request.get_input_tensor(0);
    std::cout << "type : " << inputTensor.get_element_type().c_type_string() << std::endl;
    std::cout << "shape : ";
    for (auto &s : inputTensor.get_shape()) {
        std::cout << s << " ";
    }
    std::cout << std::endl;
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
    // read image file
    cv::Mat frame = cv::imread(imgPath);
    if (frame.empty()) {
        std::cerr << "Input image " << imgPath << " load failed\n";
    }
    cv::resize(frame, frame, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

    // Fill input
    ov::Tensor inputTensor = _inner->_infer_request.get_input_tensor(0);
    auto inputData = inputTensor.data<uint8_t>();
    for (size_t i = 0; i < inputTensor.get_size(); ++i) {
        inputData[i] = *(frame.data + i);
    }

    std::cout << "Execution de l'inference ... " << std::endl;
    _inner->_infer_request.infer();
    std::cout << "... OK !" << std::endl;

    // Collect output
    ov::Tensor outputTensor = _inner->_infer_request.get_output_tensor();
    auto outData = outputTensor.data<float>();

    // Display output
    for (size_t i = 0; i < outputTensor.get_size(); ++i) {
        std::cout << outData[i] << " ";
    }
    std::cout << std::endl;

    // Get best class
    int idx = std::distance(outData, std::max_element(outData, outData + outputTensor.get_size()));
    std::cout << "index : " << idx << std::endl;
    if (idx < _classes.size()) {
        return _classes[idx];
    }

    return "unknown";
}

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