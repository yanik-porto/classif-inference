#include "model_loader_torch_script.h"

#include <chrono>
#include <torch/script.h>

struct ModelLoaderTorchScript::Inner
{
    torch::jit::script::Module *_module;
};

ModelLoaderTorchScript::ModelLoaderTorchScript() :
    ModelLoader(),
    _inner(new Inner())
{
}

ModelLoaderTorchScript::~ModelLoaderTorchScript()
{
    delete _inner->_module;
}

void ModelLoaderTorchScript::Load(const std::string &modelPath)
{
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
       _inner->_module = new torch::jit::script::Module();
       *_inner->_module = torch::jit::load(modelPath);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return;
    }
}

void ModelLoaderTorchScript::Execute(const std::string &/*imgPath*/)
{
    at::Tensor input = torch::ones({1, 3, 224, 224});
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input.cuda());

    auto start = std::chrono::system_clock::now();

    // Execute the model and turn its output into a tensor.
    at::Tensor output = _inner->_module->forward(inputs).toTensor();

    std::cout << "inference time : " << 
        std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now() - start).count()
        << " ms" << std::endl;

    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
