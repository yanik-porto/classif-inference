#include "model_loader_torch_script.h"

#include <torch/script.h> // One-stop header.
#include <chrono>

ModelLoaderTorchScript::ModelLoaderTorchScript() :
    ModelLoader()
{
}

ModelLoaderTorchScript::~ModelLoaderTorchScript()
{
    delete _module;
}

void ModelLoaderTorchScript::Load(const std::string &modelPath)
{
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        _module = new torch::jit::script::Module();
        *_module = torch::jit::load(modelPath);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return;
    }
}

void ModelLoaderTorchScript::Execute(const at::Tensor &input)
{
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input.cuda());

    auto start = std::chrono::system_clock::now();

    // Execute the model and turn its output into a tensor.
    at::Tensor output = _module->forward(inputs).toTensor();

    std::cout << "inference time : " << 
        std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now() - start).count()
        << " ms" << std::endl;

    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}