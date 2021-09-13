#include "model_loader_torch_script.h"

#include <torch/script.h> // One-stop header.
#include <chrono>

ModelLoaderTensorRt::ModelLoaderTensorRt() : ModelLoader()
{
}

ModelLoaderTensorRt::~ModelLoaderTensorRt()
{
}

void ModelLoaderTensorRt::Load(const std::string &/*modelPath*/)
{
    return;
}

void ModelLoaderTensorRt::Execute(const at::Tensor &/*input*/)
{
    return;
}