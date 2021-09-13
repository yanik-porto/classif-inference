#include "model_loader.h"

class ModelLoaderTorchScript : public ModelLoader{
public:
    ModelLoaderTorchScript();
    ~ModelLoaderTorchScript();

    /**
     * @brief load a model from a given path
     * 
     * @param modelPath path to the model to load
     */
    void Load(const std::string &modelPath) override;

    void Execute(const at::Tensor &input) override;

private:
    torch::jit::script::Module *_module;
};