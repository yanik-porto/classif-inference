#include "model_loader.h"

class ModelLoaderTensorRt : public ModelLoader
{
public:
    ModelLoaderTensorRt();
    ~ModelLoaderTensorRt();

    /**
     * @brief load a model from a given path
     * 
     * @param modelPath path to the model to load
     */
    void Load(const std::string &modelPath) override;

    void Execute(const at::Tensor &input) override;
};