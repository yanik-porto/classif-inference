#include <string>
#include <torch/script.h>

class ModelLoader {
public:
    ModelLoader(){};
    virtual ~ModelLoader(){};

    /**
     * @brief load a model from a given path
     * 
     * @param modelPath path to the model to load
     */
    virtual void Load(const std::string &modelPath) = 0;

    virtual void Execute(const at::Tensor &input) = 0;
};