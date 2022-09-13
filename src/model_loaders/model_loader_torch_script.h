#ifndef MODEL_LOADER_TORCH_SCIPT_H
#define MODEL_LOADER_TORCH_SCIPT_H

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

    /**
     * @brief run inference on an image given its path
     *
     * @param imgPath path to the image
     * @return name of the class found
     */
    std::string Execute(const std::string &imgPath) override;

    /**
     * @brief run inference on a batch of images given their paths
     * @param imgPaths paths to the images
     * @return names of the classes found
     */
    std::vector<std::string> Execute(const std::vector<std::string> &imgPaths) override;

private:
    struct Inner;
    Inner *_inner;
};

#endif /* MODEL_LOADER_TORCH_SCRIPT_H */