#ifndef MODEL_LADER_OPENVINO_H
#define MODEL_LADER_OPENVINO_H

#include "model_loader.h"

class ModelLoaderOpenVino : public ModelLoader
{
public:
    ModelLoaderOpenVino();
    ~ModelLoaderOpenVino();

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
     */
    void Execute(const std::string &imgPath);

private:
    struct Inner;
    Inner *_inner;
};
#endif // MODEL_LADER_OPENVINO_H