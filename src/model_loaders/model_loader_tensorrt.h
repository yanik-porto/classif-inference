#ifndef MODEL_LOADER_TENSORRT_H
#define MODEL_LOADER_TENSORRT_H

#include "model_loader.h"

#include <memory>
#include "NvInfer.h"
#include "NvOnnxParser.h"

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

    /**
     * @brief run inference on an image given its path
     *
     * @param imgPath path to the image
     */
    void Execute(const std::string &imgPath);

private:
    /**
     * @brief read and preprocess an image
     *
     * @param imgPath path to the image
     * @param gpuInput buffer initialized with number of binding indices
     * @param dims input dimensions
     */
    void PreprocessImage(const std::string &imgPath, float *gpuInput, const nvinfer1::Dims &dims);

    /**
     * @brief post process the inference results
     *
     * @param gpuOutput buffer filled with inference results
     * @param dims output dimensions
     * @param batchSize batch size
     */
    void PostprocessResults(float *gpuOutput, const nvinfer1::Dims &dims, int batchSize);

    class Logger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char *msg) override;
    } gLogger;

    // destroy TensorRT objects if something goes wrong
    struct TRTDestroy
    {
        template <class T>
        void operator()(T *obj) const
        {
            if (obj)
            {
                obj->destroy();
            }
        }
    };
    template <class T>
    using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;

    void parseOnnxModel(const std::string &modelPath);
    void parseEngineModel(const std::string &modelPath);

    size_t getSizeByDim(const nvinfer1::Dims &dims);

    struct Inner;
    Inner *_inner;
};

#endif /* MODEL_LOADER_TENSORRT_H */