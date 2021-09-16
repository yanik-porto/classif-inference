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

    void Execute(const std::string &imgPath);

private:
    void PreprocessImage(const std::string &imgPath, float *gpuInput, const nvinfer1::Dims &dims);
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

    void parseOnnxModel(const std::string &modelPath,
                   TRTUniquePtr<nvinfer1::ICudaEngine> &engine,
                   TRTUniquePtr<nvinfer1::IExecutionContext> &context);

    size_t getSizeByDim(const nvinfer1::Dims &dims);

    struct Inner;
    Inner *_inner;
};

#endif /* MODEL_LOADER_TENSORRT_H */