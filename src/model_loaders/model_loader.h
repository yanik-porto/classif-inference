#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <string>

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

    virtual void Execute(const std::string &imgPath) = 0;

};

#endif /* MODEL_LOADER_H */