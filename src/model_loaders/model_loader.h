#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <string>
#include <vector>

class ModelLoader {
public:
    ModelLoader(){};
    virtual ~ModelLoader(){};

    /**
     * @brief load list of classes from file
     *
     * @param classesPath path to the file containing the list of classes
     */
    void LoadClasses(const std::string &classesPath);

    /**
     * @brief load a model from a given path
     *
     * @param modelPath path to the model to load
     */
    virtual void Load(const std::string &modelPath) = 0;

    /**
     * @brief run inference on an image given its path
     *
     * @param imgPath path to the image
     */
    virtual void Execute(const std::string &imgPath) = 0;

protected:
    std::vector<std::string> _classes;
};

#endif /* MODEL_LOADER_H */