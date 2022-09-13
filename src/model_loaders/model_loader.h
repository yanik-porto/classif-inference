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
     * @return name of the class found
     */
    virtual std::string Execute(const std::string &imgPath) = 0;

    /**
     * @brief run inference on a batch of images given their paths
     * @param imgPaths paths to the images
     * @return names of the classes found
     */
    virtual std::vector<std::string> Execute(const std::vector<std::string> &imgPaths) = 0;

    /**
     * @brief run inference on several images sorted in class folders
     * 
     * @param folderPath Path to the folder containing folder of classes
     * @param nImgsPerBatch Number of images per batch
     */
    void ExecuteOnFolder(const std::string &folderPath, const int nImgsPerBatch);

protected:
    std::vector<std::string> _classes;
};

#endif /* MODEL_LOADER_H */