#include "model_loader.h"

#include <fstream>
#include <string>
#include <iostream>

void ModelLoader::LoadClasses(const std::string &classesPath)
{
    std::cout << "Read classes from : " << classesPath << std::endl;

    _classes.clear();
    std::ifstream classesFile(classesPath);
    std::string line;
    if (classesFile.is_open())
    {
        while (getline(classesFile, line))
        {
            _classes.push_back(line);
        }
        classesFile.close();
    }
    std::cout << _classes.size() << " classes loaded" << std::endl;
}
