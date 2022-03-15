#include "model_loader.h"

#include <fstream>
#include <string>
#include <iostream>
#include <map>
#include <boost/filesystem.hpp>
using namespace boost::filesystem;

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

void ModelLoader::ExecuteOnFolder(const std::string &folderPath)
{
    boost::filesystem::path dataFolderPath = boost::filesystem::path(folderPath);
    std::cout << dataFolderPath << std::endl;
    if (!boost::filesystem::exists(dataFolderPath))
        return;

    std::map<std::string, float> accByClass;

    boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
    for (boost::filesystem::directory_iterator itr(dataFolderPath); itr != boost::filesystem::directory_iterator(); ++itr)
    {
        if (boost::filesystem::is_directory(itr->status()))
        {
            // Get folder name
            std::string className = itr->path().filename().c_str();
            std::cout << className << std::endl;
            int nImages = 0;
            int nCorrects = 0;

            for (boost::filesystem::directory_iterator itrClass(itr->path()); itrClass != end_itr; ++itrClass)
            {
                if (itrClass->path().extension() == ".jpg" || itrClass->path().extension() == ".png")
                {
                    std::string classFound = this->Execute(itrClass->path().c_str());
                    if (classFound == className) {
                        nCorrects++;
                    }
                    nImages++;
                }
            }

            accByClass[className] = nCorrects / static_cast<float>(nImages);
        }
    }

    std::cout << "**************" << std::endl;
    for (auto const &acc : accByClass)
    {
        std::cout << acc.first << " : " << acc.second << std::endl;
    }
    std::cout << "**************" << std::endl;
}