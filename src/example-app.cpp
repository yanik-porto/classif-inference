#include <iostream>
#include <memory>

#include "model_loaders/model_loader_openvino.h"

int main(int argc, const char* argv[]) {
  if (argc != 4) {
    std::cerr << "usage: example-app <path-to-exported-model> <path-to-classes> <path-to-data-folder>\n";
    return -1;
  }

  ModelLoader *loader = new ModelLoaderOpenVino();
  loader->Load(argv[1]);
  loader->LoadClasses(argv[2]);
  loader->ExecuteOnFolder(argv[3]);

  std::cout << "ok\n";
  return 0;
}