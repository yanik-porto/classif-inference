#include <iostream>
#include <memory>

#include "model_loaders/model_loader_tensorrt.h"

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-model>\n";
    return -1;
  }

  ModelLoader *loader = new ModelLoaderTensorRt();
  loader->Load(argv[1]);

  std::cout << "ok\n";
}
