#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include "model_loaders/model_loader_torch_script.h"

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  ModelLoader *loader = new ModelLoaderTorchScript();
  loader->Load(argv[1]);
  loader->Execute(torch::ones({1, 3, 224, 224}));
  

  std::cout << "ok\n";
}
