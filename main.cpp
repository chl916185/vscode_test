#include "torch/script.h"
#include "torch/torch.h"
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace at;
using namespace cv;
int main()
{
    int a=0;
    string model_path = "/home/chl/workspaces/torch_cc/vs_code_learn/0/model.pt";
    // Deserialize the ScriptModule from a file using torch::jit::load().
    shared_ptr<torch::jit::script::Module> module = torch::jit::load(model_path);

    assert(module != nullptr);
    cout << "ok\n";

    // Create a vector of inputs.
    vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    // Execute the model and turn its output into a tensor.
    auto output = module->forward(inputs).toTensor();

    cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
    // cv::waitKey(0);
    return 0;
}