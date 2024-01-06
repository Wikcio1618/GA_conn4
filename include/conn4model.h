// Population.h

#ifndef CONN4MODEL_H
#define CONN4MODEL_H

#include <vector>
#include <torch/torch.h>

class Conn4Model : public torch::nn::Module{
public:
    // ... other class members ...
    Conn4Model(int width);
    torch::Tensor forward(torch::Tensor x);
    std::vector<double> get_parameters();
    std::vector<size_t> get_weight_breakpoints() const;
    void set_parameters(const std::vector<double>& parameters);

private:
    int height;
    int width;
    int conv_channels;
    int conv_output_size;
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    // ... other private members ...
};

#endif // CONN4MODEL