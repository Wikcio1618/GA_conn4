#include <../include/conn4model.h>

Conn4Model::Conn4Model(int width) : width(width)
{
    // Board dimensions (6x7)
    height = 6;
    conv_channels = 16;

    // Convolutional layer
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, conv_channels, 4)));

    // Flattened board representation after convolution
    conv_output_size = (height - 3) * (width - 3) * conv_channels;

    // Fully connected layers
    fc1 = register_module("fc1", torch::nn::Linear(conv_output_size, 16));
    fc2 = register_module("fc2", torch::nn::Linear(16, width));

    // Set the data type of parameters to double after creating the layers
    conv1->weight.data().to(torch::kDouble);
    fc1->weight.data().to(torch::kDouble);
    fc2->weight.data().to(torch::kDouble);
}

torch::Tensor Conn4Model::forward(torch::Tensor x)
{
    x = x.to(torch::kDouble);
    x = torch::relu(conv1->forward(x));
    x = x.view({x.size(0), -1}); // Flatten the output of Conv2d
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    return torch::softmax(x, 1);
}

std::vector<double> Conn4Model::get_parameters()
{
    std::vector<double> parameters;
    for (const auto &pair : named_parameters())
    {
        auto param = pair.value();
        auto param_data = param.data_ptr<double>();
        parameters.insert(parameters.end(), param_data, param_data + param.numel());
    }
    return parameters;
}

std::vector<size_t> Conn4Model::get_weight_breakpoints() const
{
    std::vector<size_t> breakpoints;

    // Iterate through each layer and add the number of parameters to breakpoints
    for (const auto &param : parameters())
    {
        // Assuming each parameter is a torch::Tensor
        size_t num_params_in_layer = param.numel();
        breakpoints.push_back(num_params_in_layer);
    }

    // Add the total number of parameters as the last breakpoint
    size_t total_params = std::accumulate(breakpoints.begin(), breakpoints.end(), 0);
    breakpoints.push_back(total_params);

    return breakpoints;
}

// In the source file (conn4model.cpp)
void Conn4Model::set_parameters(const std::vector<double> &parameters)
{
    size_t start_idx = 0;

    // Iterate through each registered parameter in the model
    for (auto &param : this->parameters())
    {
        // Calculate the number of elements in the parameter
        size_t num_elements = param.numel();

        // Extract the corresponding slice from the 1D parameter array
        std::vector<double> param_slice(parameters.begin() + start_idx, parameters.begin() + start_idx + num_elements);

        // Reshape and set the parameter
        torch::from_blob(param_slice.data(), param.sizes().vec(), torch::kDouble).copy_(param);

        // Move to the next slice
        start_idx += num_elements;
    }
}