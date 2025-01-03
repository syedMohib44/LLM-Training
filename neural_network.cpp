#include <torch/torch.h>
#include <iostream>

// Define the Neural Network
struct CNN : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};

    // Constructor
    CNN() {
        conv1 = register_module("conv1", torch::nn::Conv2d(1, 16, 3));
        conv2 = register_module("conv2", torch::nn::Conv2d(16, 32, 3));
        fc1 = register_module("fc1", torch::nn::Linear(32 * 6 * 6, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::max_pool2d(x, 2);
        x = torch::relu(conv2->forward(x));
        x = torch::max_pool2d(x, 2);
        x = x.view({x.size(0), -1});
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return torch::log_softmax(x, 1);
    }
};

int main() {
    // Check if CUDA is available
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Using device: " << (device.type() == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    // Load dataset
    auto dataset = torch::data::datasets::MNIST("./data")
                       .map(torch::data::transforms::Normalize<>(0.5, 0.5))
                       .map(torch::data::transforms::Stack<>());

    auto data_loader = torch::data::make_data_loader(
        std::move(dataset), torch::data::DataLoaderOptions().batch_size(64).workers(2));

    // Initialize model and move to GPU
    CNN model;
    model.to(device);

    // Define optimizer
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01).momentum(0.9));

    // Training loop
    for (size_t epoch = 0; epoch < 10; ++epoch) {
        size_t batch_idx = 0;
        for (auto& batch : *data_loader) {
            model.train();

            // Move data and target to GPU
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);

            // Forward pass
            auto output = model.forward(data);

            // Compute loss
            auto loss = torch::nll_loss(output, target);

            // Backward pass and optimization
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // Print loss
            if (batch_idx++ % 10 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_idx
                          << " | Loss: " << loss.item<double>() << std::endl;
            }
        }
    }

    return 0;
}
