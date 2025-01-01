#include "../graph.hpp"
using namespace std;

class DenseLayer : public Node {
public:
    shared_ptr<Tensor> weights;  // Weight matrix
    shared_ptr<Tensor> biases;   // Bias vector
    function<void()> forward;    // Forward pass function

    // Constructor for Dense Layer
    DenseLayer(Node* input, size_t output_size) {
        // Initialize weights (shape: input_size x output_size)
        this->weights = make_shared<Tensor>(Tensor(vector<float>(input->value->data.size() * output_size, 0.01f)));  // Random init

        // Initialize biases (shape: output_size)
        this->biases = make_shared<Tensor>(Tensor(vector<float>(output_size, 0.0f)));  // Zero init

        // Define the forward pass function
        this->forward = [this, input]() {
            // Create an empty tensor for the output of the layer
            this->value = make_shared<Tensor>(Tensor(vector<float>(this->biases->data.size(), 0.0f)));

            // Compute the weighted sum + bias for each output unit
            for (size_t i = 0; i < this->biases->data.size(); ++i) {
                float sum = 0.0f;
                // Perform the matrix-vector multiplication: y = Wx
                for (size_t j = 0; j < input->value->data.size(); ++j) {
                    sum += input->value->data[j] * this->weights->data[i * input->value->data.size() + j];
                }
                sum += this->biases->data[i];  // Add the bias
                this->value->data[i] = sum;   // Store the result
            }

            // Apply activation function (ReLU for example)
            for (auto& val : this->value->data) {
                val = std::max(0.0f, val);  // ReLU activation
            }
            };
    }
};

