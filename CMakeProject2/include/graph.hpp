#include <iostream>
#include <vector>
#include <memory>
#include <functional>

class Tensor {
public:
    std::vector<float> data;
    Tensor(std::vector<float> values) : data(std::move(values)) {}
};

class Node {
public:
    std::shared_ptr<Tensor> value;                      // Computed value in the forward pass
    std::function<void()> forward;                     // Forward pass computation
    std::function<void(std::vector<float>&)> backward; // Backward pass gradient computation

    std::vector<Node*> inputs;                         // Input nodes
    std::vector<float> gradient;                       // Gradient for this node

    Node() : value(nullptr) {}
};

class Graph {
public:
    static Node* add(Node* a, Node* b);
    static Node* multiply(Node* a, Node* b);
    static void forward(Node* node);
    static void backward(Node* node);
};
