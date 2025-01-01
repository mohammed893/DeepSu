#include "../include/graph.hpp"
using namespace std;
// Addition Node
Node* Graph::add(Node* a, Node* b) {
    Node* node = new Node();
    node->inputs = { a, b };

    // Forward pass
    node->forward = [node, a, b]() {
        node->value = make_shared<Tensor>(Tensor(vector<float>()));
        for (size_t i = 0; i < a->value->data.size(); ++i) {
            node->value->data.push_back(a->value->data[i] + b->value->data[i]);
        }
        };

    // Backward pass
    node->backward = [a, b](vector<float>& grad) {
        a->gradient.resize(grad.size());
        b->gradient.resize(grad.size());
        for (size_t i = 0; i < grad.size(); ++i) {
            a->gradient[i] += grad[i];
            b->gradient[i] += grad[i];
        }
        };

    return node;
}

// Multiplication Node
Node* Graph::multiply(Node* a, Node* b) {
    Node* node = new Node();
    node->inputs = { a, b };

    // Forward pass
    node->forward = [node, a, b]() {
        node->value = make_shared<Tensor>(Tensor(vector<float>()));
        for (size_t i = 0; i < a->value->data.size(); ++i) {
            node->value->data.push_back(a->value->data[i] * b->value->data[i]);
        }
        };

    // Backward pass
    node->backward = [a, b](vector<float>& grad) {
        a->gradient.resize(grad.size());
        b->gradient.resize(grad.size());
        for (size_t i = 0; i < grad.size(); ++i) {
            a->gradient[i] += grad[i] * b->value->data[i];
            b->gradient[i] += grad[i] * a->value->data[i];
        }
        };

    return node;
}

// Forward Pass
void Graph::forward(Node* node) {
    for (Node* input : node->inputs) {
        forward(input);
    }
    if (node->forward) {
        node->forward();
    }
}

// Backward Pass
void Graph::backward(Node* node) {
    if (node->gradient.empty()) {
        node->gradient.resize(node->value->data.size(), 1.0f); // Initialize gradient for the final node
    }

    if (node->backward) {
        node->backward(node->gradient);
    }

    for (Node* input : node->inputs) {
        backward(input);
    }
}
