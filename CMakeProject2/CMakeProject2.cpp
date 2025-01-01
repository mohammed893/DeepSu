
#include "include/graph.hpp"
using namespace std;
int main() {
    // Define tensors
    Node* a = new Node();
    a->value = std::make_shared<Tensor>(Tensor({ 1.0f, 2.0f, 3.0f }));

    Node* b = new Node();
    b->value = std::make_shared<Tensor>(Tensor({ 4.0f, 5.0f, 6.0f }));

    // Create graph: c = (a + b) * b
    Node* sum = Graph::add(a, b);
    Node* product = Graph::multiply(sum, b);

    // Forward pass
    Graph::forward(product);

    std::cout << "Forward pass result: ";
    for (float val : product->value->data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Backward pass
    Graph::backward(product);

    std::cout << "Gradients for 'a': ";
    for (float grad : a->gradient) {
        std::cout << grad << " ";
    }
    std::cout << std::endl;

    std::cout << "Gradients for 'b': ";
    for (float grad : b->gradient) {
        std::cout << grad << " ";
    }
    std::cout << std::endl;

    // Clean up
    delete a;
    delete b;
    delete sum;
    delete product;

    return 0;
}
