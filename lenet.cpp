#include <iostream>
#include "mnist.hpp"
#include "tensor.hpp"

using namespace std;

int main(void)
{
    std::string mnistPath = "/mnt/d/Etudes/Computer-Vision/assignment/t10k-images-idx3-ubyte";
    MNIST mnist(mnistPath);
    // cout << "No. of images: " << mnist.totalimgs() << "\n"; 
    Tensor sample = mnist.at(10);
    
    return 0; 
}
