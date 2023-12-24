#include <iostream>
#include <fstream>
#include "mnist.hpp"
#include "tensor.hpp"

using namespace std;

int main(void)
{
    std::string mnistPath = "/mnt/d/Etudes/Computer-Vision/assignment/t10k-images-idx3-ubyte";
    MNIST mnist(mnistPath);
    // cout << "No. of images: " << mnist.totalimgs() << "\n"; 
  /*
    Tensor sample = mnist.at(10);
    for (size_t n = 0; n < mnist.total_imgs(); ++n)
    {
        mnist.print(n);
    } 
    input.print(); 
       
/
   / Model definition 
    NeuralNetwork lenet(); 
    
    //Engaging the predict - forward propogation 
    Tensor output = lenet.predict(input);  
      

    /* 
    std::string lenetpath = "/mnt/d/Etudes/Computer-Vision/assignment/lenet.raw"
    //std::ifstream file(lenetpath, std::ios::binary); 

    file.read()
    weights = Tensor()
    */
    mnist.print(1); 
    return 0; 
}


/*
void load()
{
    //TODO: Read and print weights / biases for lenet.raw conv-1-...
    float weights[];    
    std::ifstream file(lenetpath, std::ios::in | std::ios::binary); 
    file.read(reinterpret_cast<char *>)
}
*/
