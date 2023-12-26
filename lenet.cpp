#include "tensor.hpp"
#include "network.hpp"
#include "mnist.hpp"
#include <iostream> 
#include <fstream> 
#include <string>
#include <algorithm>
#include <math.h>

void assertval(float expected, float actual, float epsilon = 1e-5) 
{
        assert(std::abs(expected - actual) < epsilon);
}

int main(void)
{

    std::string mnistPath = "/mnt/d/Etudes/Computer-Vision/assignment/t10k-images-idx3-ubyte";
    MNIST mnist(mnistPath);
    
  
    // Entire LeNet-5 described; // 
    // 1st Convolution 
    Conv2d conv_layer(1, 6, 5);
    Tensor input_ = mnist.at(0); 
    std::cout << input_ << std::endl; 
    conv_layer.set_input(input_); 
    std::ifstream file("/mnt/d/Etudes/Computer-Vision/assignment/lenet.raw", std::ios::in | std::ios::binary); 
    conv_layer.read_weights_bias(file); 
    conv_layer.fwd(); 
    conv_layer.print();
    // 1st ReLu 
    Tensor input_2 = conv_layer.get_output(); 
    ReLu relu_1; 
    relu_1.set_input(input_2); 
    relu_1.fwd(); 
    relu_1.print();
    // 1st MaxPool
    MaxPool2d  maxpool_1(2); // not changing dimensions as seen: 
    Tensor input_3 =  relu_1.get_output(); 
    maxpool_1.set_input(input_3);
    maxpool_1.fwd(); 
    maxpool_1.print();
    // second convolution
    Conv2d conv_layer_2(6, 16, 5); 
    Tensor input_4 = maxpool_1.get_output(); 
    conv_layer_2.set_input(input_4);
    conv_layer_2.read_weights_bias(file); 
    conv_layer_2.fwd(); 
    conv_layer_2.print(); 
    // second relu 
    ReLu relu_2; 
    relu_2.set_input(conv_layer_2.get_output()); 
    relu_2.fwd(); 
    relu_2.print();
    // second maxpool2d 
    MaxPool2d maxpool_2(2); 
    maxpool_2.set_input(relu_2.get_output()); 
    maxpool_2.fwd();
    maxpool_2.print();
    // flatten the maxpool
    Flatten flatten_1; 
    flatten_1.set_input(maxpool_2.get_output());
    flatten_1.fwd(); 
    flatten_1.print(); 
    // linear layer 1
    Linear linear_1(400, 120); 
    linear_1.set_input(flatten_1.get_output()); 
    linear_1.read_weights_bias(file);
    linear_1.fwd(); 
    linear_1.print();
    // thrid relu
    ReLu relu_3; 
    relu_3.set_input(linear_1.get_output());
    relu_3.fwd();
    relu_3.print(); 
    // linear layer 2
    Linear linear_2(120, 84); 
    linear_2.set_input(relu_3.get_output());
    linear_2.read_weights_bias(file); 
    linear_2.fwd();
    linear_2.print();
    // fourth relu 
    ReLu relu_4;
    relu_4.set_input(linear_2.get_output());
    relu_4.fwd();
    relu_4.print(); 
    // linear layer 3 
    Linear linear_3(84, 10);
    linear_3.set_input(relu_4.get_output()); 
    linear_3.read_weights_bias(file);
    linear_3.fwd();
    linear_3.print();
    // fifth relu 
    ReLu relu_5;
    relu_5.set_input(linear_3.get_output());
    relu_5.fwd();
    relu_5.print(); 
    // softmax 
    SoftMax softmax_1;
    softmax_1.set_input(relu_5.get_output());
    softmax_1.fwd();
    softmax_1.print(); 
    softmax_1.get_output().print();
    // Seeing the output values 
    
    
    

    
      /*
    // Defining the network
    NeuralNetwork LeNet;
    LeNet.add(new Conv2d(1, 6, 5));
    LeNet.add(new ReLu());
    LeNet.add(new MaxPool2d(2));
    LeNet.add(new Conv2d(6, 16, 5));
    LeNet.add(new ReLu());
    LeNet.add(new MaxPool2d(2));
    LeNet.add(new Flatten());
    LeNet.add(new Linear(400, 120));
    LeNet.add(new ReLu());
    LeNet.add(new Linear(120, 84));
    LeNet.add(new ReLu());
    LeNet.add(new Linear(84, 10));
    LeNet.add(new ReLu());
    LeNet.add(new SoftMax());


    // Loading bias & weights; 
    LeNet.load("/mnt/d/Etudes/Computer-Vision/assignment/lenet.raw");

    // Predict using an input -> 0 : "7"; 
    size_t input_index = 2;
    Tensor output = LeNet.predict(mnist.at(input_index));
    //mnist.at(input_index).print();
    mnist.print(input_index);
    output.print();
    */
    return 0; 
}

