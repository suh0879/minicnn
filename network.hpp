#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>

enum class LayerType : uint8_t {
    Conv2d = 0,
    Linear,
    MaxPool2d,
    ReLu,
    SoftMax,
    Flatten
};

std::ostream& operator<< (std::ostream& os, LayerType layer_type) {
    switch (layer_type) {
        case LayerType::Conv2d:     return os << "Conv2d";
        case LayerType::Linear:     return os << "Linear";
        case LayerType::MaxPool2d:  return os << "MaxPool2d";
        case LayerType::ReLu:       return os << "ReLu";
        case LayerType::SoftMax:    return os << "SoftMax";
        case LayerType::Flatten:    return os << "Flatten";
    };
    return os << static_cast<std::uint8_t>(layer_type);
}

class Layer {
    public:
        Layer(LayerType layer_type) : layer_type_(layer_type), input_(), weights_(), bias_(), output_() {}

        virtual void fwd() = 0;
        virtual void read_weights_bias(std::ifstream& is) = 0;

        void print() {
            std::cout << layer_type_ << std::endl;
            if (!input_.empty())   std::cout << "  input: "   << input_   << std::endl;
            if (!weights_.empty()) std::cout << "  weights: " << weights_ << std::endl;
            if (!bias_.empty())    std::cout << "  bias: "    << bias_    << std::endl;
            if (!output_.empty())  std::cout << "  output: "  << output_  << std::endl;
        }
        // TODO: additional required methods  

        Tensor get_input()
        {
            //To get input from MNIST files: with pixel values for each  
        }

        float convolve(Tensor input_, Tensor weights_, n, c, h, w)
        {
            float value = 0.0; 
            //Weights_.N would get kernel input channel  = input_channel 
            //Returns the value after the convolution operation is performed on the elements 
            float kernel_center_w = weights_.w/2;
            float kernel_center_h = weights_.h/2;
            for (size_t kernel_c = 0; kernel_c < weights_.C; ++kernel_c)
            {
                for (size_t kernel_h = 0; kernet_h < weights_.H; ++kernel_h)
                {
                    for (size_t kernel_w = 0; kernel_w < weights_.W; ++kernel_w)
                    {
                        //convolve operation on each input - adjust the input centre 
                        size_t input_h = stride*h + kernel_h - kernel_center_h; 
                        size_t input_w = stride*w + kernel_w - kernel_center_w;
                        value += input_.operator(n, kernel_c, input_h, input_w)*weights_.operator(c, kernel_c, kernel_h, kernel_w)
                    }
                }
            }
            return value 
        }

        void add_bias(Tensor bias_)
        {
            // add's bias to the network
            for (size_t n = 0; n < )   
        }

    protected:
        const LayerType layer_type_;
        Tensor input_;
        Tensor weights_;
        Tensor bias_;
        Tensor output_;
};


class Conv2d : public Layer {
    public:
        Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride=1, size_t pad=0) 
        : Layer(LayerType::Conv2d), in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), pad_(pad) {}
 
    
        void read_weights_bias(std::ifstream& inputfile) override 
        {   
            // Get intput weights and biases for each 
        }
        

        void fwd() override //To override the virtual fwd() function mentioned above.  
        {
        
        // match_kernel_o/p & i/p channels to read weight and store the kernel sizes. 
        //

        // Getting parameters from input_ 
        // Where do we get the input parameters from - other than the channel input
        size_t input_N = input_.N;
        size_t input_C = in_channels_;  
        size_t input_H = input_.H; 
        size_t input_W = input_.W; 
        input_ = Tensor(input_N, input_C, input_H, input_W)
        
        //Calculating the output_ size 
        output_H = ((input_H + 2*pad - kernel_size_) / stride) + 1;   
        output_W = ((input_W + 2*pad - kernel_size_) / stride) + 1;  
        output_ = Tensor(input_N, out_channels_, output_H, output_W); 

        for (size_t n=0; n < output_.N; ++n)
        {
            for (size_t c=0; c < output_.C; ++c)
            {
                for (size_t h=0; h < output_.H; ++h)
                {
                    for (size_t w=0; w < output_.W; ++w)
                    {
                        output_.operator(n,c,h,w) = convolve(input_, kernel_, n, c, h, w);
                        output_.operator(n,c,h,w) += bias_.operator(n,c,h,w);
                    }
                }
            }
        }
    }
        
};


class Linear : public Layer {
    public:
        Linear(size_t in_features, size_t out_features) : Layer(LayerType::Linear), in_features_(in_features), out_features_(out_features) {}
    // TODO
};


class MaxPool2d : public Layer {
    public:
        MaxPool2d(size_t kernel_size, size_t stride=1, size_t pad=0) : Layer(LayerType::MaxPool2d), kernel_size_(kernel_size), stride_(stride), pad_(pad) {}
    // TODO
    void fwd()
    {

    }
};


class ReLu : public Layer {
    public:
        ReLu() : Layer(LayerType::ReLu) {}
    // TODO
    void fwd() override
    {
        for (size_t n = 0; n < input_.N; ++n)
        {
            for (size_t c = 0; c < input_.C; ++c)
            {
                for (size_t h = 0; h < input_.H; ++h)
                {
                    for (size_t w = 0; w < input_.W; ++w)
                    {
                        output_.operator()(n,c,h,w) = max(0.0, input_.operator()(n,c,h,w))
                    }
                }
            }
        }
    }
};


class SoftMax : public Layer {
    public:
        SoftMax() : Layer(LayerType::SoftMax) {}
    // TODO
    void fwd()
    {
        
    }   
    
};


class Flatten : public Layer {
    public:
        Flatten() : Layer(LayerType::Flatten) {}
    // TODO
};


class NeuralNetwork {
    public:
        NeuralNetwork(bool debug=false) : debug_(debug) {}

        void add(Layer* layer) {
            // TODO
            assert(layer != NULL); 

        }

        void load(std::string file) {
            // TODO
        }

        Tensor predict(Tensor input) {
            // TODO
        }

    private:
        bool debug_;
        // TODO: storage for layers
};

#endif // NETWORK_HPP
