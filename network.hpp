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
                        // convolve operation on each input - adjust the input centre 
                        size_t input_h = stride*h + kernel_h - kernel_center_h; 
                        size_t input_w = stride*w + kernel_w - kernel_center_w;
                        value += input_.operator(n, kernel_c, input_h, input_w)*weights_.operator(c, kernel_c, kernel_h, kernel_w)
                    }
                }
            }
            return value 
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
            // Get intput weights and biases for each -> where you get kernels from
            float weight; 
            float bias; 
            inputfile.read(reinterpret_cast<char *>(&weight), reinterpret_cast<char *>(&weight) + sizeof(&weight)); 
            inputfile.read(reinterpret_cast<char *>(&bias), reinterpret_cast<char *>(&bias) + sizeof(bias)); 
            
            weights_ = Tensor()
        }
        
        

        void fwd() override 
        {
        
        // match_kernel_o/p & i/p channels to read weight and store the kernel sizes. 
        //

        // Getting parameters from input_ 
        // Where do we get the input parameters from - other than the channel input
        input_ = get_input() // -> input from MNIST dataset
        size_t input_N = input_.N;
        size_t input_C = in_channels_;  
        size_t input_H = input_.H; 
        size_t input_W = input_.W; 
        
        
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
        void read_weights_bias(std::ifstream& inputfile) override 
        {   
            // Get intput weights and biases for each -> where you get kernels from
        }

        void fwd()
        {

        }
};


class MaxPool2d : public Layer {
    public:
        MaxPool2d(size_t kernel_size, size_t stride=1, size_t pad=0) : Layer(LayerType::MaxPool2d), kernel_size_(kernel_size), stride_(stride), pad_(pad) {}
    // Calculating output size
    output_H = ((input_H + 2*pad - kernel_size_) / stride) + 1;   
    output_W = ((input_W + 2*pad - kernel_size_) / stride) + 1;      
    output_ = Tensor(N, C, H, W);  
    void fwd()
    {
        for (size_t n=0; n < output_.N; ++n)
        {
            for (size_t c=0; c < output_.C; ++c)
            {
                for (size_t h=0; h < output_.H; ++h)
                {
                    for(size_t w=0; w < output_.W; ++w)
                    {
                        output_.operator(n,c,h,w) = convolve(input_, kernel_, n, c, h, w);
                        output_.operator(n,c,h,w) += bias_.operator(n, c, h, w);
                    }
                }
            }
        }

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
                        output_(n,c,h,w) = max(0.0, input_(n,c,h,w))
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
        float sum_exp = 0.0;
        for (size_t w = 0; w < input_.w; ++w)
        {
            sum_exp += exp(input_(0,0,0,w)); 
        } 
        for (size_t i = 0; i < input_.W; ++i)
        {
            output_(0,0,0,w) = exp(input_(0,0,0,w)) / sum_exp; 
        }
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
.
        void add(Layer* layer) {
            // TODO
            // This works by trying to "push" all the layers into one container. 
            assert(layer != NULL); 
            lenet.push_back(layer); 

        }

        void load(std::string file) {
            // TODO
            // only for layers where you implmement read_weight_bias() you need to take care of loading the weights and biases   
            for (auto layer : lenet)
            {
                layer->read_weights_bias(); 
            }
        }

        Tensor predict(Tensor input) {
            // TODO
            //print the input digit along with the probabilities 
            Tensor ouput = input; 
            //add layers using the method 
            lenet.add(Conv2d(1, 6, 5));
            lenet.add(ReLu());
            lenet.add(MaxPool2d(2)); 
            lenet.add(Conv2d(6, 16, 5));
            lenet.add(ReLu());
            lenet.add(MaxPool2d(2)); 
            lenet.add(Flatten());
            lenet.add(Linear(400, 120));
            lenet.add(ReLu());
            lenet.add(Linear(120, 84));
            lenet.add(ReLu());
            lenet.add(Linear(84, 10));
            lenet.add(ReLu()); 
            lenet.add(SoftMax()); 

            //printting the output of the layers. 
            for (auto layer : lenet)
            {
                output = layer->fwd(output) 
            }         
            
            if(debug_)
            {
                for (size_t w = 0; w < output_.W; ++w)
                { 
                std::cout << w << "\n" << output_(0,0,0,w) << "\n"; 
                }
            }
            return output_
        }
    private:
        bool debug_;
        std::vector<Layer> lenet;   
        // TODO: storage for layers
};

#endif // NETWORK_HPP
