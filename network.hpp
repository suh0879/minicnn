#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>
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

        void set_input(Tensor input) { input_ = input; }

        void set_weights(Tensor weights) { weights_ = weights; } // only used to conduct tests

        void set_bias(Tensor bias) { bias_ = bias; } // only used to conduct tests
        
        Tensor get_bias() const { return bias_; }

        Tensor get_weights() const { return weights_; }
        
        Tensor get_output() { return output_; }

        LayerType get_layer_type() { return layer_type_; }

        Tensor get_input() { return input_; }

    protected:
        const LayerType layer_type_;
        Tensor input_;
        Tensor weights_;
        Tensor bias_;
        Tensor output_;
};


class Conv2d : public Layer {
    public:
        size_t in_channels;
        size_t out_channels;
        size_t kernel_size;
        size_t stride;
        size_t pad;
        size_t output_H;
        size_t output_W;
        
        Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride=1, size_t pad=0) 
        : Layer(LayerType::Conv2d), in_channels(in_channels), out_channels(out_channels),
          kernel_size(kernel_size), stride(stride), pad(pad)
          {
                weights_ = Tensor(out_channels, in_channels, kernel_size, kernel_size);
                bias_ = Tensor(1,1,1,out_channels); 
          }
       
        void read_weights_bias(std::ifstream& inputfile) 
        {
            if(!inputfile.is_open())
            {
                std::cerr << "Cannot open file" << std::endl;
            }
            for (size_t oc = 0; oc < out_channels; ++oc)
            {
                for (size_t ic = 0; ic < in_channels; ++ic)
                {
                    for (size_t h = 0; h < kernel_size; ++h)
                    {
                        for (size_t w = 0; w < kernel_size; ++w)
                        {
                            float weights; 
                            inputfile.read(reinterpret_cast<char*>(&weights) , sizeof(float));
                            weights_(oc, ic, h, w) = weights; 
                        }
                    }
                }
            }

            for (size_t b = 0; b < out_channels; ++b)
            {
                float bias; 
                inputfile.read(reinterpret_cast<char *>(&bias), sizeof(float));
                bias_(0,0,0,b) = bias; 
            }
        }

        void fwd() {
        if (input_.empty() || weights_.empty() || bias_.empty()) {
            if (weights_.empty()) { std::cerr << "Weights is null." << std::endl; }
            if (bias_.empty()) { std::cerr << "Bias is null." << std::endl; }
            if (input_.empty()) { std::cerr << "Input is null." << std::endl; }
            return;
        }

        output_H = ((input_.H + 2 * pad - kernel_size) / stride) + 1;
        output_W = ((input_.W + 2 * pad - kernel_size) / stride) + 1;
        output_ = Tensor(input_.N, out_channels, output_H, output_W);

        for (size_t n = 0; n < output_.N; ++n) {
            for (size_t c = 0; c < output_.C; ++c) {
                for (size_t h = 0; h < output_.H; ++h) {
                    for (size_t w = 0; w < output_.W; ++w) {
                        float value = 0.0;
                        for (size_t wi = 0; wi < weights_.C; ++wi) {
                            for (size_t wh = 0; wh < weights_.H; ++wh) {
                                for (size_t ww = 0; ww < weights_.W; ++ww) {
                                    size_t input_h = stride * h + wh - pad;
                                    size_t input_w = stride * w + ww - pad;
                                    if (input_h < 0 || input_h >= input_.H || input_w < 0 || input_w >= input_.W) {
                                        std::cerr << "Error: Input indices out of bounds." << std::endl;
                                        return;
                                    }
                                    value += input_(n, wi, input_h, input_w) * weights_(c, wi, wh, ww);
                                }
                            }
                        }
                        output_(n, c, h, w) = value + bias_(0, 0, 0, c);
                    }
                }
            }
        }

        if (output_.empty()) {
            std::cout << "Output tensor is empty after fwd()! \n";
        }
    }   

};


class Linear : public Layer {
    public:
        size_t in_features;
        size_t out_features; 
        
        Linear(size_t in_features, size_t out_features) 
        : Layer(LayerType::Linear), in_features(in_features), out_features(out_features) 
        {
            weights_ = Tensor(1, 1, out_features, in_features);
            bias_ = Tensor(1, 1, 1, out_features); 
        }

        void read_weights_bias(std::ifstream& inputfile)
        {
            if(!inputfile.is_open())
            {
                std::cerr << "Cannot open file" << std::endl;
            }
            for (size_t wn = 0; wn < weights_.N; ++wn)
            {
                for (size_t wc = 0; wc < weights_.C; ++wc)
                {
                    for (size_t wh = 0; wh < weights_.H; ++wh)
                    {
                        for (size_t ww = 0; ww < weights_.W; ++ww)
                        {
                            float weights; 
                            inputfile.read(reinterpret_cast<char*>(&weights) , sizeof(float));
                            weights_(wn, wc, wh, ww) = weights; 
                        }
                    }
                }
            }
            for (size_t b = 0; b < out_features; ++b)
            {
                float bias; 
                inputfile.read(reinterpret_cast<char *>(&bias), sizeof(float)); 
                bias_(0,0,0,b) = bias;
            }
        }

        void fwd()
        {
            if (input_.empty() || weights_.empty() || bias_.empty()) 
            {
                if (weights_.empty()) {std::cerr << "Weights is null." << std::endl; }
                if (bias_.empty()) {std::cerr << "Bias is null." << std::endl; }
                if (input_.empty()) {std::cerr << "Input is null." << std::endl; }
                return;  
            }
            output_ = Tensor(input_.N, input_.C, input_.H, out_features); 
            for (size_t n = 0; n < output_.N; ++n)
            {
                for (size_t c = 0; c < output_.C; ++c)
                {
                    for (size_t h = 0; h < output_.H; ++h)
                    {
                        for (size_t w = 0; w < output_.W; ++w)
                        {
                            // Matrix Multiplication 
                            float value = 0.0;
                            for (size_t i = 0; i < in_features; ++i)
                            {
                                value += input_(n, c, h, i)*weights_(n, c, w, i); // oc - ic  
                            }
                            output_(n, c, h, w) = value + bias_(0, 0, 0, w); 
                        }
                    }
                }
            }
            if (output_.empty())
            {
                std::cout << "Output tensor is empty after fwd()! \n"; 
            }
        }
};


class MaxPool2d : public Layer {
    public:
    size_t output_H;
    size_t output_W;
    size_t kernel_size; 
    size_t stride; 
    size_t pad; 

        MaxPool2d(size_t kernel_size, size_t stride=2, size_t pad=0) 
        : Layer(LayerType::MaxPool2d), kernel_size(kernel_size), stride(stride), pad(pad)
        {

        }

        void read_weights_bias(std::ifstream& inputfile)
        {

        }

        void fwd()
        {
            output_H = ((input_.H + 2 * pad - kernel_size) / stride) + 1;
            output_W = ((input_.W + 2 * pad - kernel_size) / stride) + 1;
            output_ = Tensor(input_.N, input_.C, output_H, output_W); 
            for (size_t n = 0; n < output_.N; ++n)
            {
                for (size_t c = 0; c < output_.C; ++c)
                {
                    for (size_t h = 0; h < output_.H; ++h)
                    {
                        for (size_t w = 0; w < output_.W; ++w)
                        {
                            std::vector<float> values; 
                            for (size_t kh = 0; kh < kernel_size; ++kh)
                            {
                                for (size_t kw = 0; kw < kernel_size; ++kw)
                                {
                                    size_t input_h = stride * h + kh - 2*pad;
                                    size_t input_w = stride * w + kw - 2*pad;
                                    if (input_h >= 0 && input_h < input_.H && input_w >= 0 && input_w < input_.W) 
                                    {
                                        values.push_back(input_(n,c,input_h,input_w)); 
                                    }
                                    else 
                                    {
                                        values.push_back(0.0); 
                                    } 
                                }
                            }
                            output_(n,c,h,w) = *(std::max_element(values.begin(), values.end())); 
                        }
                    }
                }
            }
            if (output_.empty())
            {
                std::cout << "Output tensor is empty after fwd()! \n"; 
            }
        }
};


class ReLu : public Layer {
    public:
        ReLu() : Layer(LayerType::ReLu) {}
    // TODO
        void read_weights_bias(std::ifstream& inputfile)  
        {

        }
        void fwd() 
        {
            output_ = Tensor(input_.N, input_.C, input_.H, input_.W);
            for (size_t n = 0; n < input_.N; ++n)
            {
                for (size_t c = 0; c < input_.C; ++c)
                {
                    for (size_t h = 0; h < input_.H; ++h)
                    {
                        for (size_t w = 0; w < input_.W; ++w)
                        {
                            output_(n,c,h,w) = std::max(0.0f, input_(n,c,h,w));
                        }
                    }
                }
            }
            if (output_.empty())
            {
                std::cout << "Output tensor is empty after fwd()! \n"; 
            }
        }
};


class SoftMax : public Layer {
    public:
        SoftMax() : Layer(LayerType::SoftMax) {}
    // TODO
        void read_weights_bias(std::ifstream& inputfile)
        {

        }
        void fwd()
        {   
            output_ = Tensor(input_.N, input_.C, input_.H, input_.W);

            for (size_t n = 0; n < input_.N; ++n) 
            {
                for (size_t c = 0; c < input_.C; ++c) 
                {
                    for (size_t h = 0; h < input_.H; ++h) 
                    {
                        float sum_exp = 0.0f;
                        for (size_t w = 0; w < input_.W; ++w) 
                        {
                            sum_exp += exp(input_(n, c, h, w));
                        }

                        // Apply SoftMax
                        for (size_t w = 0; w < input_.W; ++w) 
                        {
                            output_(n, c, h, w) = exp(input_(n, c, h, w)) / sum_exp;
                        }
                    }
                }
            }

            if (output_.empty())
            {
                std::cout << "Output tensor is empty after fwd()! \n"; 
            }
        } 
};


class Flatten : public Layer {
    public: 
        Flatten() : Layer(LayerType::Flatten) {}

        void read_weights_bias(std::ifstream& inputfile)
        {

        }

        void fwd()
        {
            output_ = Tensor(input_.N, 1, 1, input_.C*input_.H*input_.W); 
            for (size_t n = 0; n < input_.N; ++n)
            {
                for (size_t c = 0; c < input_.C; ++c)
                {
                    for (size_t h = 0; h < input_.H; ++h)
                    {
                        for (size_t w = 0; w < input_.W; ++w)
                        {
                            size_t flat_index = n*input_.C*input_.H*input_.W+ c*input_.H*input_.W + h*input_.W + w; 
                            output_(n,0,0,flat_index) = input_(n,c,h,w);
                        }
                    }
                }
            }
              if (output_.empty())
            {
                std::cout << "Output tensor is empty after fwd()! \n"; 
            }
        }
};


class NeuralNetwork {
    public:
        NeuralNetwork(bool debug=false) : debug_(debug) {}

        void add(Layer* layer) 
        {
            NN.push_back(layer);
        }

        void load(std::string file) 
        {
            std::ifstream inputfile(file, std::ios::in | std::ios::binary); 
             
            if (!inputfile.is_open())
            {
                std::cerr << "Failed to open file: " << file << std::endl;
                return;
            }
            for (const auto& layer : NN)
            {
                layer->read_weights_bias(inputfile); 
            }
        }

        Tensor predict(Tensor input) 
        {
            Tensor output = input;
            
            for (const auto& layer : NN)
            {
                layer->set_input(output); 
                layer->fwd();
                output = layer->get_output();
                if (debug_)
                {
                    layer->print();
                    std::cout << "input : \n";
                    layer->get_input().print();
                    std::cout << "bias : \n";
                    layer->get_bias().print();
                    std::cout << "weights : \n";
                    layer->get_weights().print();
                    std::cout << "output : \n";
                    layer->get_output().print();
                }
            }
            if (debug_)
            {
                for (size_t n = 0; n < output.W; ++n)
                {
                    
                    std::cout << n  << ": " << output(0,0,0,n) << "\n";
                }
            } 
            return output;
        }

    private:
        bool debug_;
        // storage for layers
        std::vector<Layer *> NN; 
        
};

#endif // NETWORK_HPP