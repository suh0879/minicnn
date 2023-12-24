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

        void set_input(Tensor input)
        {
            input_ = input; 
        }

        Tensor get_output()
        {
            return output_; 
        }
        // TODO: additional required methods

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
            // Just to initialize Conv2d specific variables; 
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
                            inputfile.read(reinterpret_cast<char*>(&weights), sizeof(weights));
                            weights_(oc, ic, h, w) = weights; 
                        }
                    }
                }
            }

            for (size_t b = 0; b < output_.C; ++b)
            {
                float bias; 
                inputfile.read(reinterpret_cast<char*>(&bias), sizeof(bias));
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
            output_H = ((input_.H + 2 * pad - kernel_size) / stride) + 1;
            output_W = ((input_.W + 2 * pad - kernel_size) / stride) + 1;
            output_ = Tensor(input_.N, out_channels, output_H, output_W); 
            for (size_t n = 0; n < output_.N; ++n)
            {
                for (size_t c = 0; c < output_.C; ++c)
                {
                    for (size_t h = 0; h < output_.H; ++h)
                    {
                        for (size_t w = 0; w < output_.W; ++w)
                        {
                            float value = 0.0; 
                            for (size_t wo = 0; wo < weights_.N; ++wo)
                            {
                                for (size_t wi = 0; wi < weights_.C; ++wi)
                                {
                                    for (size_t wh = 0; wh < weights_.H; ++wh)
                                    {
                                        for (size_t ww = 0; ww < weights_.W; ++ww)
                                        {
                                            size_t input_h = stride * h + wh - 2*pad;
                                            size_t input_w = stride * w + ww - 2*pad;
                                            if (input_h < 0 || input_h >= input_.H || input_w < 0 || input_w >= input_.W) 
                                            {
                                                std::cerr << "Error: Input indices out of bounds." << std::endl;
                                                return;
                                            }
                                            value += input_(n, wi, input_h, input_w)*weights_(wo, wi, wh, ww);
                                        }
                                    }
                                }

                            }
                            output_(n,c,h,w) = value + bias_(0,0,0,c);
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


class Linear : public Layer {
    public:
        size_t in_features;
        size_t out_features; 
        
        Linear(size_t in_features, size_t out_features) 
        : Layer(LayerType::Linear), in_features(in_features), out_features(out_features) 
        {
            weights_ = Tensor(1, 1, in_features, out_features);
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
                            inputfile.read(reinterpret_cast<char*>(&weights), sizeof(weights));
                            weights_(wn, wc, wh, ww) = weights; 
                        }
                    }
                }
            }
            for (size_t b = 0; b < weights_.W; ++b)
            {
                float bias; 
                inputfile.read(reinterpret_cast<char*>(&bias), sizeof(bias)); 
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
                            // output is row of a -> col of b
                            float value = 0.0;
                            for (size_t i = 0; i < in_features; ++i)
                            {
                                value += input_(n, c, h, i)*weights_(n, c, i, w); 
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
    // TODO.
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
                                    values.push_back(input_(n,c,kh,kw)); 
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
            float sum_exp = 0.0;
            for (size_t w = 0; w < input_.W; ++w)
            {
                sum_exp += exp(input_(0,0,0,w)); 
            } 
            for (size_t i = 0; i < input_.W; ++i)
            {
                output_(0,0,0,i) = exp(input_(0,0,0,i)) / sum_exp; 
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
.
        void add(Layer* layer) {
            // TODO
            // This works by trying to "push" all the layers into one container. 
            assert(layer != NULL); 
            NN.push_back(layer); 

        }

        void load(std::string file) {
            // TODO
            // only for layers where you implmement read_weights_bias() you need to take care of loading the weights and biases   
            std::ifstream is(path_.c_str(), std::ios::in | std::ios::binary);
            for (auto layer : NN)
            {
                layer->read_weights_bias(file); 
            }
        }

        Tensor predict(Tensor input) {
            // TODO
            //print the input digit along with the probabilities  
            //add layers using the method 
            add(Conv2d(1, 6, 5));
            add(ReLu());
            add(MaxPool2d(2)); 
            add(Conv2d(6, 16, 5));
            add(ReLu());
            add(MaxPool2d(2)); 
            add(Flatten());
            add(Linear(400, 120));
            add(ReLu());
            add(Linear(120, 84));
            add(ReLu());
            add(Linear(84, 10));
            add(ReLu()); 
            add(SoftMax()); 

            Tensor ouput = input;
            //printting the output of the layers. 
            for (auto layer : lenet)
            {
                layer->set_input(output);
                layer->fwd();
                output = layer->get_output();   
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
        std::vector<Layer*> NN;   
        // TODO: storage for layers
};

#endif // NETWORK_HPP
