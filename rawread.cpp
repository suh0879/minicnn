#include <iostream>
#include "tensor.hpp"
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdint>


bool is_little_endian() 
{
    int n = 1;
    return *(char *)&n == 1;
}

void readWeightsAndBiases(const std::string& filename) {

    size_t output_channel, input_channel, kernel_height, kernel_weight; 

        Tensor imgs_;

    std::ifstream file(filename, std::ios::in | std::ios::binary);
    file.read(reinterpret_cast<char *>(&output_channel), 4);
    file.read(reinterpret_cast<char *>(&input_channel), 4);
    file.read(reinterpret_cast<char *>(&kernel_height), 4);
    file.read(reinterpret_cast<char *>(&kernel_weight), 4);

    if (is_little_endian())
    {
        std::reverse(reinterpret_cast<char *>(&output_channel),reinterpret_cast<char *>(&output_channel) +sizeof(size_t)); 
        std::reverse(reinterpret_cast<char *>(&input_channel), reinterpret_cast<char *>(&input_channel) + sizeof(size_t));
        std::reverse(reinterpret_cast<char *>(&kernel_height), reinterpret_cast<char *>(&kernel_weight) + sizeof(size_t));  
        std::reverse(reinterpret_cast<char *>(&kernel_weight), reinterpret_cast<char *>(&kernel_weight) + sizeof(size_t)); 
    }
    Tensor weights_; 

    weights_ = Tensor(output_channel, input_channel, kernel_height, kernel_weight); 
        for (size_t n = 0; n < output_channel; ++n)
        {
            for (size_t c = 0; c < input_channel; ++c)
            {
                for (size_t h = 0; h < kernel_height; ++h)
                {
                    for (size_t w = 0; w < kernel_weight; ++w)
                    {
                        float val; 
                        file.read(reinterpret_cast<char *>(&val), sizeof(float)); 
                        weights_(n,c,h,w) = val; 
                        std::cout << weights_(n,c,h,w) << "/n"; 
                    }
                }
            }
        }    
}

int main() {
    // Replace "your_binary_file.dat" with the actual filename
    
    readWeightsAndBiases("/mnt/d/Etudes/Computer-Vision/assignment/lenet.raw");


    return 0;
}
