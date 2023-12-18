#include <iostream> 
//#include "mnist.hpp"
//#include "network.hpp"
#include "tensor.hpp"
using namespace std;

void tensor_val(Tensor test1); 

int main()
{   
    //Tensor test1(1, 1, 4, 4);
    //tensor_val(test1); 

    Tensor myTensor(3, 2, 4, 4);

    // Fill the tensor with sequential values for better visualization
    for (size_t n = 0; n < myTensor.N; ++n) {
        for (size_t c = 0; c < myTensor.C; ++c) {
            for (size_t h = 0; h < myTensor.H; ++h) {
                for (size_t w = 0; w < myTensor.W; ++w) {
                    myTensor(n, c, h, w) = n * 100 + c * 10 + h * 2 + w;
                }
            }
        }
    }

    // Print the original tensor
    cout << "Original Tensor:\n" << myTensor << "\n\n";

    // Create a slice starting from index 1 with 2 elements along the N dimension
    Tensor slicedTensor = myTensor.slice(1, 2);

    // Print the sliced tensor
    cout << "Sliced Tensor:\n" << slicedTensor << "\n\n";

    // Access and print individual elements of the sliced tensor
    for (size_t n = 0; n < slicedTensor.N; ++n) {
        for (size_t c = 0; c < slicedTensor.C; ++c) {
            for (size_t h = 0; h < slicedTensor.H; ++h) {
                for (size_t w = 0; w < slicedTensor.W; ++w) {
                    cout << "Element (" << n << "," << c << "," << h << "," << w << "): ";
                    cout << slicedTensor(n, c, h, w) << "\n";
                }
            }
        }
    }
    
    return 0; 
}

void tensor_val(Tensor test1)
{ 
    for (size_t n = 0; n < test1.N; ++n)
    {
        for (size_t c=0; c< test1.C; ++c)
        {
            for (size_t h=0; h< test1.H; ++h)
            {
                for (size_t w=0; w < test1.W; ++w)
                {
                    test1.operator()(n,c,h,w) = 1.0;
                    cout << test1.operator()(n,c,h,w);  
                }
            cout << "\n";
            } 
        }
    }
}