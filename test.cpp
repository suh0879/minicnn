#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "tensor.hpp"  
#include "network.hpp" 
#include <iostream> 
#include <fstream> 
#include <math.h>
#include <algorithm>

TEST_CASE("Conv2d Arithmetic", "[conv2d]")
{
    Tensor conv_weights = Tensor(1,1,2,2); 
    conv_weights(0, 0, 0, 0) = 0.5;  
    conv_weights(0, 0, 0, 1) = -0.5; 
    conv_weights(0, 0, 1, 0) = 0.3;  
    conv_weights(0, 0, 1, 1) = 0.1;  

    Tensor conv_bias = Tensor(1,1,1,1);
    conv_bias(0, 0, 0, 0) = 0.2;  

    Tensor input = Tensor(1,1,2,2); 
    input(0, 0, 0, 0) = 1.0; 
    input(0, 0, 0, 1) = 0.5; 
    input(0, 0, 1, 0) = 0.2; 
    input(0, 0, 1, 1) = -0.3;

    Conv2d test_conv(1,1,2); 
    test_conv.set_input(input);
    test_conv.set_weights(conv_weights);
    test_conv.set_bias(conv_bias); 
    test_conv.fwd();
    Tensor conv_output = test_conv.get_output();
    Tensor conv_exp_output = Tensor(1,1,1,1);
    conv_exp_output(0,0,0,0) = 1.0 * 0.5 + 0.5 * (-0.5) + 0.2 * 0.3 + (-0.3) * 0.1 + conv_bias(0,0,0,0); 
    REQUIRE(conv_output(0,0,0,0) == Approx(conv_exp_output(0,0,0,0)));
}

TEST_CASE("MaxPooling", "[maxpool2d]") 
{
    Tensor input = Tensor(1,1,2,2); 
    input(0, 0, 0, 0) = 1.0;
    input(0, 0, 0, 1) = 0.5;;
    input(0, 0, 1, 0) = 0.2;
    input(0, 0, 1, 1) = -0.3;   

    MaxPool2d test_pool(2,2); 
    test_pool.set_input(input); 
    test_pool.fwd(); 
    Tensor pool_exp_output = Tensor(1,1,1,1);
    std::vector<float> vals = {1.0, 0.5, 0.2, -0.3};
    pool_exp_output(0,0,0,0) = *(std::max_element(vals.begin(), vals.end()));  
    Tensor pool_output = test_pool.get_output();
    REQUIRE(pool_output(0,0,0,0) == Approx(pool_exp_output(0,0,0,0)));
}

TEST_CASE("Flatten", "[flatten]") 
{
    Tensor input = Tensor(1,1,2,2); 
    input(0, 0, 0, 0) = 1.0;
    input(0, 0, 0, 1) = 0.5;
    input(0, 0, 1, 0) = 0.2;
    input(0, 0, 1, 1) = -0.3;   

    Flatten test_flat; 
    test_flat.set_input(input); 
    test_flat.fwd();
    Tensor flat_output = test_flat.get_output();
    REQUIRE(flat_output.W == 4);
}

TEST_CASE("Linear", "[linear]") 
{
    Tensor input = Tensor(1,1,2,2); 
    input(0, 0, 0, 0) = 1.0;
    input(0, 0, 0, 1) = 0.5;
    input(0, 0, 1, 0) = 0.2;
    input(0, 0, 1, 1) = -0.3;   
    
    Linear test_linear(2,1); 
    Tensor linear_weights(1, 1, 2, 1);
    linear_weights(0, 0, 0, 0) = 0.4;
    linear_weights(0, 0, 1, 0) = -0.3;
    Tensor linear_bias(1, 1, 1, 1);
    linear_bias(0, 0, 0, 0) = 0.1;
    test_linear.set_input(input); 
    test_linear.set_bias(linear_bias); 
    test_linear.set_weights(linear_weights); 
    test_linear.fwd();
    Tensor linear_output = test_linear.get_output();
    Tensor linear_exp_output = Tensor(1, 1, 2, 1);
    linear_exp_output(0, 0, 0, 0) = 1.0 * 0.4 + 0.5 * (-0.3) + linear_bias(0,0,0,0);
    linear_exp_output(0, 0, 1, 0) = 0.2 * 0.4 + (-0.3) * (- 0.3) + linear_bias(0,0,0,0); 

    REQUIRE(linear_output(0,0,0,0) == Approx(linear_exp_output(0,0,0,0)));
    REQUIRE(linear_output(0,0,1,0) == Approx(linear_exp_output(0,0,1,0)));
}

TEST_CASE("ReLu", "[relu]") 
{
    Tensor input = Tensor(1,1,2,2); 
    input(0, 0, 0, 0) = 1.0;
    input(0, 0, 0, 1) = 0.5;
    input(0, 0, 1, 0) = 0.2;
    input(0, 0, 1, 1) = -0.3;   

    ReLu relu_test;
    relu_test.set_input(input); 
    relu_test.fwd();

    REQUIRE(relu_test.get_output()(0, 0, 1, 1) == 0.0f); 
}

TEST_CASE("SoftMax", "[softmax]") 
{
    Tensor input = Tensor(1,1,2,2); 
    input(0, 0, 0, 0) = 1.0;
    input(0, 0, 0, 1) = 0.5;
    input(0, 0, 1, 0) = 0.2;
    input(0, 0, 1, 1) = -0.3;

    Flatten flat_in; 
    flat_in.set_input(input); 
    flat_in.fwd();
    Tensor flat = flat_in.get_output(); 
    SoftMax soft_test; 
    soft_test.set_input(flat); 
    soft_test.fwd();
    Tensor exp_soft_out(1,1,1,4);
    float sum = 0.0f; 
    for (size_t i = 0; i < flat.W; ++i)
    {
        sum += exp(flat(0,0,0,i));
    }
     
    for (size_t j = 0; j < exp_soft_out.W; ++j)
    {
        exp_soft_out(0,0,0,j) = exp(flat_in.get_output()(0,0,0,j)) / sum; 
    }
    for (size_t k = 0; k < 4; ++k)
    {
        REQUIRE(soft_test.get_output()(0,0,0,k) == exp_soft_out(0,0,0,k));
    }
}











