#ifndef TEST_UNEVEN_POOLING_H
#define TEST_UNEVEN_POOLING_H

#include "../src/ffml/ffml.h"
#include "common.h"
#include "../src/engine/conv/ConvMLP.h"

void test_uneven_pooling() {
    test_name("Testing MaxPool2D when uneven dimensions");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    std::vector<std::vector<int>> input_data = {
        {-1, -2, -3, -4, -5, -6, -7},
        {-2, -3, -4, -5, -6, -7, -8},
        {-3, -4, -5, -6, -7, -8, -9},
        {-4, -5, -6, -7, -8, -9, -10},
        {-5, -6, -7, -8, -9, -10, -11},
        {-6, -7, -8, -9, -10, -11, -12},
        {-7, -8, -9, -10, -11, -12, -13},
        {-8, -9, -10, -11, -12, -13, -14}
    };

    ffml_tensor * inputs = ffml_tensor_create(3, {2,7,7,0}, "inputs");

    ffml_tensor * pooled = ffml_unary_op(FFML_OP_MAXPOOL2D, inputs);
    ffml_set_name(pooled, "pooled");

    ffml_tensor * two = ffml_tensor_create(1, {1,0,0,0}, "two");

    ffml_tensor * doubled = ffml_op(FFML_OP_MUL, pooled, two);
    ffml_set_name(doubled, "doubled");
    
    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(doubled);

    ffml_cgraph_alloc(cgraph, pool);

    ffml_set_data_flat(two, 0, 2.0f);

    // maybe load weights

    // configure data loader

    // configure optimizer

    // training loop:
    // for (int i = 0; i < N_EPOCHS; i++) {

    // sample fake data
    for (uint64_t i = 0; i < 7; i++) {
        for (uint64_t j = 0; j < 7; j++) {
            ffml_set_data(inputs, {0,i,j,0}, input_data[i][j]);
            ffml_set_data(inputs, {1,i,j,0}, input_data[i][j] * -10);
        }
    }

    //     forward pass
    ffml_cgraph_forward(cgraph);

    ffml_zerograd(cgraph);

    //     backward pass
    ffml_cgraph_backward(cgraph);
// 
    // ffml_debug_print_cgraph_data(cgraph);

    // ffml_debug_print_cgraph_data(cgraph);

    //     update weights (optimizer)

    //     evaluate

    // save model

    // print hello world:
    // printf("Hello, World!\n");

    test_equal(ffml_get_data(pooled, {0,0,0,0}), -1.0f);
    test_equal(ffml_get_data(pooled, {0,0,1,0}), -3.0f);
    test_equal(ffml_get_data(pooled, {0,0,2,0}), -5.0f);
    test_equal(ffml_get_data(pooled, {0,0,3,0}), -7.0f);
    test_equal(ffml_get_data(pooled, {0,1,0,0}), -3.0f);
    test_equal(ffml_get_data(pooled, {0,1,1,0}), -5.0f);
    test_equal(ffml_get_data(pooled, {0,1,2,0}), -7.0f);
    test_equal(ffml_get_data(pooled, {0,1,3,0}), -9.0f);
    test_equal(ffml_get_data(pooled, {0,2,0,0}), -5.0f);
    test_equal(ffml_get_data(pooled, {0,2,1,0}), -7.0f);
    test_equal(ffml_get_data(pooled, {0,2,2,0}), -9.0f);
    test_equal(ffml_get_data(pooled, {0,2,3,0}), -11.0f);
    test_equal(ffml_get_data(pooled, {0,3,0,0}), -7.0f);
    test_equal(ffml_get_data(pooled, {0,3,1,0}), -9.0f);
    test_equal(ffml_get_data(pooled, {0,3,2,0}), -11.0f);
    test_equal(ffml_get_data(pooled, {0,3,3,0}), -13.0f);

    test_equal(ffml_get_data(pooled, {1,0,0,0}), 30.0f);
    test_equal(ffml_get_data(pooled, {1,0,1,0}), 50.0f);
    test_equal(ffml_get_data(pooled, {1,0,2,0}), 70.0f);
    test_equal(ffml_get_data(pooled, {1,0,3,0}), 80.0f);
    test_equal(ffml_get_data(pooled, {1,1,0,0}), 50.0f);
    test_equal(ffml_get_data(pooled, {1,1,1,0}), 70.0f);
    test_equal(ffml_get_data(pooled, {1,1,2,0}), 90.0f);
    test_equal(ffml_get_data(pooled, {1,1,3,0}), 100.0f);
    test_equal(ffml_get_data(pooled, {1,2,0,0}), 70.0f);
    test_equal(ffml_get_data(pooled, {1,2,1,0}), 90.0f);
    test_equal(ffml_get_data(pooled, {1,2,2,0}), 110.0f);
    test_equal(ffml_get_data(pooled, {1,2,3,0}), 120.0f);
    test_equal(ffml_get_data(pooled, {1,3,0,0}), 80.0f);
    test_equal(ffml_get_data(pooled, {1,3,1,0}), 100.0f);
    test_equal(ffml_get_data(pooled, {1,3,2,0}), 120.0f);
    test_equal(ffml_get_data(pooled, {1,3,3,0}), 130.0f);

    test_equal(ffml_get_data(doubled, {0,0,0,0}), -2.0f);
    test_equal(ffml_get_data(doubled, {0,0,1,0}), -6.0f);

    for(int i=0; i<pooled->nelem; i++) {
        test_equal(ffml_get_grad_flat(doubled, i), 1.0f);
        test_equal(ffml_get_grad_flat(pooled, i), 2.0f);
    }
    test_equal(ffml_get_grad_flat(two, 0), 1248.0f);

    test_equal(ffml_get_grad(inputs, {0,0,0,0}), 2.0f);
    test_equal(ffml_get_grad(inputs, {0,0,1,0}), 0.0f);
    test_equal(ffml_get_grad(inputs, {0,0,2,0}), 2.0f);
    test_equal(ffml_get_grad(inputs, {0,0,3,0}), 0.0f);

    test_equal(ffml_get_grad(inputs, {0,1,0,0}), 0.0f);
    test_equal(ffml_get_grad(inputs, {0,1,1,0}), 0.0f);
    test_equal(ffml_get_grad(inputs, {0,1,2,0}), 0.0f);
    test_equal(ffml_get_grad(inputs, {0,1,3,0}), 0.0f);

    // test_equal(ffml_get_data(b, {0,0,0,0}), -3.0f);
    // test_equal(ffml_get_data(c, {0,0,0,0}), 10.0f);
    // test_equal(ffml_get_data(d, {0,0,0,0}), 4.0f);
    // test_equal(ffml_get_data(e, {0,0,0,0}), -6.0f);
    // test_equal(ffml_get_data(f, {0,0,0,0}), -2.0f);
    // test_equal(ffml_get_data(L, {0,0,0,0}), -8.0f);

    // test_equal(ffml_get_grad(L, {0,0,0,0}), 1.0f);
    // test_equal(ffml_get_grad(f, {0,0,0,0}), 4.0f);
    // test_equal(ffml_get_grad(d, {0,0,0,0}), -2.0f);
    // test_equal(ffml_get_grad(e, {0,0,0,0}), -2.0f);
    // test_equal(ffml_get_grad(c, {0,0,0,0}), -2.0f);
    // test_equal(ffml_get_grad(b, {0,0,0,0}), -4.0f);
    // test_equal(ffml_get_grad(a, {0,0,0,0}), 6.0f);

    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too
}

#endif