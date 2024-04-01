#ifndef TEST_CONV2D_H
#define TEST_CONV2D_H

#include "../src/ffml/ffml.h"
#include "common.h"
#include "../src/engine/conv/ConvMLP.h"

void test_conv2d() {
    test_name("Testing Conv2D");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    std::vector<std::vector<int>> input_data = {
        {1, 2, 3, 4, 5, 6, 7, 8},
        {2, 3, 4, 5, 6, 7, 8, 9},
        {3, 4, 5, 6, 7, 8, 9, 10},
        {4, 5, 6, 7, 8, 9, 10, 11},
        {5, 6, 7, 8, 9, 10, 11, 12},
        {6, 7, 8, 9, 10, 11, 12, 13},
        {7, 8, 9, 10, 11, 12, 13, 14},
        {8, 9, 10, 11, 12, 13, 14, 15}
    };

    ffml_tensor * inputs = ffml_tensor_create(3, {1,8,8,0}, "inputs");

    auto convmlp = new ConvMLP(pool, 1, 8, 8, {
        {3, 3, 2}, // # of filters, kernel side, pool side
        {9, 3, 2}, // # of filters, kernel side, pool side
    });

    auto out = convmlp->call(inputs);

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(out);

    ffml_cgraph_alloc(cgraph, pool, true);

    // maybe load weights

    // configure data loader

    // configure optimizer

    // training loop:
    // for (int i = 0; i < N_EPOCHS; i++) {

    // sample fake data
    for (uint64_t i = 0; i < 8; i++) {
        for (uint64_t j = 0; j < 8; j++) {
            ffml_set_data(inputs, {0,i,j,0}, input_data[i][j] * (i+1));
        }
    }

    // set filter manually
    auto convlayer0_filter = ffml_get_tensor_by_name(cgraph, "convlayer0_filter");
    for(uint64_t i = 0; i < 3; i++) {
        for(uint64_t j = 0; j < 3; j++) {
            for(uint64_t in_chan = 0; in_chan < 1; in_chan++) {
                for (uint64_t out_chan = 0; out_chan < 3; out_chan++) {
                    ffml_set_data(convlayer0_filter, {out_chan,in_chan,i,j}, (i+j) / 2.0f + 0.1f * out_chan);
                }
            }
        }
    }
    auto convlayer1_filter = ffml_get_tensor_by_name(cgraph, "convlayer1_filter");
    for(uint64_t i = 0; i < 3; i++) {
        for(uint64_t j = 0; j < 3; j++) {
            for(uint64_t in_chan = 0; in_chan < 3; in_chan++) {
                for (uint64_t out_chan = 0; out_chan < 9; out_chan++) {
                    ffml_set_data(convlayer1_filter, {out_chan,in_chan,i,j}, (i+j) / 2.0f + 0.1f * out_chan);
                }
            }
        }
    }
    convlayer0_filter->init_ran = true;
    convlayer1_filter->init_ran = true;

    //     forward pass
    ffml_cgraph_forward(cgraph);

    ffml_zerograd(cgraph);

    //     backward pass
    ffml_cgraph_backward(cgraph);

    auto convlayer0_conv = ffml_get_tensor_by_name(cgraph, "convlayer0_conv");
    test_tensor_data_flat_almost_equal(convlayer0_conv, {81.0f, 102.0f, 123.0f, 144.0f, 165.0f, 186.0f, 144.0f, 174.0f, 204.0f, 234.0f, 264.0f, 294.0f, 225.0f  }); // ...

    auto convlayer0_pool = ffml_get_tensor_by_name(cgraph, "maxpool0_pooled");
    test_tensor_data_flat_almost_equal(convlayer0_pool, {174.0f, 234.0f, 294.0f, 372.0f, 468.0f, 564.0f, 642.0f, 774.0f, 906.0f, 188.1f, 253.5f, 318.9f, 404.1f, 509.1f, 614.1f  }); //...

    // ffml_debug_print_cgraph_data(cgraph);

// 
    // ffml_debug_print_cgraph_data(cgraph);

    //     update weights (optimizer)

    //     evaluate

    // save model

    // print hello world:
    // printf("Hello, World!\n");

    // test_equal(ffml_get_data(a, {0,0,0,0}), 2.0f);
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