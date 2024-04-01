#ifndef TEST_KARPATHY_3_H
#define TEST_KARPATHY_3_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_karpathy_3() {
    test_name("Karpathy Test 3");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    uint64_t* onedim = new uint64_t[1];
    onedim[0] = 1;

    ffml_tensor * a = ffml_tensor_create(1, onedim, "a");
    ffml_tensor * b = ffml_tensor_create(1, onedim, "b");
    
    ffml_tensor * d = ffml_op(FFML_OP_MUL, a, b);
    ffml_set_name(d, "d");

    ffml_tensor * e = ffml_op(FFML_OP_ADD, a, b);
    ffml_set_name(e, "e");

    ffml_tensor * f = ffml_op(FFML_OP_MUL, d, e);
    ffml_set_name(f, "f");

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(f);

    ffml_cgraph_alloc(cgraph, pool);

    // maybe load weights

    // configure data loader

    // configure optimizer

    // training loop:
    // for (int i = 0; i < N_EPOCHS; i++) {

    // sample fake data
    ffml_set_data(a, {0,0,0,0}, -2.0f);
    ffml_set_data(b, {0,0,0,0}, 3.0f);
    
    //     forward pass
    ffml_cgraph_forward(cgraph);

    // ffml_debug_print_cgraph_data(cgraph);

    ffml_zerograd(cgraph);

    //     backward pass
    ffml_cgraph_backward(cgraph);
// 
    // ffml_debug_print_cgraph_data(cgraph);

    //     update weights (optimizer)

    //     evaluate

    // save model

    // print hello world:
    // printf("Hello, World!\n");

    test_almost_equal(ffml_get_data(a, {0,0,0,0}), -2.0f);
    test_almost_equal(ffml_get_data(b, {0,0,0,0}), 3.0f);
    
    test_almost_equal(ffml_get_data(d, {0,0,0,0}), -6.0f);
    test_almost_equal(ffml_get_data(e, {0,0,0,0}), 1.0f);
    test_almost_equal(ffml_get_data(f, {0,0,0,0}), -6.0f);

    test_almost_equal(ffml_get_grad(f, {0,0,0,0}), 1.0f);
    test_almost_equal(ffml_get_grad(e, {0,0,0,0}), -6.0f);
    test_almost_equal(ffml_get_grad(d, {0,0,0,0}), 1.0f);

    test_almost_equal(ffml_get_grad(a, {0,0,0,0}), -3.0f);
    test_almost_equal(ffml_get_grad(b, {0,0,0,0}), -8.0f);

    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too
}

#endif