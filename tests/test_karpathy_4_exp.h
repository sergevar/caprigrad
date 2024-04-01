#ifndef TEST_KARPATHY_4_H
#define TEST_KARPATHY_4_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_karpathy_4_exp() {
    test_name("Karpathy Test 4 (exp)");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    uint64_t* onedim = new uint64_t[1];
    onedim[0] = 1;

    ffml_tensor * x1 = ffml_tensor_create(1, onedim, "x1");
    ffml_tensor * x2 = ffml_tensor_create(1, onedim, "x2");
    ffml_tensor * w1 = ffml_tensor_create(1, onedim, "w1");
    ffml_tensor * w2 = ffml_tensor_create(1, onedim, "w2");

    ffml_tensor * b = ffml_tensor_create(1, onedim, "b");

    ffml_tensor * x1w1 = ffml_op(FFML_OP_MUL, x1, w1);
    ffml_set_name(x1w1, "x1w1");

    ffml_tensor * x2w2 = ffml_op(FFML_OP_MUL, x2, w2);
    ffml_set_name(x2w2, "x2w2");

    ffml_tensor * x1w1x2w2 = ffml_op(FFML_OP_ADD, x1w1, x2w2);
    ffml_set_name(x1w1x2w2, "x1w1 + x2w2");

    ffml_tensor * n = ffml_op(FFML_OP_ADD, x1w1x2w2, b);
    ffml_set_name(n, "n");

    ffml_tensor * one = ffml_tensor_create(1, onedim, "one");
    ffml_tensor * two = ffml_tensor_create(1, onedim, "two");

    ffml_tensor * twon = ffml_op(FFML_OP_MUL, two, n);
    ffml_set_name(twon, "twon");

    ffml_tensor * twon_exp = ffml_unary_op(FFML_OP_EXP, twon);
    ffml_set_name(twon_exp, "twon_exp");

    ffml_tensor * e_minus_one = ffml_op(FFML_OP_SUB, twon_exp, one);
    ffml_set_name(e_minus_one, "e_minus_one");

    ffml_tensor * e_plus_one = ffml_op(FFML_OP_ADD, twon_exp, one);
    ffml_set_name(e_plus_one, "e_plus_one");

    ffml_tensor * minus_one = ffml_tensor_create(1, onedim, "minus_one");

    ffml_tensor * one_div_b_but_via_pow = ffml_op(FFML_OP_POW, e_plus_one, minus_one);

    ffml_tensor * o = ffml_op(FFML_OP_MUL, e_minus_one, one_div_b_but_via_pow);
    ffml_set_name(o, "o");
    
    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(o);

    ffml_cgraph_alloc(cgraph, pool);

    // maybe load weights

    // configure data loader

    // configure optimizer

    // training loop:
    // for (int i = 0; i < N_EPOCHS; i++) {

    // sample fake data
    ffml_set_data(one, {0,0,0,0}, 1.0f);
    ffml_set_data(minus_one, {0,0,0,0}, -1.0f);
    ffml_set_data(two, {0,0,0,0}, 2.0f);
    ffml_set_data(x1, {0,0,0,0}, 2.0f);
    ffml_set_data(x2, {0,0,0,0}, 0.0f);
    ffml_set_data(w1, {0,0,0,0}, -3.0f);
    ffml_set_data(w2, {0,0,0,0}, 1.0f);
    ffml_set_data(b, {0,0,0,0}, 6.88137357870194321f);
    
    //     forward pass
    ffml_cgraph_forward(cgraph);

    // ffml_debug_print_cgraph_data(cgraph);

    ffml_zerograd(cgraph);

    //     backward pass
    ffml_cgraph_backward(cgraph);
// 
    ffml_debug_print_cgraph_data(cgraph);

    //     update weights (optimizer)

    //     evaluate

    // save model

    // print hello world:
    // printf("Hello, World!\n");

    test_almost_equal(ffml_get_data(x1, {0,0,0,0}), 2.0f);
    test_almost_equal(ffml_get_data(x2, {0,0,0,0}), 0.0f);
    test_almost_equal(ffml_get_data(w1, {0,0,0,0}), -3.0f);
    test_almost_equal(ffml_get_data(w2, {0,0,0,0}), 1.0f);
    test_almost_equal(ffml_get_data(b, {0,0,0,0}), 6.88137357870194321f);

    test_almost_equal(ffml_get_data(x1w1, {0,0,0,0}), -6.0f);
    test_almost_equal(ffml_get_data(x2w2, {0,0,0,0}), 0.0f);
    test_almost_equal(ffml_get_data(x1w1x2w2, {0,0,0,0}), -6.0f);
    test_almost_equal(ffml_get_data(n, {0,0,0,0}), 0.8814f);
    test_almost_equal(ffml_get_data(o, {0,0,0,0}), 0.7071f);


    test_almost_equal(ffml_get_grad(o, {0,0,0,0}), 1.0f);
    test_almost_equal(ffml_get_grad(n, {0,0,0,0}), 0.5f);
    test_almost_equal(ffml_get_grad(b, {0,0,0,0}), 0.5f);
    test_almost_equal(ffml_get_grad(x1w1x2w2, {0,0,0,0}), 0.5f);

    test_almost_equal(ffml_get_grad(x1w1, {0,0,0,0}), 0.5f);
    test_almost_equal(ffml_get_grad(x2w2, {0,0,0,0}), 0.5f);

    test_almost_equal(ffml_get_grad(x1, {0,0,0,0}), -1.5f);
    test_almost_equal(ffml_get_grad(x2, {0,0,0,0}), 0.5f);
    test_almost_equal(ffml_get_grad(w1, {0,0,0,0}), 1.0f);
    test_almost_equal(ffml_get_grad(w2, {0,0,0,0}), 0.0f);

    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too
}

#endif