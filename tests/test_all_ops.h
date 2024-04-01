#ifndef TEST_ALL_OPS_H
#define TEST_ALL_OPS_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_all_ops() {
    test_name("Test all ops");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    uint64_t* onedim = new uint64_t[1];
    onedim[0] = 1;

    ffml_tensor * a = ffml_tensor_create(1, onedim, "a");
    ffml_tensor * b = ffml_tensor_create(1, onedim, "b");
    ffml_tensor * c = ffml_tensor_create(1, onedim, "c");
    ffml_tensor * d = ffml_tensor_create(1, onedim, "d");
    ffml_tensor * e = ffml_tensor_create(1, onedim, "e");
    ffml_tensor * f = ffml_tensor_create(1, onedim, "f");
    ffml_tensor * two = ffml_tensor_create(1, onedim, "two");
    
    ffml_tensor * mul = ffml_op(FFML_OP_MUL, a, b);
    ffml_set_name(mul, "mul");

    ffml_tensor * add = ffml_op(FFML_OP_ADD, mul, c);
    ffml_set_name(add, "add");

    ffml_tensor * sub = ffml_op(FFML_OP_SUB, add, d);
    ffml_set_name(sub, "sub");

    ffml_tensor * div = ffml_op(FFML_OP_DIV, sub, e);
    ffml_set_name(div, "div");

    ffml_tensor * add2 = ffml_op(FFML_OP_ADD, div, f);
    ffml_set_name(add2, "add2");

    ffml_tensor * pow = ffml_op(FFML_OP_POW, add2, two);
    ffml_set_name(pow, "pow");

    ffml_tensor * neg = ffml_unary_op(FFML_OP_NEG, pow);
    ffml_set_name(neg, "neg");

    ffml_tensor * tanh = ffml_unary_op(FFML_OP_TANH, neg);
    ffml_set_name(tanh, "tanh");

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(tanh);

    ffml_cgraph_alloc(cgraph, pool);

    // maybe load weights

    // configure data loader

    // configure optimizer

    // training loop:
    // for (int i = 0; i < N_EPOCHS; i++) {

    // sample fake data
    ffml_set_data(a, {0,0,0,0}, -2.0f);
    ffml_set_data(b, {0,0,0,0}, 3.0f);
    ffml_set_data(c, {0,0,0,0}, 10.0f);
    ffml_set_data(d, {0,0,0,0}, 6.0f);
    ffml_set_data(e, {0,0,0,0}, 2.0f);
    ffml_set_data(f, {0,0,0,0}, 0.5f);
    ffml_set_data(two, {0,0,0,0}, 2.0f);
    
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
    test_almost_equal(ffml_get_data(mul, {0,0,0,0}), -6.0f);

    test_almost_equal(ffml_get_data(c, {0,0,0,0}), 10.0f);
    test_almost_equal(ffml_get_data(add, {0,0,0,0}), 4.0f);

    test_almost_equal(ffml_get_data(d, {0,0,0,0}), 6.0f);
    test_almost_equal(ffml_get_data(sub, {0,0,0,0}), -2.0f);

    test_almost_equal(ffml_get_data(e, {0,0,0,0}), 2.0f);
    test_almost_equal(ffml_get_data(div, {0,0,0,0}), -1.0f);

    test_almost_equal(ffml_get_data(f, {0,0,0,0}), 0.5f);
    test_almost_equal(ffml_get_data(add2, {0,0,0,0}), -0.5f);

    test_almost_equal(ffml_get_data(two, {0,0,0,0}), 2.0f);
    test_almost_equal(ffml_get_data(pow, {0,0,0,0}), 0.25f);

    test_almost_equal(ffml_get_data(neg, {0,0,0,0}), -0.25f);

    test_almost_equal(ffml_get_data(tanh, {0,0,0,0}), -0.2449186624037091292778f);

    //------

    test_equal(ffml_get_grad(tanh, {0,0,0,0}), 1.0f);

    test_almost_equal_grad(neg, {0,0,0,0}, 1.0f - std::pow(std::tanh( ffml_get_data(neg, {0,0,0,0}) ),2)); // derivative of tanh is 1 - tanh^2
    test_almost_equal_grad(neg, {0,0,0,0}, 0.940014849f); // derivative of tanh is 1 - tanh^2

    test_almost_equal_grad(pow, {0,0,0,0}, -0.940014849f); // grad of neg is simply -1 * previous grad

    test_almost_equal_grad(add2, {0,0,0,0}, 2.0f * std::pow( ffml_get_data(add2, {0,0,0,0}), 2.0f - 1.0f) * ffml_get_grad(pow, {0,0,0,0}) ); // grad of pow is n * x^(n-1) * previous grad
    test_almost_equal_grad(add2, {0,0,0,0}, 0.940014849f);

    test_almost_equal_grad(div, {0,0,0,0}, 1.0f * ffml_get_grad(add2, {0,0,0,0})); // grad of add2 is simply 1 * previous grad
    test_almost_equal_grad(div, {0,0,0,0}, 0.940014849f);
    test_almost_equal_grad(f, {0,0,0,0}, 0.940014849f);

    // If you're referring to basic division of two variables, say f(x, y) = x / y, then:
    //     The partial derivative with respect to x is 1/y,
    //     The partial derivative with respect to y is -x / y^2.
    test_almost_equal_grad(sub, {0,0,0,0}, (1.0f / ffml_get_data(e, {0,0,0,0})) * ffml_get_grad(div, {0,0,0,0}));
    test_almost_equal_grad(sub, {0,0,0,0}, 0.470007f);
    test_almost_equal_grad(e, {0,0,0,0}, (-1.0f * ffml_get_data(sub, {0,0,0,0})) / std::pow(ffml_get_data(e, {0,0,0,0}), 2.0f) * ffml_get_grad(div, {0,0,0,0}));
    test_almost_equal_grad(e, {0,0,0,0}, 0.470007f);

    test_almost_equal_grad(add, {0,0,0,0}, 1.0f * ffml_get_grad(sub, {0,0,0,0}));
    test_almost_equal_grad(add, {0,0,0,0}, 0.470007f);

    test_almost_equal_grad(c, {0,0,0,0}, 1.0f * ffml_get_grad(add, {0,0,0,0}));
    test_almost_equal_grad(c, {0,0,0,0}, 0.470007f);

    test_almost_equal_grad(mul, {0,0,0,0}, 1.0f * ffml_get_grad(add, {0,0,0,0}));
    test_almost_equal_grad(mul, {0,0,0,0}, 0.470007f);

    // grad of multiplication is the other variable
    test_almost_equal_grad(b, {0,0,0,0}, ffml_get_data(a, {0,0,0,0}) * ffml_get_grad(mul, {0,0,0,0}));
    test_almost_equal_grad(b, {0,0,0,0}, -0.940015);

    test_almost_equal_grad(a, {0,0,0,0}, ffml_get_data(b, {0,0,0,0}) * ffml_get_grad(mul, {0,0,0,0}));
    test_almost_equal_grad(a, {0,0,0,0}, 1.410022);
    
    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too
}

#endif