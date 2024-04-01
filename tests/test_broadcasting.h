#ifndef TEST_BROADCASTING_H
#define TEST_BROADCASTING_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_broadcasting() {
    test_name("Test broadcasting");
    // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    ffml_tensor * w_2d = ffml_tensor_create(2, {1,5,0,0}, "w");
    ffml_tensor * b = ffml_tensor_create(1, {1,0,0,0}, "b");

    ffml_tensor * add = ffml_op(FFML_OP_ADD, w_2d, b);
    ffml_set_name(add, "add");

    ffml_tensor * mul = ffml_op(FFML_OP_MUL, w_2d, b);
    ffml_set_name(mul, "mul");

    ffml_tensor * sub = ffml_op(FFML_OP_SUB, add, mul);
    ffml_set_name(sub, "sub");

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(sub);

    ffml_cgraph_alloc(cgraph, pool);

    // sample fake data
    ffml_set_data(w_2d, {0,0,0,0}, 1.0f);
    ffml_set_data(w_2d, {0,1,0,0}, 2.0f);
    ffml_set_data(w_2d, {0,2,0,0}, 3.0f);
    ffml_set_data(w_2d, {0,3,0,0}, 4.0f);
    ffml_set_data(w_2d, {0,4,0,0}, 5.0f);

    ffml_set_data(b, {0,0,0,0}, 10.0f);
    
    //     forward pass
    ffml_cgraph_forward(cgraph);

    // ffml_debug_print_cgraph_data(cgraph);

    ffml_zerograd(cgraph);

    //     backward pass
    ffml_cgraph_backward(cgraph);
// 
    // ffml_debug_print_cgraph_data(cgraph);

    test_almost_equal(ffml_get_data(w_2d, {0,0,0,0}), 1.0f);
    test_almost_equal(ffml_get_data(w_2d, {0,1,0,0}), 2.0f);
    test_almost_equal(ffml_get_data(w_2d, {0,2,0,0}), 3.0f);
    test_almost_equal(ffml_get_data(w_2d, {0,3,0,0}), 4.0f);
    test_almost_equal(ffml_get_data(w_2d, {0,4,0,0}), 5.0f);

    test_almost_equal(ffml_get_data(b, {0,0,0,0}), 10.0f);

    test_almost_equal(ffml_get_data(add, {0,0,0,0}), 11.0f);
    test_almost_equal(ffml_get_data(add, {0,1,0,0}), 12.0f);
    test_almost_equal(ffml_get_data(add, {0,2,0,0}), 13.0f);
    test_almost_equal(ffml_get_data(add, {0,3,0,0}), 14.0f);
    test_almost_equal(ffml_get_data(add, {0,4,0,0}), 15.0f);

    test_almost_equal(ffml_get_data(mul, {0,0,0,0}), 10.0f);
    test_almost_equal(ffml_get_data(mul, {0,1,0,0}), 20.0f);
    test_almost_equal(ffml_get_data(mul, {0,2,0,0}), 30.0f);
    test_almost_equal(ffml_get_data(mul, {0,3,0,0}), 40.0f);
    test_almost_equal(ffml_get_data(mul, {0,4,0,0}), 50.0f);

    test_almost_equal(ffml_get_data(sub, {0,0,0,0}), 1.0f);
    test_almost_equal(ffml_get_data(sub, {0,1,0,0}), -8.0f);
    test_almost_equal(ffml_get_data(sub, {0,2,0,0}), -17.0f);
    test_almost_equal(ffml_get_data(sub, {0,3,0,0}), -26.0f);
    test_almost_equal(ffml_get_data(sub, {0,4,0,0}), -35.0f);

    test_almost_equal_grad(sub, {0,0,0,0}, 1.0f);
    test_almost_equal_grad(sub, {0,1,0,0}, 1.0f);
    test_almost_equal_grad(sub, {0,2,0,0}, 1.0f);
    test_almost_equal_grad(sub, {0,3,0,0}, 1.0f);
    test_almost_equal_grad(sub, {0,4,0,0}, 1.0f);

    test_almost_equal_grad(add, {0,0,0,0}, 1.0f);
    test_almost_equal_grad(add, {0,1,0,0}, 1.0f);
    test_almost_equal_grad(add, {0,2,0,0}, 1.0f);
    test_almost_equal_grad(add, {0,3,0,0}, 1.0f);
    test_almost_equal_grad(add, {0,4,0,0}, 1.0f);

    test_almost_equal_grad(mul, {0,0,0,0}, -1.0f);
    test_almost_equal_grad(mul, {0,1,0,0}, -1.0f);
    test_almost_equal_grad(mul, {0,2,0,0}, -1.0f);
    test_almost_equal_grad(mul, {0,3,0,0}, -1.0f);
    test_almost_equal_grad(mul, {0,4,0,0}, -1.0f);

    // verify gradients
    test_almost_equal_grad(sub, {0,0,0,0}, 1.0f);
    test_almost_equal_grad(sub, {0,1,0,0}, 1.0f);
    test_almost_equal_grad(sub, {0,2,0,0}, 1.0f);
    test_almost_equal_grad(sub, {0,3,0,0}, 1.0f);
    test_almost_equal_grad(sub, {0,4,0,0}, 1.0f);

    test_almost_equal_grad(mul, {0,0,0,0}, -1.0f);
    test_almost_equal_grad(mul, {0,1,0,0}, -1.0f);
    // gave up here, we have test_broadcasting2


    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too
}

#endif