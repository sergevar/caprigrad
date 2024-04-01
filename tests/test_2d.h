#ifndef TEST_MULTIDIM_H
#define TEST_MULTIDIM_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_2d() {
    test_name("Test 2d");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    ffml_tensor * a_2d = ffml_tensor_create(2, {3,4,0,0}, "a_2d");
    ffml_tensor * b_2d = ffml_tensor_create(2, {3,4,0,0}, "b_2d");

    ffml_tensor * add_2d = ffml_op(FFML_OP_ADD, a_2d, b_2d);
    ffml_set_name(add_2d, "add_2d");

    ffml_tensor * c_2d = ffml_tensor_create(2, {3,4,0,0}, "c_2d");

    ffml_tensor * mul_2d = ffml_op(FFML_OP_MUL, add_2d, c_2d);
    ffml_set_name(mul_2d, "mul_2d");


    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(mul_2d);

    ffml_cgraph_alloc(cgraph, pool);

    // maybe load weights

    // configure data loader

    // configure optimizer

    // training loop:
    // for (int i = 0; i < N_EPOCHS; i++) {

    test_equal<float>(a_2d->size_bytes, 12 * sizeof(float));
    test_equal<float>(b_2d->size_bytes, 12 * sizeof(float));
    test_equal<float>(add_2d->size_bytes, 12 * sizeof(float));
    test_equal<float>(c_2d->size_bytes, 12 * sizeof(float));
    test_equal<float>(mul_2d->size_bytes, 12 * sizeof(float));

    test_equal<float>(ffml_offset_bytes(a_2d, {0,0,0,0}), 0 * sizeof(float));
    test_equal<float>(ffml_offset_bytes(a_2d, {0,1,0,0}), 1 * sizeof(float));
    test_equal<float>(ffml_offset_bytes(a_2d, {0,2,0,0}), 2 * sizeof(float));
    test_equal<float>(ffml_offset_bytes(a_2d, {0,3,0,0}), 3 * sizeof(float));
    test_equal<float>(ffml_offset_bytes(a_2d, {1,0,0,0}), 4 * sizeof(float));
    test_equal<float>(ffml_offset_bytes(a_2d, {1,1,0,0}), 5 * sizeof(float));
    test_equal<float>(ffml_offset_bytes(a_2d, {1,2,0,0}), 6 * sizeof(float));
    test_equal<float>(ffml_offset_bytes(a_2d, {1,3,0,0}), 7 * sizeof(float));
    test_equal<float>(ffml_offset_bytes(a_2d, {2,0,0,0}), 8 * sizeof(float));
    test_equal<float>(ffml_offset_bytes(a_2d, {2,1,0,0}), 9 * sizeof(float));
    test_equal<float>(ffml_offset_bytes(a_2d, {2,2,0,0}), 10 * sizeof(float));
    test_equal<float>(ffml_offset_bytes(a_2d, {2,3,0,0}), 11 * sizeof(float));

    test_equal<float>(ffml_offset_flat_bytes(a_2d, 0), 0 * sizeof(float));
    test_equal<float>(ffml_offset_flat_bytes(a_2d, 1), 1 * sizeof(float));
    test_equal<float>(ffml_offset_flat_bytes(a_2d, 2), 2 * sizeof(float));
    test_equal<float>(ffml_offset_flat_bytes(a_2d, 3), 3 * sizeof(float));
    test_equal<float>(ffml_offset_flat_bytes(a_2d, 4), 4 * sizeof(float));
    test_equal<float>(ffml_offset_flat_bytes(a_2d, 5), 5 * sizeof(float));

    // sample fake data
    ffml_set_data(a_2d, {0,0,0,0}, 2.0f);
    ffml_set_data(a_2d, {1,0,0,0}, 3.0f);
    ffml_set_data(a_2d, {2,0,0,0}, 4.0f);

    ffml_set_data(a_2d, {0,1,0,0}, 5.0f);
    ffml_set_data(a_2d, {1,1,0,0}, 6.0f);
    ffml_set_data(a_2d, {2,1,0,0}, 7.0f);

    ffml_set_data(a_2d, {0,2,0,0}, 8.0f);
    ffml_set_data(a_2d, {1,2,0,0}, 9.0f);
    ffml_set_data(a_2d, {2,2,0,0}, 10.0f);

    ffml_set_data(a_2d, {0,3,0,0}, 11.0f);
    ffml_set_data(a_2d, {1,3,0,0}, 12.0f);
    ffml_set_data(a_2d, {2,3,0,0}, 13.0f);


    ffml_set_data(b_2d, {0,0,0,0}, 100.0f);
    ffml_set_data(b_2d, {1,0,0,0}, 101.0f);
    ffml_set_data(b_2d, {2,0,0,0}, 102.0f);

    ffml_set_data(b_2d, {0,1,0,0}, 103.0f);
    ffml_set_data(b_2d, {1,1,0,0}, 104.0f);
    ffml_set_data(b_2d, {2,1,0,0}, 105.0f);

    ffml_set_data(b_2d, {0,2,0,0}, 106.0f);
    ffml_set_data(b_2d, {1,2,0,0}, 107.0f);
    ffml_set_data(b_2d, {2,2,0,0}, 108.0f);

    ffml_set_data(b_2d, {0,3,0,0}, 109.0f);
    ffml_set_data(b_2d, {1,3,0,0}, 110.0f);
    ffml_set_data(b_2d, {2,3,0,0}, 111.0f);


    ffml_set_data(c_2d, {0,0,0,0}, 10.0f);
    ffml_set_data(c_2d, {1,0,0,0}, 20.0f);
    ffml_set_data(c_2d, {2,0,0,0}, 30.0f);

    ffml_set_data(c_2d, {0,1,0,0}, 40.0f);
    ffml_set_data(c_2d, {1,1,0,0}, 50.0f);
    ffml_set_data(c_2d, {2,1,0,0}, 60.0f);

    ffml_set_data(c_2d, {0,2,0,0}, 70.0f);
    ffml_set_data(c_2d, {1,2,0,0}, 80.0f);
    ffml_set_data(c_2d, {2,2,0,0}, 90.0f);

    ffml_set_data(c_2d, {0,3,0,0}, 100.0f);
    ffml_set_data(c_2d, {1,3,0,0}, 110.0f);
    ffml_set_data(c_2d, {2,3,0,0}, 120.0f);



    //     forward pass
    ffml_cgraph_forward(cgraph);

    // ffml_debug_print_cgraph_data(cgraph);

    ffml_zerograd(cgraph);

    //     backward pass
    ffml_cgraph_backward(cgraph);
    // 

    // ffml_debug_print_cgraph_data(cgraph);

    test_equal<float>(ffml_get_data(a_2d, {0,0,0,0}), 2.0f);
    test_equal<float>(ffml_get_data(a_2d, {1,0,0,0}), 3.0f);
    test_equal<float>(ffml_get_data(a_2d, {2,0,0,0}), 4.0f);

    test_equal<float>(ffml_get_data(a_2d, {0,1,0,0}), 5.0f);
    test_equal<float>(ffml_get_data(a_2d, {1,1,0,0}), 6.0f);
    test_equal<float>(ffml_get_data(a_2d, {2,1,0,0}), 7.0f);

    test_equal<float>(ffml_get_data(a_2d, {0,2,0,0}), 8.0f);
    test_equal<float>(ffml_get_data(a_2d, {1,2,0,0}), 9.0f);
    test_equal<float>(ffml_get_data(a_2d, {2,2,0,0}), 10.0f);

    test_equal<float>(ffml_get_data(a_2d, {0,3,0,0}), 11.0f);
    test_equal<float>(ffml_get_data(a_2d, {1,3,0,0}), 12.0f);
    test_equal<float>(ffml_get_data(a_2d, {2,3,0,0}), 13.0f);

    test_equal<float>(ffml_get_data(b_2d, {0,0,0,0}), 100.0f);
    test_equal<float>(ffml_get_data(b_2d, {1,0,0,0}), 101.0f);
    test_equal<float>(ffml_get_data(b_2d, {2,0,0,0}), 102.0f);

    test_equal<float>(ffml_get_data(b_2d, {0,1,0,0}), 103.0f);
    test_equal<float>(ffml_get_data(b_2d, {1,1,0,0}), 104.0f);
    test_equal<float>(ffml_get_data(b_2d, {2,1,0,0}), 105.0f);

    test_equal<float>(ffml_get_data(b_2d, {0,2,0,0}), 106.0f);
    test_equal<float>(ffml_get_data(b_2d, {1,2,0,0}), 107.0f);
    test_equal<float>(ffml_get_data(b_2d, {2,2,0,0}), 108.0f);

    test_equal<float>(ffml_get_data(b_2d, {0,3,0,0}), 109.0f);
    test_equal<float>(ffml_get_data(b_2d, {1,3,0,0}), 110.0f);
    test_equal<float>(ffml_get_data(b_2d, {2,3,0,0}), 111.0f);

    test_equal<float>(ffml_get_data(add_2d, {0,0,0,0}), 2.0f + 100.0f);
    test_equal<float>(ffml_get_data(add_2d, {1,0,0,0}), 3.0f + 101.0f);
    test_equal<float>(ffml_get_data(add_2d, {2,0,0,0}), 4.0f + 102.0f);

    test_equal<float>(ffml_get_data(add_2d, {0,1,0,0}), 5.0f + 103.0f);
    test_equal<float>(ffml_get_data(add_2d, {1,1,0,0}), 6.0f + 104.0f);
    test_equal<float>(ffml_get_data(add_2d, {2,1,0,0}), 7.0f + 105.0f);

    test_equal<float>(ffml_get_data(add_2d, {0,2,0,0}), 8.0f + 106.0f);
    test_equal<float>(ffml_get_data(add_2d, {1,2,0,0}), 9.0f + 107.0f);
    test_equal<float>(ffml_get_data(add_2d, {2,2,0,0}), 10.0f + 108.0f);

    test_equal<float>(ffml_get_data(add_2d, {0,3,0,0}), 11.0f + 109.0f);
    test_equal<float>(ffml_get_data(add_2d, {1,3,0,0}), 12.0f + 110.0f);
    test_equal<float>(ffml_get_data(add_2d, {2,3,0,0}), 13.0f + 111.0f);

    test_equal<float>(ffml_get_data(c_2d, {0,0,0,0}), 10.0f);
    test_equal<float>(ffml_get_data(c_2d, {1,0,0,0}), 20.0f);
    test_equal<float>(ffml_get_data(c_2d, {2,0,0,0}), 30.0f);

    test_equal<float>(ffml_get_data(c_2d, {0,1,0,0}), 40.0f);
    test_equal<float>(ffml_get_data(c_2d, {1,1,0,0}), 50.0f);
    test_equal<float>(ffml_get_data(c_2d, {2,1,0,0}), 60.0f);

    test_equal<float>(ffml_get_data(c_2d, {0,2,0,0}), 70.0f);
    test_equal<float>(ffml_get_data(c_2d, {1,2,0,0}), 80.0f);
    test_equal<float>(ffml_get_data(c_2d, {2,2,0,0}), 90.0f);

    test_equal<float>(ffml_get_data(c_2d, {0,3,0,0}), 100.0f);
    test_equal<float>(ffml_get_data(c_2d, {1,3,0,0}), 110.0f);
    test_equal<float>(ffml_get_data(c_2d, {2,3,0,0}), 120.0f);

    test_equal<float>(ffml_get_data(mul_2d, {0,0,0,0}), (2.0f + 100.0f) * 10.0f);
    test_equal<float>(ffml_get_data(mul_2d, {1,0,0,0}), (3.0f + 101.0f) * 20.0f);
    test_equal<float>(ffml_get_data(mul_2d, {2,0,0,0}), (4.0f + 102.0f) * 30.0f);

    test_equal<float>(ffml_get_data(mul_2d, {0,1,0,0}), (5.0f + 103.0f) * 40.0f);
    test_equal<float>(ffml_get_data(mul_2d, {1,1,0,0}), (6.0f + 104.0f) * 50.0f);
    test_equal<float>(ffml_get_data(mul_2d, {2,1,0,0}), (7.0f + 105.0f) * 60.0f);

    test_equal<float>(ffml_get_data(mul_2d, {0,2,0,0}), (8.0f + 106.0f) * 70.0f);
    test_equal<float>(ffml_get_data(mul_2d, {1,2,0,0}), (9.0f + 107.0f) * 80.0f);
    test_equal<float>(ffml_get_data(mul_2d, {2,2,0,0}), (10.0f + 108.0f) * 90.0f);

    test_equal<float>(ffml_get_data(mul_2d, {0,3,0,0}), (11.0f + 109.0f) * 100.0f);
    test_equal<float>(ffml_get_data(mul_2d, {1,3,0,0}), (12.0f + 110.0f) * 110.0f);
    test_equal<float>(ffml_get_data(mul_2d, {2,3,0,0}), (13.0f + 111.0f) * 120.0f);

    // grads

    test_equal<float>(ffml_get_grad(mul_2d, {0,0,0,0}), 1.0f);
    test_equal<float>(ffml_get_grad(mul_2d, {1,0,0,0}), 1.0f);
    test_equal<float>(ffml_get_grad(mul_2d, {2,0,0,0}), 1.0f);
    test_equal<float>(ffml_get_grad(mul_2d, {0,1,0,0}), 1.0f);
    test_equal<float>(ffml_get_grad(mul_2d, {1,1,0,0}), 1.0f);
    test_equal<float>(ffml_get_grad(mul_2d, {2,1,0,0}), 1.0f);
    test_equal<float>(ffml_get_grad(mul_2d, {0,2,0,0}), 1.0f);
    test_equal<float>(ffml_get_grad(mul_2d, {1,2,0,0}), 1.0f);
    test_equal<float>(ffml_get_grad(mul_2d, {2,2,0,0}), 1.0f);
    test_equal<float>(ffml_get_grad(mul_2d, {0,3,0,0}), 1.0f);
    test_equal<float>(ffml_get_grad(mul_2d, {1,3,0,0}), 1.0f);
    test_equal<float>(ffml_get_grad(mul_2d, {2,3,0,0}), 1.0f);

    test_equal<float>(ffml_get_grad(add_2d, {0,0,0,0}), ffml_get_grad(mul_2d, {0,0,0,0}) * ffml_get_data(c_2d, {0,0,0,0}));
    test_equal<float>(ffml_get_grad(add_2d, {1,0,0,0}), ffml_get_grad(mul_2d, {1,0,0,0}) * ffml_get_data(c_2d, {1,0,0,0}));
    test_equal<float>(ffml_get_grad(add_2d, {2,0,0,0}), ffml_get_grad(mul_2d, {2,0,0,0}) * ffml_get_data(c_2d, {2,0,0,0}));
    test_equal<float>(ffml_get_grad(add_2d, {0,1,0,0}), ffml_get_grad(mul_2d, {0,1,0,0}) * ffml_get_data(c_2d, {0,1,0,0}));
    test_equal<float>(ffml_get_grad(add_2d, {1,1,0,0}), ffml_get_grad(mul_2d, {1,1,0,0}) * ffml_get_data(c_2d, {1,1,0,0}));
    test_equal<float>(ffml_get_grad(add_2d, {2,1,0,0}), ffml_get_grad(mul_2d, {2,1,0,0}) * ffml_get_data(c_2d, {2,1,0,0}));
    test_equal<float>(ffml_get_grad(add_2d, {0,2,0,0}), ffml_get_grad(mul_2d, {0,2,0,0}) * ffml_get_data(c_2d, {0,2,0,0}));
    test_equal<float>(ffml_get_grad(add_2d, {1,2,0,0}), ffml_get_grad(mul_2d, {1,2,0,0}) * ffml_get_data(c_2d, {1,2,0,0}));
    test_equal<float>(ffml_get_grad(add_2d, {2,2,0,0}), ffml_get_grad(mul_2d, {2,2,0,0}) * ffml_get_data(c_2d, {2,2,0,0}));
    test_equal<float>(ffml_get_grad(add_2d, {0,3,0,0}), ffml_get_grad(mul_2d, {0,3,0,0}) * ffml_get_data(c_2d, {0,3,0,0}));
    test_equal<float>(ffml_get_grad(add_2d, {1,3,0,0}), ffml_get_grad(mul_2d, {1,3,0,0}) * ffml_get_data(c_2d, {1,3,0,0}));
    test_equal<float>(ffml_get_grad(add_2d, {2,3,0,0}), ffml_get_grad(mul_2d, {2,3,0,0}) * ffml_get_data(c_2d, {2,3,0,0}));

    test_equal<float>(ffml_get_grad(c_2d, {0,0,0,0}), ffml_get_grad(mul_2d, {0,0,0,0}) * ffml_get_data(add_2d, {0,0,0,0}));
    test_equal<float>(ffml_get_grad(c_2d, {1,0,0,0}), ffml_get_grad(mul_2d, {1,0,0,0}) * ffml_get_data(add_2d, {1,0,0,0}));
    test_equal<float>(ffml_get_grad(c_2d, {2,0,0,0}), ffml_get_grad(mul_2d, {2,0,0,0}) * ffml_get_data(add_2d, {2,0,0,0}));
    test_equal<float>(ffml_get_grad(c_2d, {0,1,0,0}), ffml_get_grad(mul_2d, {0,1,0,0}) * ffml_get_data(add_2d, {0,1,0,0}));
    test_equal<float>(ffml_get_grad(c_2d, {1,1,0,0}), ffml_get_grad(mul_2d, {1,1,0,0}) * ffml_get_data(add_2d, {1,1,0,0}));
    test_equal<float>(ffml_get_grad(c_2d, {2,1,0,0}), ffml_get_grad(mul_2d, {2,1,0,0}) * ffml_get_data(add_2d, {2,1,0,0}));
    test_equal<float>(ffml_get_grad(c_2d, {0,2,0,0}), ffml_get_grad(mul_2d, {0,2,0,0}) * ffml_get_data(add_2d, {0,2,0,0}));
    test_equal<float>(ffml_get_grad(c_2d, {1,2,0,0}), ffml_get_grad(mul_2d, {1,2,0,0}) * ffml_get_data(add_2d, {1,2,0,0}));
    test_equal<float>(ffml_get_grad(c_2d, {2,2,0,0}), ffml_get_grad(mul_2d, {2,2,0,0}) * ffml_get_data(add_2d, {2,2,0,0}));
    test_equal<float>(ffml_get_grad(c_2d, {0,3,0,0}), ffml_get_grad(mul_2d, {0,3,0,0}) * ffml_get_data(add_2d, {0,3,0,0}));
    test_equal<float>(ffml_get_grad(c_2d, {1,3,0,0}), ffml_get_grad(mul_2d, {1,3,0,0}) * ffml_get_data(add_2d, {1,3,0,0}));
    test_equal<float>(ffml_get_grad(c_2d, {2,3,0,0}), ffml_get_grad(mul_2d, {2,3,0,0}) * ffml_get_data(add_2d, {2,3,0,0}));

    test_equal<float>(ffml_get_grad(a_2d, {0,0,0,0}), ffml_get_grad(add_2d, {0,0,0,0}));
    test_equal<float>(ffml_get_grad(a_2d, {1,0,0,0}), ffml_get_grad(add_2d, {1,0,0,0}));
    test_equal<float>(ffml_get_grad(a_2d, {2,0,0,0}), ffml_get_grad(add_2d, {2,0,0,0}));
    test_equal<float>(ffml_get_grad(a_2d, {0,1,0,0}), ffml_get_grad(add_2d, {0,1,0,0}));
    test_equal<float>(ffml_get_grad(a_2d, {1,1,0,0}), ffml_get_grad(add_2d, {1,1,0,0}));
    test_equal<float>(ffml_get_grad(a_2d, {2,1,0,0}), ffml_get_grad(add_2d, {2,1,0,0}));
    test_equal<float>(ffml_get_grad(a_2d, {0,2,0,0}), ffml_get_grad(add_2d, {0,2,0,0}));
    test_equal<float>(ffml_get_grad(a_2d, {1,2,0,0}), ffml_get_grad(add_2d, {1,2,0,0}));
    test_equal<float>(ffml_get_grad(a_2d, {2,2,0,0}), ffml_get_grad(add_2d, {2,2,0,0}));
    test_equal<float>(ffml_get_grad(a_2d, {0,3,0,0}), ffml_get_grad(add_2d, {0,3,0,0}));
    test_equal<float>(ffml_get_grad(a_2d, {1,3,0,0}), ffml_get_grad(add_2d, {1,3,0,0}));
    test_equal<float>(ffml_get_grad(a_2d, {2,3,0,0}), ffml_get_grad(add_2d, {2,3,0,0}));

    test_equal<float>(ffml_get_grad(b_2d, {0,0,0,0}), ffml_get_grad(add_2d, {0,0,0,0}));
    test_equal<float>(ffml_get_grad(b_2d, {1,0,0,0}), ffml_get_grad(add_2d, {1,0,0,0}));
    test_equal<float>(ffml_get_grad(b_2d, {2,0,0,0}), ffml_get_grad(add_2d, {2,0,0,0}));
    test_equal<float>(ffml_get_grad(b_2d, {0,1,0,0}), ffml_get_grad(add_2d, {0,1,0,0}));
    test_equal<float>(ffml_get_grad(b_2d, {1,1,0,0}), ffml_get_grad(add_2d, {1,1,0,0}));
    test_equal<float>(ffml_get_grad(b_2d, {2,1,0,0}), ffml_get_grad(add_2d, {2,1,0,0}));
    test_equal<float>(ffml_get_grad(b_2d, {0,2,0,0}), ffml_get_grad(add_2d, {0,2,0,0}));
    test_equal<float>(ffml_get_grad(b_2d, {1,2,0,0}), ffml_get_grad(add_2d, {1,2,0,0}));
    test_equal<float>(ffml_get_grad(b_2d, {2,2,0,0}), ffml_get_grad(add_2d, {2,2,0,0}));
    test_equal<float>(ffml_get_grad(b_2d, {0,3,0,0}), ffml_get_grad(add_2d, {0,3,0,0}));
    test_equal<float>(ffml_get_grad(b_2d, {1,3,0,0}), ffml_get_grad(add_2d, {1,3,0,0}));
    test_equal<float>(ffml_get_grad(b_2d, {2,3,0,0}), ffml_get_grad(add_2d, {2,3,0,0}));
    

    ffml_debug_print_cgraph_data(cgraph);

    //     update weights (optimizer)

    //     evaluate

    // save model

    // print hello world:
    // printf("Hello, World!\n");

    // test_almost_equal(ffml_get_data(a, {0,0,0,0}), -2.0f);
    // test_almost_equal(ffml_get_data(b, {0,0,0,0}), 3.0f);
    // test_almost_equal(ffml_get_data(mul, {0,0,0,0}), -6.0f);

    // test_almost_equal(ffml_get_data(c, {0,0,0,0}), 10.0f);
    // test_almost_equal(ffml_get_data(add, {0,0,0,0}), 4.0f);

    // test_almost_equal(ffml_get_data(d, {0,0,0,0}), 6.0f);
    // test_almost_equal(ffml_get_data(sub, {0,0,0,0}), -2.0f);

    // test_almost_equal(ffml_get_data(e, {0,0,0,0}), 2.0f);
    // test_almost_equal(ffml_get_data(div, {0,0,0,0}), -1.0f);

    // test_almost_equal(ffml_get_data(f, {0,0,0,0}), 0.5f);
    // test_almost_equal(ffml_get_data(add2, {0,0,0,0}), -0.5f);

    // test_almost_equal(ffml_get_data(two, {0,0,0,0}), 2.0f);
    // test_almost_equal(ffml_get_data(pow, {0,0,0,0}), 0.25f);

    // test_almost_equal(ffml_get_data(neg, {0,0,0,0}), -0.25f);

    // test_almost_equal(ffml_get_data(tanh, {0,0,0,0}), -0.2449186624037091292778f);

    // //------

    // test_equal(ffml_get_grad(tanh, {0,0,0,0}), 1.0f);

    // test_almost_equal_grad(neg, {0,0,0,0}, 1.0f - std::pow(std::tanh( ffml_get_data(neg, {0,0,0,0}) ),2)); // derivative of tanh is 1 - tanh^2
    // test_almost_equal_grad(neg, {0,0,0,0}, 0.940014849f); // derivative of tanh is 1 - tanh^2

    // test_almost_equal_grad(pow, {0,0,0,0}, -0.940014849f); // grad of neg is simply -1 * previous grad

    // test_almost_equal_grad(add2, {0,0,0,0}, 2.0f * std::pow( ffml_get_data(add2, {0,0,0,0}), 2.0f - 1.0f) * ffml_get_grad(pow, {0,0,0,0}) ); // grad of pow is n * x^(n-1) * previous grad
    // test_almost_equal_grad(add2, {0,0,0,0}, 0.940014849f);

    // test_almost_equal_grad(div, {0,0,0,0}, 1.0f * ffml_get_grad(add2, {0,0,0,0})); // grad of add2 is simply 1 * previous grad
    // test_almost_equal_grad(div, {0,0,0,0}, 0.940014849f);
    // test_almost_equal_grad(f, {0,0,0,0}, 0.940014849f);

    // // If you're referring to basic division of two variables, say f(x, y) = x / y, then:
    // //     The partial derivative with respect to x is 1/y,
    // //     The partial derivative with respect to y is -x / y^2.
    // test_almost_equal_grad(sub, {0,0,0,0}, (1.0f / ffml_get_data(e, {0,0,0,0})) * ffml_get_grad(div, {0,0,0,0}));
    // test_almost_equal_grad(sub, {0,0,0,0}, 0.470007f);
    // test_almost_equal_grad(e, {0,0,0,0}, (-1.0f * ffml_get_data(sub, {0,0,0,0})) / std::pow(ffml_get_data(e, {0,0,0,0}), 2.0f) * ffml_get_grad(div, {0,0,0,0}));
    // test_almost_equal_grad(e, {0,0,0,0}, 0.470007f);

    // test_almost_equal_grad(add, {0,0,0,0}, 1.0f * ffml_get_grad(sub, {0,0,0,0}));
    // test_almost_equal_grad(add, {0,0,0,0}, 0.470007f);

    // test_almost_equal_grad(c, {0,0,0,0}, 1.0f * ffml_get_grad(add, {0,0,0,0}));
    // test_almost_equal_grad(c, {0,0,0,0}, 0.470007f);

    // test_almost_equal_grad(mul, {0,0,0,0}, 1.0f * ffml_get_grad(add, {0,0,0,0}));
    // test_almost_equal_grad(mul, {0,0,0,0}, 0.470007f);

    // // grad of multiplication is the other variable
    // test_almost_equal_grad(b, {0,0,0,0}, ffml_get_data(a, {0,0,0,0}) * ffml_get_grad(mul, {0,0,0,0}));
    // test_almost_equal_grad(b, {0,0,0,0}, -0.940015);

    // test_almost_equal_grad(a, {0,0,0,0}, ffml_get_data(b, {0,0,0,0}) * ffml_get_grad(mul, {0,0,0,0}));
    // test_almost_equal_grad(a, {0,0,0,0}, 1.410022);
    
    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too
}

#endif