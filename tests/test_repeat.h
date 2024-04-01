#ifndef TEST_REPEAT_H
#define TEST_REPEAT_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_repeat() {
    test_name("Testing Repeat");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    auto t0 = ffml_tensor_create(2, {20, 12, 0, 0}, "t0");
    auto t1 = ffml_tensor_create(2, {2, 3, 0, 0}, "t1");
    auto repeat = ffml_op(FFML_OP_REPEAT, t1, t0, "repeat");
    auto mul = ffml_op(FFML_OP_MUL, t0, repeat);

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(mul);
    ffml_cgraph_alloc(cgraph, pool, true);

    for(uint64_t i = 0; i < 20*12; i++) {
        ffml_set_data_flat(t0, i, (float)i);
    }

    for (uint64_t i = 0; i < 2*3; i++) {
        ffml_set_data_flat(t1, i, (float)i);
    }

    ffml_cgraph_forward(cgraph);

    ffml_debug_print_cgraph_data(cgraph);

    ffml_zerograd(cgraph);

    ffml_cgraph_backward(cgraph);

    ffml_debug_print_cgraph_data(cgraph);

    test_tensor_data_flat_almost_equal(repeat, {0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                                3.0f, 4.0f, 5.0f, 3.0f, 4.0f, 5.0f});

    test_tensor_data_flat_almost_equal(mul, {0.0f, 1.0f, 4.0f, 0.0f, 4.0f, 10.0f, 0.0f, 7.0f, 16.0f, 0.0f, 10.0f, 22.0f, 36.0f, 52.0f});

    test_tensor_grad_flat_almost_equal(t0, {0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                                3.0f, 4.0f, 5.0f, 3.0f, 4.0f, 5.0f});

    test_tensor_grad_flat_almost_equal(t1, {
        0.0f + 3.0f + 6.0f + 9.0f +
        24.0f + 27.0f + 30.0f + 33.0f +
        48.0f + 51.0f + 54.0f + 57.0f +
        72.0f + 75.0f + 78.0f + 81.0f +
        96.0f + 99.0f + 102.0f + 105.0f +
        120.0f + 123.0f + 126.0f + 129.0f +
        144.0f + 147.0f + 150.0f + 153.0f +
        168.0f + 171.0f + 174.0f + 177.0f +
        192.0f + 195.0f + 198.0f + 201.0f +
        216.0f + 219.0f + 222.0f + 225.0f, });

    // free memory pool/context
    ffml_memory_pool_destroy(pool);

}

#endif