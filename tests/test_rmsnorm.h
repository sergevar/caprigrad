#ifndef TEST_RMSNORM_H
#define TEST_RMSNORM_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_rmsnorm() {
    test_name("Testing RMSNorm");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    auto src = ffml_tensor_create(1, {5,0,0,0}, "src");

    auto rmsnorm = ffml_unary_op(FFML_OP_RMS_NORM, src);

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(rmsnorm);
    ffml_cgraph_alloc(cgraph, pool, true);

    ffml_set_data(src, {0,0,0,0}, 1.0f);
    ffml_set_data(src, {1,0,0,0}, 2.0f);
    ffml_set_data(src, {2,0,0,0}, 3.0f);
    ffml_set_data(src, {3,0,0,0}, 4.0f);
    ffml_set_data(src, {4,0,0,0}, 5.0f);

    ffml_cgraph_forward(cgraph);

    ffml_zerograd(cgraph);

    ffml_cgraph_backward(cgraph);

    // ffml_debug_print_cgraph_data(cgraph);

    test_equal<int>(src->n_dims, 1);
    test_equal<int>(src->ne[0], 5);

    test_equal<int>(rmsnorm->n_dims, 1);
    test_equal<int>(rmsnorm->ne[0], 5);

    test_tensor_data_flat_almost_equal(src, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f
    });

    test_tensor_data_flat_almost_equal(rmsnorm, {
        0.3015f, 0.6030f, 0.9045f, 1.2060f, 1.5076f
    });

    test_tensor_grad_flat_almost_equal(src, {
        0.2193f, 0.1371f, 0.0548f, -0.0274f, -0.1096f
    });

    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too

}

#endif