#ifndef TEST_RESHAPE_H
#define TEST_RESHAPE_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_reshape() {
    test_name("Testing Reshape");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    auto src = ffml_tensor_create(3, {2, 3, 4, 0}, "src");
    auto reshaped = ffml_reshape(src, 2, {2, 12, 0, 0});

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(reshaped);
    ffml_cgraph_alloc(cgraph, pool, true);

    ffml_set_data(src, {0,0,0,0}, 1.0f);
    ffml_set_data(src, {0,0,1,0}, 2.0f);
    ffml_set_data(src, {0,0,2,0}, 3.0f);
    ffml_set_data(src, {0,0,3,0}, 4.0f);
    ffml_set_data(src, {0,1,0,0}, 5.0f);
    ffml_set_data(src, {0,1,1,0}, 6.0f);
    ffml_set_data(src, {0,1,2,0}, 7.0f);
    ffml_set_data(src, {0,1,3,0}, 8.0f);
    ffml_set_data(src, {0,2,0,0}, 9.0f);
    ffml_set_data(src, {0,2,1,0}, 10.0f);
    ffml_set_data(src, {0,2,2,0}, 11.0f);
    ffml_set_data(src, {0,2,3,0}, 12.0f);

    ffml_set_data(src, {1,0,0,0}, 13.0f);
    ffml_set_data(src, {1,0,1,0}, 14.0f);
    ffml_set_data(src, {1,0,2,0}, 15.0f);
    ffml_set_data(src, {1,0,3,0}, 16.0f);
    ffml_set_data(src, {1,1,0,0}, 17.0f);
    ffml_set_data(src, {1,1,1,0}, 18.0f);
    ffml_set_data(src, {1,1,2,0}, 19.0f);
    ffml_set_data(src, {1,1,3,0}, 20.0f);
    ffml_set_data(src, {1,2,0,0}, 21.0f);
    ffml_set_data(src, {1,2,1,0}, 22.0f);
    ffml_set_data(src, {1,2,2,0}, 23.0f);
    ffml_set_data(src, {1,2,3,0}, 24.0f);

    ffml_cgraph_forward(cgraph);

    ffml_zerograd(cgraph);

    ffml_cgraph_backward(cgraph);

    // ffml_debug_print_cgraph_data(cgraph);

    test_equal<int>(src->n_dims, 3);
    test_equal<int>(src->ne[0], 2);
    test_equal<int>(src->ne[1], 3);
    test_equal<int>(src->ne[2], 4);

    test_equal<int>(reshaped->n_dims, 2);
    test_equal<int>(reshaped->ne[0], 2);
    test_equal<int>(reshaped->ne[1], 12);

    test_tensor_data_flat_almost_equal(src, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,

        13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f,
    });

    test_tensor_data_flat_almost_equal(reshaped, {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,

        13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 24.0f,
    });

    for(uint64_t i = 0; i < 2*3*4; i++) {
        test_almost_equal(ffml_get_grad_flat(reshaped, i), 1.0f);
    }

    for(uint64_t i = 0; i < 2*12; i++) {
        test_almost_equal(ffml_get_grad_flat(src, i), 1.0f);
    }

    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too

}

#endif