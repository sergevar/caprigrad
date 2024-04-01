#ifndef TEST_MATMUL_HIGHER_DIM_H
#define TEST_MATMUL_HIGHER_DIM_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_matmul_higher_dim() {
    test_name("Testing Matmul Higher Dim");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    auto M1 = ffml_tensor_create(3, {3,2,3,0}, "M1");
    auto M2 = ffml_tensor_create(3, {3,3,2,0}, "M2");
    auto mm = ffml_op(FFML_OP_MATMUL, M1, M2);
    ffml_set_name(mm, "mm");
    auto multiplicator = ffml_tensor_create(3, {3,2,2,0}, "multiplicator");
    auto out = ffml_op(FFML_OP_MUL, mm, multiplicator);
    ffml_set_name(out, "out");

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(out);
    ffml_cgraph_alloc(cgraph, pool, true);

    ffml_set_data(M1, {0,0,0,0}, 1.0f);
    ffml_set_data(M1, {0,0,1,0}, 2.0f);
    ffml_set_data(M1, {0,0,2,0}, 3.0f);
    ffml_set_data(M1, {0,1,0,0}, 4.0f);
    ffml_set_data(M1, {0,1,1,0}, 5.0f);
    ffml_set_data(M1, {0,1,2,0}, 6.0f);
    ffml_set_data(M1, {1,0,0,0}, 7.0f);
    ffml_set_data(M1, {1,0,1,0}, 8.0f);
    ffml_set_data(M1, {1,0,2,0}, 9.0f);
    ffml_set_data(M1, {1,1,0,0}, 10.0f);
    ffml_set_data(M1, {1,1,1,0}, 11.0f);
    ffml_set_data(M1, {1,1,2,0}, 12.0f);
    ffml_set_data(M1, {2,0,0,0}, 13.0f);
    ffml_set_data(M1, {2,0,1,0}, 14.0f);
    ffml_set_data(M1, {2,0,2,0}, 15.0f);
    ffml_set_data(M1, {2,1,0,0}, 16.0f);
    ffml_set_data(M1, {2,1,1,0}, 17.0f);
    ffml_set_data(M1, {2,1,2,0}, 18.0f);
    
    ffml_set_data(M2, {0,0,0,0}, 0.11f);
    ffml_set_data(M2, {0,0,1,0}, 0.22f);
    ffml_set_data(M2, {0,1,0,0}, 0.33f);
    ffml_set_data(M2, {0,1,1,0}, 0.44f);
    ffml_set_data(M2, {0,2,0,0}, 0.55f);
    ffml_set_data(M2, {0,2,1,0}, 0.66f);
    ffml_set_data(M2, {1,0,0,0}, 0.77f);
    ffml_set_data(M2, {1,0,1,0}, 0.88f);
    ffml_set_data(M2, {1,1,0,0}, 0.99f);
    ffml_set_data(M2, {1,1,1,0}, 1.10f);
    ffml_set_data(M2, {1,2,0,0}, 1.21f);
    ffml_set_data(M2, {1,2,1,0}, 1.32f);
    ffml_set_data(M2, {2,0,0,0}, 1.43f);
    ffml_set_data(M2, {2,0,1,0}, 1.54f);
    ffml_set_data(M2, {2,1,0,0}, 1.65f);
    ffml_set_data(M2, {2,1,1,0}, 1.76f);
    ffml_set_data(M2, {2,2,0,0}, 1.87f);
    ffml_set_data(M2, {2,2,1,0}, 1.98f);

    for (uint64_t i = 0; i < 3; i++) {
        for (uint64_t j = 0; j < 2; j++) {
            for (uint64_t k = 0; k < 2; k++) {
                ffml_set_data(multiplicator, {i,j,k,0}, i+j+k);
            }
        }
    }

    ffml_cgraph_forward(cgraph);

    ffml_zerograd(cgraph);

    ffml_cgraph_backward(cgraph);

    // ffml_debug_print_cgraph_data(cgraph);

    // tests
    test_almost_equal(ffml_get_data(mm, {0,0,0,0}), 2.42f);
    test_almost_equal(ffml_get_data(mm, {0,0,1,0}), 3.08f);
    test_almost_equal(ffml_get_data(mm, {0,1,0,0}), 5.39f);
    test_almost_equal(ffml_get_data(mm, {0,1,1,0}), 7.04f);
    test_almost_equal(ffml_get_data(mm, {1,0,0,0}), 24.2f);
    test_almost_equal(ffml_get_data(mm, {1,0,1,0}), 26.84f);
    test_almost_equal(ffml_get_data(mm, {1,1,0,0}), 33.11f);
    test_almost_equal(ffml_get_data(mm, {1,1,1,0}), 36.74f);
    test_almost_equal(ffml_get_data(mm, {2,0,0,0}), 69.74f);
    test_almost_equal(ffml_get_data(mm, {2,0,1,0}), 74.36f);
    test_almost_equal(ffml_get_data(mm, {2,1,0,0}), 84.59f);
    test_almost_equal(ffml_get_data(mm, {2,1,1,0}), 90.20f);

    test_almost_equal(ffml_get_data(out, {0,0,0,0}), 0.0f);
    test_almost_equal(ffml_get_data(out, {0,0,1,0}), 3.08f);
    test_almost_equal(ffml_get_data(out, {0,1,0,0}), 5.39f);
    test_almost_equal(ffml_get_data(out, {0,1,1,0}), 14.08f);
    test_almost_equal(ffml_get_data(out, {1,0,0,0}), 24.2f);
    test_almost_equal(ffml_get_data(out, {1,0,1,0}), 53.68f);
    test_almost_equal(ffml_get_data(out, {1,1,0,0}), 66.22f);
    test_almost_equal(ffml_get_data(out, {1,1,1,0}), 110.22f);
    test_almost_equal(ffml_get_data(out, {2,0,0,0}), 139.48f);
    test_almost_equal(ffml_get_data(out, {2,0,1,0}), 223.08f);
    test_almost_equal(ffml_get_data(out, {2,1,0,0}), 253.77f);
    test_almost_equal(ffml_get_data(out, {2,1,1,0}), 360.80f);

    // all out grads should be 1.0f
    for (uint64_t i = 0; i < 3; i++) {
        for (uint64_t j = 0; j < 2; j++) {
            for (uint64_t k = 0; k < 2; k++) {
                test_almost_equal(ffml_get_grad(out, {i,j,k,0}), 1.0f);
            }
        }
    }

    test_almost_equal(ffml_get_grad(mm, {0,0,0,0}), 0.0f);
    test_almost_equal(ffml_get_grad(mm, {0,0,1,0}), 1.0f);
    test_almost_equal(ffml_get_grad(mm, {0,1,0,0}), 1.0f);
    test_almost_equal(ffml_get_grad(mm, {0,1,1,0}), 2.0f);
    test_almost_equal(ffml_get_grad(mm, {1,0,0,0}), 1.0f);
    test_almost_equal(ffml_get_grad(mm, {1,0,1,0}), 2.0f);
    test_almost_equal(ffml_get_grad(mm, {1,1,0,0}), 2.0f);
    test_almost_equal(ffml_get_grad(mm, {1,1,1,0}), 3.0f);
    test_almost_equal(ffml_get_grad(mm, {2,0,0,0}), 2.0f);
    test_almost_equal(ffml_get_grad(mm, {2,0,1,0}), 3.0f);
    test_almost_equal(ffml_get_grad(mm, {2,1,0,0}), 3.0f);
    test_almost_equal(ffml_get_grad(mm, {2,1,1,0}), 4.0f);

    // multiplicator.grad should match mm data exactly
    for (uint64_t i = 0; i < 3; i++) {
        for (uint64_t j = 0; j < 2; j++) {
            for (uint64_t k = 0; k < 2; k++) {
                test_almost_equal(ffml_get_grad(multiplicator, {i,j,k,0}), ffml_get_data(mm, {i,j,k,0}));
            }
        }
    }

    test_almost_equal(ffml_get_grad(M1, {0,0,0,0}), 0.22f);
    test_almost_equal(ffml_get_grad(M1, {0,0,1,0}), 0.44f);
    test_almost_equal(ffml_get_grad(M1, {0,0,2,0}), 0.66f);
    test_almost_equal(ffml_get_grad(M1, {0,1,0,0}), 0.55f);
    test_almost_equal(ffml_get_grad(M1, {0,1,1,0}), 1.21f);
    test_almost_equal(ffml_get_grad(M1, {0,1,2,0}), 1.87f);
    test_almost_equal(ffml_get_grad(M1, {1,0,0,0}), 2.53f);
    test_almost_equal(ffml_get_grad(M1, {1,0,1,0}), 3.19f);
    test_almost_equal(ffml_get_grad(M1, {1,0,2,0}), 3.85f);
    test_almost_equal(ffml_get_grad(M1, {1,1,0,0}), 4.18f);
    test_almost_equal(ffml_get_grad(M1, {1,1,1,0}), 5.28f);
    test_almost_equal(ffml_get_grad(M1, {1,1,2,0}), 6.38f);
    test_almost_equal(ffml_get_grad(M1, {2,0,0,0}), 7.48f);
    test_almost_equal(ffml_get_grad(M1, {2,0,1,0}), 8.58f);
    test_almost_equal(ffml_get_grad(M1, {2,0,2,0}), 9.68f);
    test_almost_equal(ffml_get_grad(M1, {2,1,0,0}), 10.45f);
    test_almost_equal(ffml_get_grad(M1, {2,1,1,0}), 11.99f);
    test_almost_equal(ffml_get_grad(M1, {2,1,2,0}), 13.53f);

    test_almost_equal(ffml_get_grad(M2, {0,0,0,0}), 4.0f);
    test_almost_equal(ffml_get_grad(M2, {0,0,1,0}), 9.0f);
    test_almost_equal(ffml_get_grad(M2, {0,1,0,0}), 5.0f);
    test_almost_equal(ffml_get_grad(M2, {0,1,1,0}), 12.0f);
    test_almost_equal(ffml_get_grad(M2, {0,2,0,0}), 6.0f);
    test_almost_equal(ffml_get_grad(M2, {0,2,1,0}), 15.0f);
    test_almost_equal(ffml_get_grad(M2, {1,0,0,0}), 27.0f);
    test_almost_equal(ffml_get_grad(M2, {1,0,1,0}), 44.0f);
    test_almost_equal(ffml_get_grad(M2, {1,1,0,0}), 30.0f);
    test_almost_equal(ffml_get_grad(M2, {1,1,1,0}), 49.0f);
    test_almost_equal(ffml_get_grad(M2, {1,2,0,0}), 33.0f);
    test_almost_equal(ffml_get_grad(M2, {1,2,1,0}), 54.0f);
    test_almost_equal(ffml_get_grad(M2, {2,0,0,0}), 74.0f);
    test_almost_equal(ffml_get_grad(M2, {2,0,1,0}), 103.0f);
    test_almost_equal(ffml_get_grad(M2, {2,1,0,0}), 79.0f);
    test_almost_equal(ffml_get_grad(M2, {2,1,1,0}), 110.0f);
    test_almost_equal(ffml_get_grad(M2, {2,2,0,0}), 84.0f);
    test_almost_equal(ffml_get_grad(M2, {2,2,1,0}), 117.0f);

    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too
}

#endif