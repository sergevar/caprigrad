#ifndef TEST_GELU_APPROX_H
#define TEST_GELU_APPROX_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_gelu_approx() {
    test_name("Test GELU Approx=tanh");
        // 

    // printf("Skipping GELU");
    // return;

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    ffml_tensor * a = ffml_tensor_create(1, {10,0,0,0}, "a");

    ffml_tensor * gelu = ffml_unary_op(FFML_OP_GELU_APPROX_TANH, a, "gelu");

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(gelu);

    ffml_cgraph_alloc(cgraph, pool);

    // sample fake data
    ffml_set_data(a, {0,0,0,0}, 5.0f);
    ffml_set_data(a, {1,0,0,0}, 3.0f);
    ffml_set_data(a, {2,0,0,0}, 1.0f);
    ffml_set_data(a, {3,0,0,0}, 2.0f);
    ffml_set_data(a, {4,0,0,0}, 3.0f);
    ffml_set_data(a, {5,0,0,0}, 4.0f);
    ffml_set_data(a, {6,0,0,0}, 5.0f);
    ffml_set_data(a, {7,0,0,0}, 6.0f);
    ffml_set_data(a, {8,0,0,0}, 7.0f);
    ffml_set_data(a, {9,0,0,0}, 8.0f);

    //     forward pass
    ffml_cgraph_forward(cgraph);

    ffml_zerograd(cgraph);

    //     backward pass
    ffml_cgraph_backward(cgraph);
    // 


    test_equal(ffml_get_data(a, {0,0,0,0}), 5.0f);
    test_equal(ffml_get_data(a, {1,0,0,0}), 3.0f);
    test_equal(ffml_get_data(a, {2,0,0,0}), 1.0f);
    test_equal(ffml_get_data(a, {3,0,0,0}), 2.0f);
    test_equal(ffml_get_data(a, {4,0,0,0}), 3.0f);
    test_equal(ffml_get_data(a, {5,0,0,0}), 4.0f);
    test_equal(ffml_get_data(a, {6,0,0,0}), 5.0f);
    test_equal(ffml_get_data(a, {7,0,0,0}), 6.0f);
    test_equal(ffml_get_data(a, {8,0,0,0}), 7.0f);
    test_equal(ffml_get_data(a, {9,0,0,0}), 8.0f);

// b1	0.03039557324979
// b2	0.0041135935148996
// b3	5.5671434345923E-4
// b4	0.0015133064834677
// b5	0.0041135935148996
// b6	0.011181906501219
// b7	0.03039557324979
// b8	0.082623734430501
// b9	0.22459459590186
// b10	0.61051140881012

    test_almost_equal(ffml_get_data(gelu, {0,0,0,0}), 5.0f);
    test_almost_equal(ffml_get_data(gelu, {1,0,0,0}), 2.9964f);
    test_almost_equal(ffml_get_data(gelu, {2,0,0,0}), 0.8412f);
    test_almost_equal(ffml_get_data(gelu, {3,0,0,0}), 1.9546f);
    test_almost_equal(ffml_get_data(gelu, {4,0,0,0}), 2.9964f);
    test_almost_equal(ffml_get_data(gelu, {5,0,0,0}), 3.9999f);
    test_almost_equal(ffml_get_data(gelu, {6,0,0,0}), 5.0f);
    test_almost_equal(ffml_get_data(gelu, {7,0,0,0}), 6.0f);
    test_almost_equal(ffml_get_data(gelu, {8,0,0,0}), 7.0f);
    test_almost_equal(ffml_get_data(gelu, {9,0,0,0}), 8.0f);

    test_almost_equal_grad(a, {0,0,0,0}, 1.0f);
    test_almost_equal_grad(a, {1,0,0,0}, 1.0116f);
    test_almost_equal_grad(a, {2,0,0,0}, 1.0830f);
    test_almost_equal_grad(a, {3,0,0,0}, 1.0861f);
    test_almost_equal_grad(a, {4,0,0,0}, 1.0116f);
    test_almost_equal_grad(a, {5,0,0,0}, 1.0003f);
    test_almost_equal_grad(a, {6,0,0,0}, 1.0f);
    test_almost_equal_grad(a, {7,0,0,0}, 1.0f);
    test_almost_equal_grad(a, {8,0,0,0}, 1.0f);
    test_almost_equal_grad(a, {9,0,0,0}, 1.0f);


    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too
}

#endif