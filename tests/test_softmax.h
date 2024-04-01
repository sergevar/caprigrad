#ifndef TEST_SOFTMAX_H
#define TEST_SOFTMAX_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_softmax() {
    test_name("Test softmax");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    ffml_tensor * a = ffml_tensor_create(1, {10,0,0,0}, "a");

    ffml_tensor * softmax = ffml_unary_op(FFML_OP_SOFTMAX, a);
    ffml_set_name(softmax, "softmax");

    ffml_tensor * c = ffml_tensor_create(1, {10,0,0,0}, "c");

    ffml_tensor * normal_mul = ffml_op(FFML_OP_MUL, softmax, c);
    ffml_set_name(normal_mul, "normal_mul");

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(normal_mul);

    ffml_cgraph_alloc(cgraph, pool);

    test_equal<float>(softmax->size_bytes, 10 * sizeof(float));
    test_equal<float>(a->size_bytes, 10 * sizeof(float));
    test_equal<float>(c->size_bytes, 10 * sizeof(float));
    test_equal<float>(normal_mul->size_bytes, 10 * sizeof(float));

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

    ffml_set_data(c, {0,0,0,0}, 11.0f);
    ffml_set_data(c, {1,0,0,0}, 22.0f);
    ffml_set_data(c, {2,0,0,0}, 33.0f);
    ffml_set_data(c, {3,0,0,0}, 44.0f);
    ffml_set_data(c, {4,0,0,0}, 55.0f);
    ffml_set_data(c, {5,0,0,0}, 66.0f);
    ffml_set_data(c, {6,0,0,0}, 77.0f);
    ffml_set_data(c, {7,0,0,0}, 88.0f);
    ffml_set_data(c, {8,0,0,0}, 99.0f);
    ffml_set_data(c, {9,0,0,0}, 1010.0f);

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

    test_almost_equal(ffml_get_data(softmax, {0,0,0,0}), 0.03039557324979f);
    test_almost_equal(ffml_get_data(softmax, {1,0,0,0}), 0.0041135935148996f);
    test_almost_equal(ffml_get_data(softmax, {2,0,0,0}), 5.5671434345923E-4f);
    test_almost_equal(ffml_get_data(softmax, {3,0,0,0}), 0.0015133064834677f);
    test_almost_equal(ffml_get_data(softmax, {4,0,0,0}), 0.0041135935148996f);
    test_almost_equal(ffml_get_data(softmax, {5,0,0,0}), 0.011181906501219f);
    test_almost_equal(ffml_get_data(softmax, {6,0,0,0}), 0.03039557324979f);
    test_almost_equal(ffml_get_data(softmax, {7,0,0,0}), 0.082623734430501f);
    test_almost_equal(ffml_get_data(softmax, {8,0,0,0}), 0.22459459590186f);
    test_almost_equal(ffml_get_data(softmax, {9,0,0,0}), 0.61051140881012f);

    // they must all add up to 1
    float sum = 0.0f;
    for (uint64_t i = 0; i < 10; i++) {
        sum += ffml_get_data(softmax, {i,0,0,0});
    }
    test_almost_equal(sum, 1.0f);

    test_almost_equal(ffml_get_data(c, {0,0,0,0}), 11.0f);
    test_almost_equal(ffml_get_data(c, {1,0,0,0}), 22.0f);
    test_almost_equal(ffml_get_data(c, {2,0,0,0}), 33.0f);
    test_almost_equal(ffml_get_data(c, {3,0,0,0}), 44.0f);
    test_almost_equal(ffml_get_data(c, {4,0,0,0}), 55.0f);
    test_almost_equal(ffml_get_data(c, {5,0,0,0}), 66.0f);
    test_almost_equal(ffml_get_data(c, {6,0,0,0}), 77.0f);
    test_almost_equal(ffml_get_data(c, {7,0,0,0}), 88.0f);
    test_almost_equal(ffml_get_data(c, {8,0,0,0}), 99.0f);
    test_almost_equal(ffml_get_data(c, {9,0,0,0}), 1010.0f);
    
    test_almost_equal_grad(normal_mul, {0,0,0,0}, 1.0f);
    test_almost_equal_grad(normal_mul, {1,0,0,0}, 1.0f);
    test_almost_equal_grad(normal_mul, {2,0,0,0}, 1.0f);
    test_almost_equal_grad(normal_mul, {3,0,0,0}, 1.0f);
    test_almost_equal_grad(normal_mul, {4,0,0,0}, 1.0f);
    test_almost_equal_grad(normal_mul, {5,0,0,0}, 1.0f);
    test_almost_equal_grad(normal_mul, {6,0,0,0}, 1.0f);
    test_almost_equal_grad(normal_mul, {7,0,0,0}, 1.0f);
    test_almost_equal_grad(normal_mul, {8,0,0,0}, 1.0f);
    test_almost_equal_grad(normal_mul, {9,0,0,0}, 1.0f);

    test_almost_equal_grad(softmax, {0,0,0,0}, 11.0f);
    test_almost_equal_grad(softmax, {1,0,0,0}, 22.0f);
    test_almost_equal_grad(softmax, {2,0,0,0}, 33.0f);
    test_almost_equal_grad(softmax, {3,0,0,0}, 44.0f);
    test_almost_equal_grad(softmax, {4,0,0,0}, 55.0f);
    test_almost_equal_grad(softmax, {5,0,0,0}, 66.0f);
    test_almost_equal_grad(softmax, {6,0,0,0}, 77.0f);
    test_almost_equal_grad(softmax, {7,0,0,0}, 88.0f);
    test_almost_equal_grad(softmax, {8,0,0,0}, 99.0f);

//     a.grad:  tensor([ -19.4208,   -2.5831,   -0.3435,   -0.9170,   -2.4473,   -6.5295,
//          -17.4147,  -46.4293, -123.7374,  219.8227])
// c.grad:  tensor([3.0396e-02, 4.1136e-03, 5.5671e-04, 1.5133e-03, 4.1136e-03, 1.1182e-02,
//         3.0396e-02, 8.2624e-02, 2.2459e-01, 6.1051e-01])

    test_almost_equal_grad(a, {0,0,0,0}, -19.4208f);
    test_almost_equal_grad(a, {1,0,0,0}, -2.5831f);
    test_almost_equal_grad(a, {2,0,0,0}, -0.3435f);
    test_almost_equal_grad(a, {3,0,0,0}, -0.9170f);
    test_almost_equal_grad(a, {4,0,0,0}, -2.4473f);
    test_almost_equal_grad(a, {5,0,0,0}, -6.5295f);
    test_almost_equal_grad(a, {6,0,0,0}, -17.4147f);
    test_almost_equal_grad(a, {7,0,0,0}, -46.4293f);
    test_almost_equal_grad(a, {8,0,0,0}, -123.7374f);
    test_almost_equal_grad(a, {9,0,0,0}, 219.8227f);

    test_almost_equal_grad(c, {0,0,0,0}, 3.0396e-02f);
    test_almost_equal_grad(c, {1,0,0,0}, 4.1136e-03f);
    test_almost_equal_grad(c, {2,0,0,0}, 5.5671e-04f);
    test_almost_equal_grad(c, {3,0,0,0}, 1.5133e-03f);
    test_almost_equal_grad(c, {4,0,0,0}, 4.1136e-03f);
    test_almost_equal_grad(c, {5,0,0,0}, 1.1182e-02f);
    test_almost_equal_grad(c, {6,0,0,0}, 3.0396e-02f);
    test_almost_equal_grad(c, {7,0,0,0}, 8.2624e-02f);
    test_almost_equal_grad(c, {8,0,0,0}, 2.2459e-01f);
    test_almost_equal_grad(c, {9,0,0,0}, 6.1051e-01f);


    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too
}

#endif