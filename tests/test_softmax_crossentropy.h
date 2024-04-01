#ifndef TEST_SOFTMAX_CROSS_ENTROPY_H
#define TEST_SOFTMAX_CROSS_ENTROPY_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_softmax_crossentropy() {
    test_name("Test softmax crossentropy");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    ffml_tensor * a = ffml_tensor_create(1, {10,0,0,0}, "a");
    ffml_tensor * labels = ffml_tensor_create(1, {10,0,0,0}, "labels");

    ffml_tensor * softmax_cross = ffml_op(FFML_OP_SOFTMAX_CROSS_ENTROPY, a, labels);
    ffml_set_name(softmax_cross, "softmax_cross");


    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(softmax_cross);

    ffml_cgraph_alloc(cgraph, pool);

    test_equal<float>(softmax_cross->size_bytes, 1 * sizeof(float));
    test_equal<float>(a->size_bytes, 10 * sizeof(float));
    test_equal<float>(labels->size_bytes, 10 * sizeof(float));

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

    ffml_set_data(labels, {0,0,0,0}, 0.0f);
    ffml_set_data(labels, {1,0,0,0}, 0.0f);
    ffml_set_data(labels, {2,0,0,0}, 1.0f);
    ffml_set_data(labels, {3,0,0,0}, 0.0f);
    ffml_set_data(labels, {4,0,0,0}, 0.0f);
    ffml_set_data(labels, {5,0,0,0}, 0.0f);
    ffml_set_data(labels, {6,0,0,0}, 0.0f);
    ffml_set_data(labels, {7,0,0,0}, 0.0f);
    ffml_set_data(labels, {8,0,0,0}, 0.0f);
    ffml_set_data(labels, {9,0,0,0}, 0.0f);

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

    test_equal(ffml_get_data(labels, {0,0,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {1,0,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {2,0,0,0}), 1.0f);
    test_equal(ffml_get_data(labels, {3,0,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {4,0,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {5,0,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {6,0,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {7,0,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {8,0,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {9,0,0,0}), 0.0f);

// Loss:  tensor(7.4935, grad_fn=<NegBackward0>)
// Logits.grad:  tensor([ 0.0304,  0.0041, -0.9994,  0.0015,  0.0041,  0.0112,  0.0304,  0.0826,
//          0.2246,  0.6105])

    test_almost_equal(ffml_get_data(softmax_cross, {0,0,0,0}), 7.4935f);

    test_almost_equal_grad(softmax_cross, {0,0,0,0}, 1.0f);

    test_almost_equal_grad(a, {0,0,0,0}, 0.0304f);
    test_almost_equal_grad(a, {1,0,0,0}, 0.0041f);
    test_almost_equal_grad(a, {2,0,0,0}, -0.9994f);
    test_almost_equal_grad(a, {3,0,0,0}, 0.0015f);
    test_almost_equal_grad(a, {4,0,0,0}, 0.0041f);
    test_almost_equal_grad(a, {5,0,0,0}, 0.0112f);
    test_almost_equal_grad(a, {6,0,0,0}, 0.0304f);
    test_almost_equal_grad(a, {7,0,0,0}, 0.0826f);
    test_almost_equal_grad(a, {8,0,0,0}, 0.2246f);
    test_almost_equal_grad(a, {9,0,0,0}, 0.6105f);


    ffml_debug_print_cgraph_data(cgraph);

    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too
}

#endif