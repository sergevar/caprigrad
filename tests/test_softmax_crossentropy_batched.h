#ifndef TEST_SOFTMAX_CROSS_ENTROPY_BATCHED_H
#define TEST_SOFTMAX_CROSS_ENTROPY_BATCHED_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_softmax_crossentropy_batched() {
    test_name("Test softmax crossentropy batched");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    ffml_tensor * a = ffml_tensor_create(2, {2,10,0,0}, "a");
    ffml_tensor * labels = ffml_tensor_create(2, {2,10,0,0}, "labels");

    ffml_tensor * softmax_cross = ffml_op(FFML_OP_SOFTMAX_CROSS_ENTROPY, a, labels);
    ffml_set_name(softmax_cross, "softmax_cross");

    ffml_tensor * final_loss = ffml_unary_op(FFML_OP_MEAN, softmax_cross);
    ffml_set_name(final_loss, "final_loss");

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(final_loss);

    ffml_cgraph_alloc(cgraph, pool);

    test_equal<float>(softmax_cross->size_bytes, 2 * sizeof(float));
    test_equal<float>(a->size_bytes, 2 * 10 * sizeof(float));
    test_equal<float>(labels->size_bytes, 2 * 10 * sizeof(float));

    // sample fake data
    ffml_set_data(a, {0,0,0,0}, 5.0f);
    ffml_set_data(a, {0,1,0,0}, 3.0f);
    ffml_set_data(a, {0,2,0,0}, 1.0f);
    ffml_set_data(a, {0,3,0,0}, 2.0f);
    ffml_set_data(a, {0,4,0,0}, 3.0f);
    ffml_set_data(a, {0,5,0,0}, 4.0f);
    ffml_set_data(a, {0,6,0,0}, 5.0f);
    ffml_set_data(a, {0,7,0,0}, 6.0f);
    ffml_set_data(a, {0,8,0,0}, 7.0f);
    ffml_set_data(a, {0,9,0,0}, 8.0f);

    ffml_set_data(a, {1,0,0,0}, -5.0f);
    ffml_set_data(a, {1,1,0,0}, -3.0f);
    ffml_set_data(a, {1,2,0,0}, -1.0f);
    ffml_set_data(a, {1,3,0,0}, -2.0f);
    ffml_set_data(a, {1,4,0,0}, -3.0f);
    ffml_set_data(a, {1,5,0,0}, -4.0f);
    ffml_set_data(a, {1,6,0,0}, -5.0f);
    ffml_set_data(a, {1,7,0,0}, -6.0f);
    ffml_set_data(a, {1,8,0,0}, -7.0f);
    ffml_set_data(a, {1,9,0,0}, -8.0f);

    ffml_set_data(labels, {0,0,0,0}, 0.0f);
    ffml_set_data(labels, {0,1,0,0}, 0.0f);
    ffml_set_data(labels, {0,2,0,0}, 1.0f);
    ffml_set_data(labels, {0,3,0,0}, 0.0f);
    ffml_set_data(labels, {0,4,0,0}, 0.0f);
    ffml_set_data(labels, {0,5,0,0}, 0.0f);
    ffml_set_data(labels, {0,6,0,0}, 0.0f);
    ffml_set_data(labels, {0,7,0,0}, 0.0f);
    ffml_set_data(labels, {0,8,0,0}, 0.0f);
    ffml_set_data(labels, {0,9,0,0}, 0.0f);

    ffml_set_data(labels, {1,0,0,0}, 0.0f);
    ffml_set_data(labels, {1,1,0,0}, 0.0f);
    ffml_set_data(labels, {1,2,0,0}, 0.0f);
    ffml_set_data(labels, {1,3,0,0}, 1.0f);
    ffml_set_data(labels, {1,4,0,0}, 0.0f);
    ffml_set_data(labels, {1,5,0,0}, 0.0f);
    ffml_set_data(labels, {1,6,0,0}, 0.0f);
    ffml_set_data(labels, {1,7,0,0}, 0.0f);
    ffml_set_data(labels, {1,8,0,0}, 0.0f);
    ffml_set_data(labels, {1,9,0,0}, 0.0f);

    //     forward pass
    ffml_cgraph_forward(cgraph);

    ffml_zerograd(cgraph);

    //     backward pass
    ffml_cgraph_backward(cgraph);
    // 


    test_equal(ffml_get_data(a, {0,0,0,0}), 5.0f);
    test_equal(ffml_get_data(a, {0,1,0,0}), 3.0f);
    test_equal(ffml_get_data(a, {0,2,0,0}), 1.0f);
    test_equal(ffml_get_data(a, {0,3,0,0}), 2.0f);
    test_equal(ffml_get_data(a, {0,4,0,0}), 3.0f);
    test_equal(ffml_get_data(a, {0,5,0,0}), 4.0f);
    test_equal(ffml_get_data(a, {0,6,0,0}), 5.0f);
    test_equal(ffml_get_data(a, {0,7,0,0}), 6.0f);
    test_equal(ffml_get_data(a, {0,8,0,0}), 7.0f);
    test_equal(ffml_get_data(a, {0,9,0,0}), 8.0f);

    test_equal(ffml_get_data(a, {1,0,0,0}), -5.0f);
    test_equal(ffml_get_data(a, {1,1,0,0}), -3.0f);
    test_equal(ffml_get_data(a, {1,2,0,0}), -1.0f);
    test_equal(ffml_get_data(a, {1,3,0,0}), -2.0f);
    test_equal(ffml_get_data(a, {1,4,0,0}), -3.0f);
    test_equal(ffml_get_data(a, {1,5,0,0}), -4.0f);
    test_equal(ffml_get_data(a, {1,6,0,0}), -5.0f);
    test_equal(ffml_get_data(a, {1,7,0,0}), -6.0f);
    test_equal(ffml_get_data(a, {1,8,0,0}), -7.0f);
    test_equal(ffml_get_data(a, {1,9,0,0}), -8.0f);

    test_equal(ffml_get_data(labels, {0,0,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {0,1,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {0,2,0,0}), 1.0f);
    test_equal(ffml_get_data(labels, {0,3,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {0,4,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {0,5,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {0,6,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {0,7,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {0,8,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {0,9,0,0}), 0.0f);

    test_equal(ffml_get_data(labels, {1,0,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {1,1,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {1,2,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {1,3,0,0}), 1.0f);
    test_equal(ffml_get_data(labels, {1,4,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {1,5,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {1,6,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {1,7,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {1,8,0,0}), 0.0f);
    test_equal(ffml_get_data(labels, {1,9,0,0}), 0.0f);

// Loss:  tensor(7.4935, grad_fn=<NegBackward0>)
// Logits.grad:  tensor([ 0.0304,  0.0041, -0.9994,  0.0015,  0.0041,  0.0112,  0.0304,  0.0826,
//          0.2246,  0.6105])

    test_almost_equal(ffml_get_data(softmax_cross, {0,0,0,0}), 7.4934f);
    test_almost_equal(ffml_get_data(softmax_cross, {1,0,0,0}), 1.5511f);

    test_almost_equal_grad(softmax_cross, {0,0,0,0}, 0.5f);
    test_almost_equal_grad(softmax_cross, {1,0,0,0}, 0.5f);

    test_almost_equal(ffml_get_data(final_loss, {0,0,0,0}), 4.5223f);

    test_almost_equal_grad(a, {0,0,0,0}, 0.0304f / 2.0f);
    test_almost_equal_grad(a, {0,1,0,0}, 0.0041f / 2.0f);
    test_almost_equal_grad(a, {0,2,0,0}, -0.9994f / 2.0f);
    test_almost_equal_grad(a, {0,3,0,0}, 0.0015f / 2.0f);
    test_almost_equal_grad(a, {0,4,0,0}, 0.0041f / 2.0f);
    test_almost_equal_grad(a, {0,5,0,0}, 0.0112f / 2.0f);
    test_almost_equal_grad(a, {0,6,0,0}, 0.0304f / 2.0f);
    test_almost_equal_grad(a, {0,7,0,0}, 0.0826f / 2.0f);
    test_almost_equal_grad(a, {0,8,0,0}, 0.2246f / 2.0f);
    test_almost_equal_grad(a, {0,9,0,0}, 0.6105f / 2.0f);

    test_tensor_grad_flat_almost_equal(a, {
        1.5198e-02,  2.0568e-03, -4.9972e-01,  7.5665e-04,  2.0568e-03,
        5.5910e-03,  1.5198e-02,  4.1312e-02,  1.1230e-01,  3.0526e-01,
        5.2780e-03,  3.8999e-02,  2.8817e-01, -3.9399e-01,  3.8999e-02,
        1.4347e-02,  5.2780e-03,  1.9417e-03,  7.1430e-04,  2.6278e-04
    });

    ffml_debug_print_cgraph_data(cgraph);

    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too
}

#endif