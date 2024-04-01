#ifndef TEST_BROADCASTING2_H
#define TEST_BROADCASTING2_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_broadcasting2() {
    test_name("Test broadcasting2");
    // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    ffml_tensor * a = ffml_tensor_create(3, {2,4,4,0}, "a");
    ffml_tensor * two = ffml_tensor_create(1, {1,0,0,0}, "two");

    ffml_tensor * doubled = ffml_op(FFML_OP_MUL, a, two);
    ffml_set_name(doubled, "doubled");

    ffml_tensor * one = ffml_tensor_create(1, {1,0,0,0}, "one");

    ffml_tensor * add1 = ffml_op(FFML_OP_ADD, doubled, one);
    ffml_set_name(add1, "add1");

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(add1);

    ffml_cgraph_alloc(cgraph, pool);

    // sample fake data
    ffml_set_data(a, {0,0,0,0}, 1.0f);
    ffml_set_data(a, {0,0,1,0}, 2.0f);
    ffml_set_data(a, {0,0,2,0}, 3.0f);
    ffml_set_data(a, {0,0,3,0}, 4.0f);
    ffml_set_data(a, {0,1,0,0}, 5.0f);
    ffml_set_data(a, {0,1,1,0}, 6.0f);
    ffml_set_data(a, {0,1,2,0}, 7.0f);
    ffml_set_data(a, {0,1,3,0}, 8.0f);
    ffml_set_data(a, {0,2,0,0}, 9.0f);
    ffml_set_data(a, {0,2,1,0}, 10.0f);
    ffml_set_data(a, {0,2,2,0}, 11.0f);
    ffml_set_data(a, {0,2,3,0}, 12.0f);
    ffml_set_data(a, {0,3,0,0}, 13.0f);
    ffml_set_data(a, {0,3,1,0}, 14.0f);
    ffml_set_data(a, {0,3,2,0}, 15.0f);
    ffml_set_data(a, {0,3,3,0}, 16.0f);
    ffml_set_data(a, {1,0,0,0}, -1.0f);
    ffml_set_data(a, {1,0,1,0}, -2.0f);
    ffml_set_data(a, {1,0,2,0}, -3.0f);
    ffml_set_data(a, {1,0,3,0}, -4.0f);
    ffml_set_data(a, {1,1,0,0}, -5.0f);
    ffml_set_data(a, {1,1,1,0}, -6.0f);
    ffml_set_data(a, {1,1,2,0}, -7.0f);
    ffml_set_data(a, {1,1,3,0}, -8.0f);
    ffml_set_data(a, {1,2,0,0}, -9.0f);
    ffml_set_data(a, {1,2,1,0}, -10.0f);
    ffml_set_data(a, {1,2,2,0}, -11.0f);
    ffml_set_data(a, {1,2,3,0}, -12.0f);
    ffml_set_data(a, {1,3,0,0}, -13.0f);
    ffml_set_data(a, {1,3,1,0}, -14.0f);
    ffml_set_data(a, {1,3,2,0}, -15.0f);
    ffml_set_data(a, {1,3,3,0}, 1000.0f);

    ffml_set_data(two, {0,0,0,0}, 2.0f);

    ffml_set_data(one, {0,0,0,0}, 1.0f);
    
    //     forward pass
    ffml_cgraph_forward(cgraph);

    // ffml_debug_print_cgraph_data(cgraph);

    ffml_zerograd(cgraph);

    //     backward pass
    ffml_cgraph_backward(cgraph);
// 
    // ffml_debug_print_cgraph_data(cgraph);

    test_almost_equal(ffml_get_data(a, {0,0,0,0}), 1.0f);
    test_almost_equal(ffml_get_data(a, {0,0,1,0}), 2.0f);
    test_almost_equal(ffml_get_data(a, {0,0,2,0}), 3.0f);
    test_almost_equal(ffml_get_data(a, {0,0,3,0}), 4.0f);
    test_almost_equal(ffml_get_data(a, {0,1,0,0}), 5.0f);
    test_almost_equal(ffml_get_data(a, {0,1,1,0}), 6.0f);
    test_almost_equal(ffml_get_data(a, {0,1,2,0}), 7.0f);
    test_almost_equal(ffml_get_data(a, {0,1,3,0}), 8.0f);
    test_almost_equal(ffml_get_data(a, {0,2,0,0}), 9.0f);
    test_almost_equal(ffml_get_data(a, {0,2,1,0}), 10.0f);
    test_almost_equal(ffml_get_data(a, {0,2,2,0}), 11.0f);
    test_almost_equal(ffml_get_data(a, {0,2,3,0}), 12.0f);
    test_almost_equal(ffml_get_data(a, {0,3,0,0}), 13.0f);
    test_almost_equal(ffml_get_data(a, {0,3,1,0}), 14.0f);
    test_almost_equal(ffml_get_data(a, {0,3,2,0}), 15.0f);
    test_almost_equal(ffml_get_data(a, {0,3,3,0}), 16.0f);
    test_almost_equal(ffml_get_data(a, {1,0,0,0}), -1.0f);
    test_almost_equal(ffml_get_data(a, {1,0,1,0}), -2.0f);
    test_almost_equal(ffml_get_data(a, {1,0,2,0}), -3.0f);
    test_almost_equal(ffml_get_data(a, {1,0,3,0}), -4.0f);
    test_almost_equal(ffml_get_data(a, {1,1,0,0}), -5.0f);
    test_almost_equal(ffml_get_data(a, {1,1,1,0}), -6.0f);
    test_almost_equal(ffml_get_data(a, {1,1,2,0}), -7.0f);
    test_almost_equal(ffml_get_data(a, {1,1,3,0}), -8.0f);
    test_almost_equal(ffml_get_data(a, {1,2,0,0}), -9.0f);
    test_almost_equal(ffml_get_data(a, {1,2,1,0}), -10.0f);
    test_almost_equal(ffml_get_data(a, {1,2,2,0}), -11.0f);
    test_almost_equal(ffml_get_data(a, {1,2,3,0}), -12.0f);
    test_almost_equal(ffml_get_data(a, {1,3,0,0}), -13.0f);
    test_almost_equal(ffml_get_data(a, {1,3,1,0}), -14.0f);
    test_almost_equal(ffml_get_data(a, {1,3,2,0}), -15.0f);
    test_almost_equal(ffml_get_data(a, {1,3,3,0}), 1000.0f);

    test_almost_equal(ffml_get_data(two, {0,0,0,0}), 2.0f);

    test_almost_equal(ffml_get_data(one, {0,0,0,0}), 1.0f);

    test_almost_equal(ffml_get_data(doubled, {0,0,0,0}), 2.0f);
    test_almost_equal(ffml_get_data(doubled, {0,0,1,0}), 4.0f);
    test_almost_equal(ffml_get_data(doubled, {0,0,2,0}), 6.0f);
    test_almost_equal(ffml_get_data(doubled, {0,0,3,0}), 8.0f);
    test_almost_equal(ffml_get_data(doubled, {0,1,0,0}), 10.0f);
    test_almost_equal(ffml_get_data(doubled, {0,1,1,0}), 12.0f);
    test_almost_equal(ffml_get_data(doubled, {0,1,2,0}), 14.0f);
    test_almost_equal(ffml_get_data(doubled, {0,1,3,0}), 16.0f);
    test_almost_equal(ffml_get_data(doubled, {0,2,0,0}), 18.0f);
    test_almost_equal(ffml_get_data(doubled, {0,2,1,0}), 20.0f);
    test_almost_equal(ffml_get_data(doubled, {0,2,2,0}), 22.0f);
    test_almost_equal(ffml_get_data(doubled, {0,2,3,0}), 24.0f);
    test_almost_equal(ffml_get_data(doubled, {0,3,0,0}), 26.0f);
    test_almost_equal(ffml_get_data(doubled, {0,3,1,0}), 28.0f);
    test_almost_equal(ffml_get_data(doubled, {0,3,2,0}), 30.0f);
    test_almost_equal(ffml_get_data(doubled, {0,3,3,0}), 32.0f);
    test_almost_equal(ffml_get_data(doubled, {1,0,0,0}), -2.0f);
    test_almost_equal(ffml_get_data(doubled, {1,0,1,0}), -4.0f);
    test_almost_equal(ffml_get_data(doubled, {1,0,2,0}), -6.0f);
    test_almost_equal(ffml_get_data(doubled, {1,0,3,0}), -8.0f);
    test_almost_equal(ffml_get_data(doubled, {1,1,0,0}), -10.0f);
    test_almost_equal(ffml_get_data(doubled, {1,1,1,0}), -12.0f);
    test_almost_equal(ffml_get_data(doubled, {1,1,2,0}), -14.0f);
    test_almost_equal(ffml_get_data(doubled, {1,1,3,0}), -16.0f);
    test_almost_equal(ffml_get_data(doubled, {1,2,0,0}), -18.0f);
    test_almost_equal(ffml_get_data(doubled, {1,2,1,0}), -20.0f);
    test_almost_equal(ffml_get_data(doubled, {1,2,2,0}), -22.0f);
    test_almost_equal(ffml_get_data(doubled, {1,2,3,0}), -24.0f);
    test_almost_equal(ffml_get_data(doubled, {1,3,0,0}), -26.0f);
    test_almost_equal(ffml_get_data(doubled, {1,3,1,0}), -28.0f);
    test_almost_equal(ffml_get_data(doubled, {1,3,2,0}), -30.0f);
    test_almost_equal(ffml_get_data(doubled, {1,3,3,0}), 2000.0f);

    test_almost_equal(ffml_get_data(add1, {0,0,0,0}), 3.0f);
    test_almost_equal(ffml_get_data(add1, {0,0,1,0}), 5.0f);
    test_almost_equal(ffml_get_data(add1, {0,0,2,0}), 7.0f);
    test_almost_equal(ffml_get_data(add1, {0,0,3,0}), 9.0f);
    test_almost_equal(ffml_get_data(add1, {0,1,0,0}), 11.0f);
    test_almost_equal(ffml_get_data(add1, {0,1,1,0}), 13.0f);
    test_almost_equal(ffml_get_data(add1, {0,1,2,0}), 15.0f);
    test_almost_equal(ffml_get_data(add1, {0,1,3,0}), 17.0f);
    test_almost_equal(ffml_get_data(add1, {0,2,0,0}), 19.0f);
    test_almost_equal(ffml_get_data(add1, {0,2,1,0}), 21.0f);
    test_almost_equal(ffml_get_data(add1, {0,2,2,0}), 23.0f);
    test_almost_equal(ffml_get_data(add1, {0,2,3,0}), 25.0f);
    test_almost_equal(ffml_get_data(add1, {0,3,0,0}), 27.0f);
    test_almost_equal(ffml_get_data(add1, {0,3,1,0}), 29.0f);
    test_almost_equal(ffml_get_data(add1, {0,3,2,0}), 31.0f);
    test_almost_equal(ffml_get_data(add1, {0,3,3,0}), 33.0f);
    test_almost_equal(ffml_get_data(add1, {1,0,0,0}), -1.0f);
    test_almost_equal(ffml_get_data(add1, {1,0,1,0}), -3.0f);
    test_almost_equal(ffml_get_data(add1, {1,0,2,0}), -5.0f);
    test_almost_equal(ffml_get_data(add1, {1,0,3,0}), -7.0f);
    test_almost_equal(ffml_get_data(add1, {1,1,0,0}), -9.0f);
    test_almost_equal(ffml_get_data(add1, {1,1,1,0}), -11.0f);
    test_almost_equal(ffml_get_data(add1, {1,1,2,0}), -13.0f);
    test_almost_equal(ffml_get_data(add1, {1,1,3,0}), -15.0f);
    test_almost_equal(ffml_get_data(add1, {1,2,0,0}), -17.0f);
    test_almost_equal(ffml_get_data(add1, {1,2,1,0}), -19.0f);
    test_almost_equal(ffml_get_data(add1, {1,2,2,0}), -21.0f);
    test_almost_equal(ffml_get_data(add1, {1,2,3,0}), -23.0f);
    test_almost_equal(ffml_get_data(add1, {1,3,0,0}), -25.0f);
    test_almost_equal(ffml_get_data(add1, {1,3,1,0}), -27.0f);
    test_almost_equal(ffml_get_data(add1, {1,3,2,0}), -29.0f);
    test_almost_equal(ffml_get_data(add1, {1,3,3,0}), 2001.0f);

    // dimensions
    test_equal<int>(doubled->n_dims, 3);
    test_equal<int>(doubled->ne[0], 2);
    test_equal<int>(doubled->ne[1], 4);
    test_equal<int>(doubled->ne[2], 4);

    test_equal<int>(one->n_dims, 1);
    test_equal<int>(one->ne[0], 1);

    test_equal<int>(two->n_dims, 1);
    test_equal<int>(two->ne[0], 1);

    test_equal<int>(add1->n_dims, 3);
    test_equal<int>(add1->ne[0], 2);
    test_equal<int>(add1->ne[1], 4);
    test_equal<int>(add1->ne[2], 4);

    // gradients
    test_almost_equal_grad(add1, {0,0,0,0}, 1.0f);
    test_almost_equal_grad(add1, {0,0,1,0}, 1.0f);
    test_almost_equal_grad(add1, {0,0,2,0}, 1.0f);
    test_almost_equal_grad(add1, {0,0,3,0}, 1.0f);
    test_almost_equal_grad(add1, {0,1,0,0}, 1.0f);
    test_almost_equal_grad(add1, {0,1,1,0}, 1.0f);
    test_almost_equal_grad(add1, {0,1,2,0}, 1.0f);
    test_almost_equal_grad(add1, {0,1,3,0}, 1.0f);
    test_almost_equal_grad(add1, {0,2,0,0}, 1.0f);
    test_almost_equal_grad(add1, {0,2,1,0}, 1.0f);
    test_almost_equal_grad(add1, {0,2,2,0}, 1.0f);
    test_almost_equal_grad(add1, {0,2,3,0}, 1.0f);
    test_almost_equal_grad(add1, {0,3,0,0}, 1.0f);
    test_almost_equal_grad(add1, {0,3,1,0}, 1.0f);
    test_almost_equal_grad(add1, {0,3,2,0}, 1.0f);
    test_almost_equal_grad(add1, {0,3,3,0}, 1.0f);
    test_almost_equal_grad(add1, {1,0,0,0}, 1.0f);
    test_almost_equal_grad(add1, {1,0,1,0}, 1.0f);
    test_almost_equal_grad(add1, {1,0,2,0}, 1.0f);
    test_almost_equal_grad(add1, {1,0,3,0}, 1.0f);
    test_almost_equal_grad(add1, {1,1,0,0}, 1.0f);
    test_almost_equal_grad(add1, {1,1,1,0}, 1.0f);
    test_almost_equal_grad(add1, {1,1,2,0}, 1.0f);
    test_almost_equal_grad(add1, {1,1,3,0}, 1.0f);
    test_almost_equal_grad(add1, {1,2,0,0}, 1.0f);
    test_almost_equal_grad(add1, {1,2,1,0}, 1.0f);
    test_almost_equal_grad(add1, {1,2,2,0}, 1.0f);
    test_almost_equal_grad(add1, {1,2,3,0}, 1.0f);
    test_almost_equal_grad(add1, {1,3,0,0}, 1.0f);
    test_almost_equal_grad(add1, {1,3,1,0}, 1.0f);
    test_almost_equal_grad(add1, {1,3,2,0}, 1.0f);
    test_almost_equal_grad(add1, {1,3,3,0}, 1.0f);

    test_almost_equal_grad(one, {0,0,0,0}, 32.0f);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                test_almost_equal_grad(doubled, {i,j,k,0}, 1.0f);
            }
        }
    }

    // for each increase in source in +1, the result increases by +2 due to multiplication
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                test_almost_equal_grad(a, {i,j,k,0}, 2.0f);
            }
        }
    }

    // basically every time we increase this small broadcasted value by +1, it increases every single thing by +1, so the gradient is "every single thing"
    test_almost_equal_grad(two, {0,0,0,0}, 1.0f+2.0f+3.0f+4.0f+5.0f+6.0f+7.0f+8.0f+9.0f+10.0f+11.0f+12.0f+13.0f+14.0f+15.0f+16.0f-1.0f-2.0f-3.0f-4.0f-5.0f-6.0f-7.0f-8.0f-9.0f-10.0f-11.0f-12.0f-13.0f-14.0f-15.0f+1000.0f);

    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too
}

#endif