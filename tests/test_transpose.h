#ifndef TEST_TRANSPOSE_H
#define TEST_TRANSPOSE_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_transpose() {
    test_name("Test transpose");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    // MATRIX A   2x4
    // 1 2 3 4
    // 5 6 7 8

    // MATRIX B   3x4
    // 10 40 70 100
    // 20 50 80 110
    // 30 60 90 120

    // MATRIX BT  4x3
    // 10 20 30
    // 40 50 60
    // 70 80 90
    // 100 110 120

    // RESULT   2x3
    // 700 800 900
    // 1580 1840 2100


    // BACKPROP

    // t.src0 (2x4) = t.dst (2x3) @ t.src1.T (3x4)
    // t.src1 (4x3) = t.src0.T (4x2) @ t.dst (2x3)

    ffml_tensor * a_2d = ffml_tensor_create(2, {2,4,0,0}, "a_2d");
    ffml_tensor * b_2d = ffml_tensor_create(2, {3,4,0,0}, "b_2d");

    ffml_tensor * bt_2d = ffml_unary_op(FFML_OP_TRANSPOSE, b_2d);
    ffml_set_name(bt_2d, "bt_2d");

    ffml_tensor * matmul_2d = ffml_op(FFML_OP_MATMUL, a_2d, bt_2d);
    ffml_set_name(matmul_2d, "matmul_2d");

    ffml_tensor * c_2d = ffml_tensor_create(2, {2,3,0,0}, "c_2d");

    ffml_tensor * normal_mul_2d = ffml_op(FFML_OP_MUL, matmul_2d, c_2d);
    ffml_set_name(normal_mul_2d, "normal_mul_2d");

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(normal_mul_2d);

    ffml_cgraph_alloc(cgraph, pool);

    test_equal<float>(a_2d->size_bytes, 2*4 * sizeof(float));
    test_equal<float>(b_2d->size_bytes, 4*3 * sizeof(float));
    test_equal<float>(matmul_2d->size_bytes, 2*3 * sizeof(float));

    // sample fake data
    ffml_set_data(a_2d, {0,0,0,0}, 1.0f);
    ffml_set_data(a_2d, {0,1,0,0}, 2.0f);
    ffml_set_data(a_2d, {0,2,0,0}, 3.0f);
    ffml_set_data(a_2d, {0,3,0,0}, 4.0f);

    ffml_set_data(a_2d, {1,0,0,0}, 5.0f);
    ffml_set_data(a_2d, {1,1,0,0}, 6.0f);
    ffml_set_data(a_2d, {1,2,0,0}, 7.0f);
    ffml_set_data(a_2d, {1,3,0,0}, 8.0f);

    // ffml_set_data(b_2d, {0,0,0,0}, 10.0f);
    // ffml_set_data(b_2d, {0,1,0,0}, 20.0f);
    // ffml_set_data(b_2d, {0,2,0,0}, 30.0f);

    // ffml_set_data(b_2d, {1,0,0,0}, 40.0f);
    // ffml_set_data(b_2d, {1,1,0,0}, 50.0f);
    // ffml_set_data(b_2d, {1,2,0,0}, 60.0f);

    // ffml_set_data(b_2d, {2,0,0,0}, 70.0f);
    // ffml_set_data(b_2d, {2,1,0,0}, 80.0f);
    // ffml_set_data(b_2d, {2,2,0,0}, 90.0f);

    // ffml_set_data(b_2d, {3,0,0,0}, 100.0f);
    // ffml_set_data(b_2d, {3,1,0,0}, 110.0f);
    // ffml_set_data(b_2d, {3,2,0,0}, 120.0f);

    ffml_set_data(b_2d, {0,0,0,0}, 10.0f);
    ffml_set_data(b_2d, {0,1,0,0}, 40.0f);
    ffml_set_data(b_2d, {0,2,0,0}, 70.0f);
    ffml_set_data(b_2d, {0,3,0,0}, 100.0f);

    ffml_set_data(b_2d, {1,0,0,0}, 20.0f);
    ffml_set_data(b_2d, {1,1,0,0}, 50.0f);
    ffml_set_data(b_2d, {1,2,0,0}, 80.0f);
    ffml_set_data(b_2d, {1,3,0,0}, 110.0f);

    ffml_set_data(b_2d, {2,0,0,0}, 30.0f);
    ffml_set_data(b_2d, {2,1,0,0}, 60.0f);
    ffml_set_data(b_2d, {2,2,0,0}, 90.0f);
    ffml_set_data(b_2d, {2,3,0,0}, 120.0f);

    ffml_set_data(c_2d, {0,0,0,0}, 11.0f);
    ffml_set_data(c_2d, {0,1,0,0}, 22.0f);
    ffml_set_data(c_2d, {0,2,0,0}, 33.0f);

    ffml_set_data(c_2d, {1,0,0,0}, 44.0f);
    ffml_set_data(c_2d, {1,1,0,0}, 55.0f);
    ffml_set_data(c_2d, {1,2,0,0}, 66.0f);

    //     forward pass
    ffml_cgraph_forward(cgraph);


    ffml_zerograd(cgraph);

    test_equal(ffml_get_data(a_2d, {0,0,0,0}), 1.0f);
    test_equal(ffml_get_data(a_2d, {0,1,0,0}), 2.0f);
    test_equal(ffml_get_data(a_2d, {0,2,0,0}), 3.0f);
    test_equal(ffml_get_data(a_2d, {0,3,0,0}), 4.0f);
    test_equal(ffml_get_data(a_2d, {1,0,0,0}), 5.0f);
    test_equal(ffml_get_data(a_2d, {1,1,0,0}), 6.0f);
    test_equal(ffml_get_data(a_2d, {1,2,0,0}), 7.0f);
    test_equal(ffml_get_data(a_2d, {1,3,0,0}), 8.0f);

    test_equal(ffml_get_data(b_2d, {0,0,0,0}), 10.0f);
    test_equal(ffml_get_data(b_2d, {0,1,0,0}), 40.0f);
    test_equal(ffml_get_data(b_2d, {0,2,0,0}), 70.0f);
    test_equal(ffml_get_data(b_2d, {0,3,0,0}), 100.0f);
    test_equal(ffml_get_data(b_2d, {1,0,0,0}), 20.0f);
    test_equal(ffml_get_data(b_2d, {1,1,0,0}), 50.0f);
    test_equal(ffml_get_data(b_2d, {1,2,0,0}), 80.0f);
    test_equal(ffml_get_data(b_2d, {1,3,0,0}), 110.0f);
    test_equal(ffml_get_data(b_2d, {2,0,0,0}), 30.0f);
    test_equal(ffml_get_data(b_2d, {2,1,0,0}), 60.0f);
    test_equal(ffml_get_data(b_2d, {2,2,0,0}), 90.0f);
    test_equal(ffml_get_data(b_2d, {2,3,0,0}), 120.0f);

    test_equal(ffml_get_data(c_2d, {0,0,0,0}), 11.0f);
    test_equal(ffml_get_data(c_2d, {0,1,0,0}), 22.0f);
    test_equal(ffml_get_data(c_2d, {0,2,0,0}), 33.0f);
    test_equal(ffml_get_data(c_2d, {1,0,0,0}), 44.0f);
    test_equal(ffml_get_data(c_2d, {1,1,0,0}), 55.0f);
    test_equal(ffml_get_data(c_2d, {1,2,0,0}), 66.0f);

    test_equal(ffml_get_data(matmul_2d, {0,0,0,0}), 700.0f);
    test_equal(ffml_get_data(matmul_2d, {0,1,0,0}), 800.0f);
    test_equal(ffml_get_data(matmul_2d, {0,2,0,0}), 900.0f);
    test_equal(ffml_get_data(matmul_2d, {1,0,0,0}), 1580.0f);
    test_equal(ffml_get_data(matmul_2d, {1,1,0,0}), 1840.0f);
    test_equal(ffml_get_data(matmul_2d, {1,2,0,0}), 2100.0f);

    test_equal(ffml_get_data(normal_mul_2d, {0,0,0,0}), 11.0f * 700.0f);
    test_equal(ffml_get_data(normal_mul_2d, {0,1,0,0}), 22.0f * 800.0f);
    test_equal(ffml_get_data(normal_mul_2d, {0,2,0,0}), 33.0f * 900.0f);
    test_equal(ffml_get_data(normal_mul_2d, {1,0,0,0}), 44.0f * 1580.0f);
    test_equal(ffml_get_data(normal_mul_2d, {1,1,0,0}), 55.0f * 1840.0f);
    test_equal(ffml_get_data(normal_mul_2d, {1,2,0,0}), 66.0f * 2100.0f);

    //     backward pass
    ffml_cgraph_backward(cgraph);
    // 

    test_almost_equal_grad(normal_mul_2d, {0,0,0,0}, 1.0f);
    test_almost_equal_grad(normal_mul_2d, {0,1,0,0}, 1.0f);
    test_almost_equal_grad(normal_mul_2d, {0,2,0,0}, 1.0f);
    test_almost_equal_grad(normal_mul_2d, {1,0,0,0}, 1.0f);
    test_almost_equal_grad(normal_mul_2d, {1,1,0,0}, 1.0f);
    test_almost_equal_grad(normal_mul_2d, {1,2,0,0}, 1.0f);

    test_almost_equal_grad(matmul_2d, {0,0,0,0}, 11.0f);
    test_almost_equal_grad(matmul_2d, {0,1,0,0}, 22.0f);
    test_almost_equal_grad(matmul_2d, {0,2,0,0}, 33.0f);
    test_almost_equal_grad(matmul_2d, {1,0,0,0}, 44.0f);
    test_almost_equal_grad(matmul_2d, {1,1,0,0}, 55.0f);
    test_almost_equal_grad(matmul_2d, {1,2,0,0}, 66.0f);

    // ffml_debug_print_cgraph_data(cgraph);

    // MATRIX A   2x4
    // 1 2 3 4
    // 5 6 7 8

    // MATRIX B   4x3
    // 10 20 30
    // 40 50 60
    // 70 80 90
    // 100 110 120

    // RESULT   2x3
    // 700 800 900
    // 1580 1840 2100


    // BACKPROP

    // t.src0 A (2x4) = t.dst GRAD (2x3) @ t.src1_B.T (3x4)
    // t.src1 B (4x3) = t.src0_A.T (4x2) @ t.dst GRAD (2x3)

    // A GRAD
    // 1540 3520 5500 7480
    // 3520 8470 13420 18370

    // B GRAD
    // 231 297 363
    // 286 374 462
    // 341 451 561
    // 396 528 660
    
    test_almost_equal_grad(a_2d, {0,0,0,0}, 1540.0f);
    test_almost_equal_grad(a_2d, {0,1,0,0}, 3520.0f);
    test_almost_equal_grad(a_2d, {0,2,0,0}, 5500.0f);
    test_almost_equal_grad(a_2d, {0,3,0,0}, 7480.0f);

    test_almost_equal_grad(a_2d, {1,0,0,0}, 3520.0f);
    test_almost_equal_grad(a_2d, {1,1,0,0}, 8470.0f);
    test_almost_equal_grad(a_2d, {1,2,0,0}, 13420.0f);
    test_almost_equal_grad(a_2d, {1,3,0,0}, 18370.0f);


    test_almost_equal_grad(bt_2d, {0,0,0,0}, 231.0f);
    test_almost_equal_grad(bt_2d, {0,1,0,0}, 297.0f);
    test_almost_equal_grad(bt_2d, {0,2,0,0}, 363.0f);

    test_almost_equal_grad(bt_2d, {1,0,0,0}, 286.0f);
    test_almost_equal_grad(bt_2d, {1,1,0,0}, 374.0f);
    test_almost_equal_grad(bt_2d, {1,2,0,0}, 462.0f);

    test_almost_equal_grad(bt_2d, {2,0,0,0}, 341.0f);
    test_almost_equal_grad(bt_2d, {2,1,0,0}, 451.0f);
    test_almost_equal_grad(bt_2d, {2,2,0,0}, 561.0f);

    test_almost_equal_grad(bt_2d, {3,0,0,0}, 396.0f);
    test_almost_equal_grad(bt_2d, {3,1,0,0}, 528.0f);
    test_almost_equal_grad(bt_2d, {3,2,0,0}, 660.0f);

    test_almost_equal_grad(b_2d, {0,0,0,0}, 231.0f);
    test_almost_equal_grad(b_2d, {1,0,0,0}, 297.0f);
    test_almost_equal_grad(b_2d, {2,0,0,0}, 363.0f);

    test_almost_equal_grad(b_2d, {0,1,0,0}, 286.0f);
    test_almost_equal_grad(b_2d, {1,1,0,0}, 374.0f);
    test_almost_equal_grad(b_2d, {2,1,0,0}, 462.0f);

    test_almost_equal_grad(b_2d, {0,2,0,0}, 341.0f);
    test_almost_equal_grad(b_2d, {1,2,0,0}, 451.0f);
    test_almost_equal_grad(b_2d, {2,2,0,0}, 561.0f);
    

    // ffml_debug_print_cgraph_data(cgraph);

    

    //     update weights (optimizer)

    //     evaluate

    // save model


    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too
}

#endif