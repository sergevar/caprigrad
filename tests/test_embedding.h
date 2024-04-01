#ifndef TEST_EMBEDDING_H
#define TEST_EMBEDDING_H

#include "../src/ffml/ffml.h"
#include "common.h"

void test_embedding() {
    test_name("Testing Embedding");
        // 

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    const int batch_size = 3;
    const int BLOCK_SIZE = 5;
    const int vocab_size = 27;
    const int embedding_size = 4;

    auto Xb = ffml_tensor_create(2, {batch_size, BLOCK_SIZE, 0, 0}, "Xb"); // (batch_size, BLOCK_SIZE)
    auto Yb_onehot = ffml_tensor_create(2, {batch_size, vocab_size, 0, 0}, "Yb"); // (batch_size, vocab_size)

    auto C = ffml_tensor_create(2, {vocab_size, embedding_size, 0, 0}, "C"); // (vocab_size, embedding_size)
    C->op = FFML_OP_INIT_RND_NORMAL;
    
    auto emb = ffml_op(FFML_OP_SELECT, Xb, C); // (batch_size, BLOCK_SIZE, embedding_size)
    ffml_set_name(emb, "emb");

    auto embprojected = ffml_reshape(emb, 2, {batch_size, BLOCK_SIZE * embedding_size, 0, 0}); // (batch_size, BLOCK_SIZE * embedding_size)
    ffml_set_name(embprojected, "embprojected");

    auto W1 = ffml_tensor_create(2, {BLOCK_SIZE * embedding_size, 10, 0, 0}, "W1"); // (BLOCK_SIZE * embedding_size, 10)
    W1->op = FFML_OP_INIT_RND_NORMAL;
    auto b1 = ffml_tensor_create(1, {10, 0, 0, 0}, "b1"); // (10)
    b1->op = FFML_OP_INIT_RND_NORMAL;
    auto embprojxW1 = ffml_op(FFML_OP_MATMUL, embprojected, W1);
    ffml_set_name(embprojxW1, "embprojxW1");
    auto out = ffml_op(FFML_OP_ADD, embprojxW1, b1); // (batch_size, 10)
    ffml_set_name(out, "out");

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(out);
    ffml_cgraph_alloc(cgraph, pool, true);

    ffml_set_data(Xb, {0,0,0,0}, 3.0f);
    ffml_set_data(Xb, {0,1,0,0}, 3.0f);
    ffml_set_data(Xb, {0,2,0,0}, 6.0f);
    ffml_set_data(Xb, {0,3,0,0}, 6.0f);
    ffml_set_data(Xb, {0,4,0,0}, 9.0f);

    ffml_set_data(Xb, {1,0,0,0}, 13.0f);
    ffml_set_data(Xb, {1,1,0,0}, 13.0f);
    ffml_set_data(Xb, {1,2,0,0}, 16.0f);
    ffml_set_data(Xb, {1,3,0,0}, 16.0f);
    ffml_set_data(Xb, {1,4,0,0}, 19.0f);

    ffml_set_data(Xb, {2,0,0,0}, 23.0f);
    ffml_set_data(Xb, {2,1,0,0}, 23.0f);
    ffml_set_data(Xb, {2,2,0,0}, 26.0f);
    ffml_set_data(Xb, {2,3,0,0}, 26.0f);
    ffml_set_data(Xb, {2,4,0,0}, 1.0f);

    for(uint64_t i = 0; i < vocab_size; i++) {
        for(uint64_t j = 0; j < embedding_size; j++) {
            ffml_set_data(C, {i,j,0,0}, i * 0.1f + j * 0.01f);
        }
    }
    C->init_ran = true;

    for(uint64_t i = 0; i < 20; i++) {
        for(uint64_t j = 0; j < 10; j++) {
            ffml_set_data(W1, {i,j,0,0}, i * 1.0f + j * 1.0f);
        }
    }
    W1->init_ran = true;

    for(uint64_t j = 0; j < 10; j++) {
        ffml_set_data(b1, {j,0,0,0}, -1 * ((float)j) - 1.0f);
    }
    b1->init_ran = true;

    ffml_cgraph_forward(cgraph);

    ffml_zerograd(cgraph);

    ffml_cgraph_backward(cgraph);

    // ffml_debug_print_cgraph_data(cgraph);

    test_tensor_data_flat_almost_equal(Xb, {3.0f, 3.0f, 6.0f, 6.0f, 9.0f, 13.0f, 13.0f, 16.0f, 16.0f, 19.0f, 23.0f, 23.0f, 26.0f, 26.0f, 1.0f});
    test_tensor_data_flat_almost_equal(C, {
        0.0f, 0.01f, 0.02f, 0.03f,
        0.1f, 0.11f, 0.12f, 0.13f,
        0.2f, 0.21f, 0.22f, 0.23f,
        0.3f, 0.31f, 0.32f, 0.33f,
        0.4f, 0.41f, 0.42f, 0.43f,
        0.5f, 0.51f, 0.52f, 0.53f,
        0.6f, 0.61f, 0.62f, 0.63f,
        0.7f, 0.71f, 0.72f, 0.73f,
        0.8f, 0.81f, 0.82f, 0.83f,
        0.9f, 0.91f, 0.92f, 0.93f,
        1.0f, 1.01f, 1.02f, 1.03f,
        1.1f, 1.11f, 1.12f, 1.13f,
        1.2f, 1.21f, 1.22f, 1.23f,
        1.3f, 1.31f, 1.32f, 1.33f,
        1.4f, 1.41f, 1.42f, 1.43f,
        1.5f, 1.51f, 1.52f, 1.53f,
        1.6f, 1.61f, 1.62f, 1.63f,
        1.7f, 1.71f, 1.72f, 1.73f,
        1.8f, 1.81f, 1.82f, 1.83f,
        1.9f, 1.91f, 1.92f, 1.93f,
        2.0f, 2.01f, 2.02f, 2.03f,
        2.1f, 2.11f, 2.12f, 2.13f, });

    test_tensor_data_flat_almost_equal(emb, {
        0.3f, 0.31f, 0.32f, 0.33f,
        0.3f, 0.31f, 0.32f, 0.33f,
        0.6f, 0.61f, 0.62f, 0.63f,
        0.6f, 0.61f, 0.62f, 0.63f,
        0.9f, 0.91f, 0.92f, 0.93f,
        1.3f, 1.31f, 1.32f, 1.33f,
        1.3f, 1.31f, 1.32f, 1.33f,
        1.6f, 1.61f, 1.62f, 1.63f,
        1.6f, 1.61f, 1.62f, 1.63f,
        1.9f, 1.91f, 1.92f, 1.93f,
        2.3f, 2.31f, 2.32f, 2.33f,
        2.3f, 2.31f, 2.32f, 2.33f,
        2.6f, 2.61f, 2.62f, 2.63f,
        2.6f, 2.61f, 2.62f, 2.63f,
        0.1f, 0.11f, 0.12f, 0.13f,
    });

    test_almost_equal(ffml_get_data(embprojected, {0,0,0,0}), 0.3f);
    test_almost_equal(ffml_get_data(embprojected, {0,1,0,0}), 0.31f);
    test_almost_equal(ffml_get_data(embprojected, {0,2,0,0}), 0.32f);
    test_almost_equal(ffml_get_data(embprojected, {0,3,0,0}), 0.33f);
    test_almost_equal(ffml_get_data(embprojected, {0,4,0,0}), 0.3f);
    test_almost_equal(ffml_get_data(embprojected, {0,5,0,0}), 0.31f);
    test_almost_equal(ffml_get_data(embprojected, {0,6,0,0}), 0.32f);
    test_almost_equal(ffml_get_data(embprojected, {0,7,0,0}), 0.33f);
    test_almost_equal(ffml_get_data(embprojected, {0,8,0,0}), 0.6f);
    test_almost_equal(ffml_get_data(embprojected, {0,9,0,0}), 0.61f);
    test_almost_equal(ffml_get_data(embprojected, {0,10,0,0}), 0.62f);
    test_almost_equal(ffml_get_data(embprojected, {0,11,0,0}), 0.63f);
    test_almost_equal(ffml_get_data(embprojected, {0,12,0,0}), 0.6f);
    test_almost_equal(ffml_get_data(embprojected, {0,13,0,0}), 0.61f);
    test_almost_equal(ffml_get_data(embprojected, {0,14,0,0}), 0.62f);
    test_almost_equal(ffml_get_data(embprojected, {0,15,0,0}), 0.63f);
    test_almost_equal(ffml_get_data(embprojected, {0,16,0,0}), 0.9f);
    test_almost_equal(ffml_get_data(embprojected, {0,17,0,0}), 0.91f);
    test_almost_equal(ffml_get_data(embprojected, {0,18,0,0}), 0.92f);
    test_almost_equal(ffml_get_data(embprojected, {0,19,0,0}), 0.93f);

    test_almost_equal(ffml_get_data(embprojected, {1,0,0,0}), 1.3f);
    test_almost_equal(ffml_get_data(embprojected, {1,1,0,0}), 1.31f);
    test_almost_equal(ffml_get_data(embprojected, {1,2,0,0}), 1.32f);
    test_almost_equal(ffml_get_data(embprojected, {1,3,0,0}), 1.33f);
    test_almost_equal(ffml_get_data(embprojected, {1,4,0,0}), 1.3f);
    test_almost_equal(ffml_get_data(embprojected, {1,5,0,0}), 1.31f);
    test_almost_equal(ffml_get_data(embprojected, {1,6,0,0}), 1.32f);
    test_almost_equal(ffml_get_data(embprojected, {1,7,0,0}), 1.33f);
    test_almost_equal(ffml_get_data(embprojected, {1,8,0,0}), 1.6f);
    test_almost_equal(ffml_get_data(embprojected, {1,9,0,0}), 1.61f);
    test_almost_equal(ffml_get_data(embprojected, {1,10,0,0}), 1.62f);
    test_almost_equal(ffml_get_data(embprojected, {1,11,0,0}), 1.63f);
    test_almost_equal(ffml_get_data(embprojected, {1,12,0,0}), 1.6f);
    test_almost_equal(ffml_get_data(embprojected, {1,13,0,0}), 1.61f);
    test_almost_equal(ffml_get_data(embprojected, {1,14,0,0}), 1.62f);
    test_almost_equal(ffml_get_data(embprojected, {1,15,0,0}), 1.63f);
    test_almost_equal(ffml_get_data(embprojected, {1,16,0,0}), 1.9f);
    test_almost_equal(ffml_get_data(embprojected, {1,17,0,0}), 1.91f);
    test_almost_equal(ffml_get_data(embprojected, {1,18,0,0}), 1.92f);
    test_almost_equal(ffml_get_data(embprojected, {1,19,0,0}), 1.93f);

    test_almost_equal(ffml_get_data(embprojected, {2,0,0,0}), 2.3f);
    test_almost_equal(ffml_get_data(embprojected, {2,1,0,0}), 2.31f);
    test_almost_equal(ffml_get_data(embprojected, {2,2,0,0}), 2.32f);
    test_almost_equal(ffml_get_data(embprojected, {2,3,0,0}), 2.33f);
    test_almost_equal(ffml_get_data(embprojected, {2,4,0,0}), 2.3f);
    test_almost_equal(ffml_get_data(embprojected, {2,5,0,0}), 2.31f);
    test_almost_equal(ffml_get_data(embprojected, {2,6,0,0}), 2.32f);
    test_almost_equal(ffml_get_data(embprojected, {2,7,0,0}), 2.33f);
    test_almost_equal(ffml_get_data(embprojected, {2,8,0,0}), 2.6f);
    test_almost_equal(ffml_get_data(embprojected, {2,9,0,0}), 2.61f);
    test_almost_equal(ffml_get_data(embprojected, {2,10,0,0}), 2.62f);
    test_almost_equal(ffml_get_data(embprojected, {2,11,0,0}), 2.63f);
    test_almost_equal(ffml_get_data(embprojected, {2,12,0,0}), 2.6f);
    test_almost_equal(ffml_get_data(embprojected, {2,13,0,0}), 2.61f);
    test_almost_equal(ffml_get_data(embprojected, {2,14,0,0}), 2.62f);
    test_almost_equal(ffml_get_data(embprojected, {2,15,0,0}), 2.63f);
    test_almost_equal(ffml_get_data(embprojected, {2,16,0,0}), 0.1f);
    test_almost_equal(ffml_get_data(embprojected, {2,17,0,0}), 0.11f);
    test_almost_equal(ffml_get_data(embprojected, {2,18,0,0}), 0.12f);
    test_almost_equal(ffml_get_data(embprojected, {2,19,0,0}), 0.13f);

    for(int i=0; i<batch_size*BLOCK_SIZE*embedding_size; i++) {
        test_almost_equal(ffml_get_grad_flat(b1, i), (float)batch_size); // 3.0f, because the same bias element is getting added to 3 rows due to broadcasting
        test_almost_equal(ffml_get_grad_flat(embprojxW1, i), 1.0f); // simply passing the gradient due to addition
        test_almost_equal(ffml_get_grad_flat(out, i), 1.0f);

        test_almost_equal(ffml_get_grad_flat(Xb, i), 0.0f); // shouldn't receive any grad when selecting

        // test_almost_equal(ffml_get_grad_flat(W1, i), ffml_get_data_flat(embprojected, i));
        // test_almost_equal(ffml_get_grad_flat(embprojected, i), ffml_get_data_flat(W1, i));
    }

    test_tensor_data_flat_almost_equal(emb, {
        0.3f, 0.31f, 0.32f, 0.33f, 0.3f, 0.31f, 0.32f, 0.33f, 0.6f, 0.61f, 0.62f, 0.63f, 0.6f, 0.61f, 0.62f, 0.63f, 0.9f, 0.91f, 0.92f, 0.93f, 1.3f, 1.31f, 1.32f, 1.33f, 1.30f, 1.31f, 1.32f, 1.33f, 1.6f, 1.61f, 1.62f, 1.63f, 1.6f, 1.61f, 1.62f, 1.63f, 1.9f, 1.91f, 1.92f, 1.93f, 2.3f, 2.31f, 2.32f, 2.33f, 2.3f, 2.31f, 2.32f, 2.33f, 2.6f, 2.61f, 2.62f, 2.63f, 2.6f, 2.61f, 2.62f, 2.63f, 0.1f, 0.11f, 0.12f, 0.13f,
    });
    // emb shape
    test_equal<int>(emb->n_dims, 3);
    test_equal<int>(emb->ne[0], batch_size);
    test_equal<int>(emb->ne[1], BLOCK_SIZE);
    test_equal<int>(emb->ne[2], embedding_size); 

    test_tensor_data_flat_almost_equal(embprojected, {
0.3f, 0.31f, 0.32f, 0.33f, 0.3f, 0.31f, 0.32f, 0.33f, 0.6f, 0.61f, 0.62f, 0.63f, 0.6f, 0.61f, 0.62f, 0.63f, 0.9f, 0.91f, 0.92f, 0.93f, 1.3f, 1.31f, 1.32f, 1.33f, 1.30f, 1.31f, 1.32f, 1.33f, 1.6f, 1.61f, 1.62f, 1.63f, 1.6f, 1.61f, 1.62f, 1.63f, 1.9f, 1.91f, 1.92f, 1.93f, 2.3f, 2.31f, 2.32f, 2.33f, 2.3f, 2.31f, 2.32f, 2.33f, 2.6f, 2.61f, 2.62f, 2.63f, 2.6f, 2.61f, 2.62f, 2.63f, 0.1f, 0.11f, 0.12f, 0.13f,
    });
    // embprojected shape
    test_equal<int>(embprojected->n_dims, 2);
    test_equal<int>(embprojected->ne[0], batch_size);
    test_equal<int>(embprojected->ne[1], BLOCK_SIZE * embedding_size);

    test_tensor_data_flat_almost_equal(embprojxW1, {
        129.7f, 140.8f, 151.9f, 163.0f, 174.1f, 185.2f, 196.3f, 207.4f, 218.5f, 229.6f, 319.7f, 350.8f, 381.9f, 413.0f, 444.1f, 475.2f, 506.3f, 537.4f, 568.5f, 599.6f, 313.7f, 353.6f, 393.5f, 433.4f, 473.3f, 513.2f, 553.1f, 593.0f, 632.9f, 672.8f
    });

    test_tensor_data_flat_almost_equal(out, {
        128.7f, 138.8f, 148.9f, 159.0f, 169.1f, 179.2f, 189.3f, 199.4f, 209.5f, 219.6f, 318.7f, 348.8f, 378.9f, 409.0f, 439.1f, 469.2f, 499.3f, 529.4f, 559.5f, 589.6f, 312.7f, 351.6f, 390.5f, 429.4f, 468.3f, 507.2f, 546.1f, 585.0f, 623.9f, 662.8f
    });

    test_tensor_grad_flat_almost_equal(W1, {
        3.9f, 3.9f, 3.9f, 3.9f, 3.9f, 3.9f, 3.9f, 3.9f, 3.9f, 3.9f,
        3.93f, 3.93f, 3.93f, 3.93f, 3.93f, 3.93f, 3.93f, 3.93f, 3.93f, 3.93f, 
        3.96f, 3.96f, 3.96f, 3.96f, 3.96f, 3.96f, 3.96f, 3.96f, 3.96f, 3.96f, 
        3.99f, 3.99f, 3.99f, 3.99f, 3.99f, 3.99f, 3.99f, 3.99f, 3.99f, 3.99f, 
        3.90f, 3.90f, 3.90f, 3.90f, 3.90f, 3.90f, 3.90f, 3.90f, 3.90f, 3.90f, 
        3.93f, 3.93f, 3.93f, 3.93f, 3.93f, 3.93f, 3.93f, 3.93f, 3.93f, 3.93f, 
        3.96f, 3.96f, 3.96f, 3.96f, 3.96f, 3.96f, 3.96f, 3.96f, 3.96f, 3.96f, 
        3.99f, 3.99f, 3.99f, 3.99f, 3.99f, 3.99f, 3.99f, 3.99f, 3.99f, 3.99f, 
        4.80f, 4.80f, 4.80f, 4.80f, 4.80f, 4.80f, 4.80f, 4.80f, 4.80f, 4.80f, 
        4.83f, 4.83f, 4.83f, 4.83f, 4.83f, 4.83f, 4.83f, 4.83f, 4.83f, 4.83f, 
        4.86f, 4.86f, 4.86f, 4.86f, 4.86f, 4.86f, 4.86f, 4.86f, 4.86f, 4.86f, 
        4.89f, 4.89f, 4.89f, 4.89f, 4.89f, 4.89f, 4.89f, 4.89f, 4.89f, 4.89f, 
        4.80f, 4.80f, 4.80f, 4.80f, 4.80f, 4.80f, 4.80f, 4.80f, 4.80f, 4.80f, 
        4.83f, 4.83f, 4.83f, 4.83f, 4.83f, 4.83f, 4.83f, 4.83f, 4.83f, 4.83f, 
        4.86f, 4.86f, 4.86f, 4.86f, 4.86f, 4.86f, 4.86f, 4.86f, 4.86f, 4.86f, 
        4.89f, 4.89f, 4.89f, 4.89f, 4.89f, 4.89f, 4.89f, 4.89f, 4.89f, 4.89f, 
        2.90f, 2.90f, 2.90f, 2.90f, 2.90f, 2.90f, 2.90f, 2.90f, 2.90f, 2.90f, 
        2.93f, 2.93f, 2.93f, 2.93f, 2.93f, 2.93f, 2.93f, 2.93f, 2.93f, 2.93f, 
        2.96f, 2.96f, 2.96f, 2.96f, 2.96f, 2.96f, 2.96f, 2.96f, 2.96f, 2.96f, 
        2.99f, 2.99f, 2.99f, 2.99f, 2.99f, 2.99f, 2.99f, 2.99f, 2.99f, 2.99f, 
    });

    test_tensor_grad_flat_almost_equal(embprojected, {
        45.0f, 55.0f, 65.0f, 75.0f, 85.0f, 95.0f, 105.0f, 115.0f, 125.0f, 135.0f, 145.0f, 155.0f, 165.0f, 175.0f, 185.0f, 195.0f, 205.0f, 215.0f, 225.0f, 235.0f,
        45.0f, 55.0f, 65.0f, 75.0f, 85.0f, 95.0f, 105.0f, 115.0f, 125.0f, 135.0f, 145.0f, 155.0f, 165.0f, 175.0f, 185.0f, 195.0f, 205.0f, 215.0f, 225.0f, 235.0f,
        45.0f, 55.0f, 65.0f, 75.0f, 85.0f, 95.0f, 105.0f, 115.0f, 125.0f, 135.0f, 145.0f, 155.0f, 165.0f, 175.0f, 185.0f, 195.0f, 205.0f, 215.0f, 225.0f, 235.0f,
    });

    test_tensor_grad_flat_almost_equal(emb, {
        45.0f, 55.0f, 65.0f, 75.0f, 85.0f, 95.0f, 105.0f, 115.0f, 125.0f, 135.0f, 145.0f, 155.0f, 165.0f, 175.0f, 185.0f, 195.0f, 205.0f, 215.0f, 225.0f, 235.0f,
        45.0f, 55.0f, 65.0f, 75.0f, 85.0f, 95.0f, 105.0f, 115.0f, 125.0f, 135.0f, 145.0f, 155.0f, 165.0f, 175.0f, 185.0f, 195.0f, 205.0f, 215.0f, 225.0f, 235.0f,
        45.0f, 55.0f, 65.0f, 75.0f, 85.0f, 95.0f, 105.0f, 115.0f, 125.0f, 135.0f, 145.0f, 155.0f, 165.0f, 175.0f, 185.0f, 195.0f, 205.0f, 215.0f, 225.0f, 235.0f,
    });

    test_tensor_grad_flat_almost_equal(C, {
        0.0f, 0.0f, 0.0f, 0.0f,
        205.0f, 215.0f, 225.0f, 235.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        130.0f, 150.0f, 170.0f, 190.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        290.0f, 310.0f, 330.0f, 350.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        205.0f, 215.0f, 225.0f, 235.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        130.0f, 150.0f, 170.0f, 190.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        290.0f, 310.0f, 330.0f, 350.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        205.0f, 215.0f, 225.0f, 235.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        130.0f, 150.0f, 170.0f, 190.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        290.0f, 310.0f, 330.0f, 350.0f,

    });

    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too

    // exit(0);
}

#endif