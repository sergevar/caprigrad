#ifndef SIMPLE_MLP_H
#define SIMPLE_MLP_H

#include "../ffml/ffml.h"
#include "../engine/Layer.h"
#include "../engine/MLP.h"

#include <vector>

#include <stdio.h>


void simple_mlp() {
    const int N_EPOCHS = 100;

    std::vector<std::vector<float>> xs;
    xs = {
        { 2.0f, 3.0f, -1.0f },
        { 3.0f, -1.0f, 0.5f },
        { 0.5f, 1.0f, 1.0f },
        { 1.0f, 1.0f, -1.0f }
    };

    std::vector<float> ys;
    ys = {
        1.0f,
        -1.0f,
        -1.0f,
        1.0f
    }; // desired targets


    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(10 * MB);

    // configure a model

    auto const2 = ffml_tensor_create(1, {1,0,0,0}, "2");

    auto mlp = new MLP(pool, xs[0].size(), {4, 4, 1});

    std::vector<ffml_tensor*> ypreds;

    std::vector<ffml_tensor*> input_tensors;
    std::vector<ffml_tensor*> y_tensors;
    std::vector<ffml_tensor*> ypred_tensors;

    std::vector<ffml_tensor*> losses;

    for(int i = 0; i < xs.size(); i++) {
        ffml_tensor * inputs = ffml_tensor_create(1, {xs[0].size(),0,0,0}, "inputs");
        input_tensors.push_back(inputs);

        ffml_tensor * y = ffml_tensor_create(1, {1,0,0,0}, "y");
        y_tensors.push_back(y);

        auto ypred = mlp->call(inputs);
        ypred_tensors.push_back(ypred);

        auto y_diff = ffml_op(FFML_OP_SUB, y, ypred);
        ffml_set_name(y_diff, "y_diff");

        auto loss = ffml_op(FFML_OP_POW, y_diff, const2);
        ffml_set_name(loss, "loss");

        losses.push_back(loss);
    }

    // sum losses
    ffml_tensor* total_loss = nullptr;
    for(int i = 0; i < losses.size(); i++) {
        if(total_loss == nullptr) {
            total_loss = losses[i];
        } else {
            total_loss = ffml_op(FFML_OP_ADD, total_loss, losses[i]);
        }
    }
    ffml_set_name(total_loss, "total_loss");

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(total_loss);

    ffml_cgraph_alloc(cgraph, pool);

    ffml_set_data(const2, {0,0,0,0}, 2.0f);

    // maybe load weights

    // configure data loader

    // configure optimizer

    for(uint64_t i = 0; i < xs.size(); i++) {
        //     load data
        for(uint64_t j = 0; j < xs[i].size(); j++) {
            ffml_set_data(input_tensors[i], {j, 0, 0, 0}, xs[i][j]);
        }
        ffml_set_data(y_tensors[i], {0, 0, 0, 0}, ys[i]);
    }

    // training loop:
    for (int i = 0; i < N_EPOCHS; i++) {

        //     forward pass
        ffml_cgraph_forward(cgraph);

        // ffml_debug_print_cgraph_data(cgraph);

        //    print loss
        printf("total loss: %f\n", ffml_get_data(total_loss, {0,0,0,0}));

        // print ypred
        printf("ypred: ");
        for(uint64_t i = 0; i < ypred_tensors.size(); i++) {
            printf("%f ", ffml_get_data(ypred_tensors[i], {0,0,0,0}));
        }
        printf("\n");

        ffml_zerograd(cgraph);

        //     backward pass
        ffml_cgraph_backward(cgraph);

        // ffml_debug_print_cgraph_data(cgraph);

        // ffml_debug_print_cgraph_data(cgraph);

        //     update weights (optimizer)
        const float LR = 0.01f;

        for(auto layer = mlp->begin(); layer != mlp->end(); layer++) {
            for(auto t_iter = (*layer)->begin(); t_iter != (*layer)->end(); t_iter++) {
                auto t = *t_iter;

                // printf("adjusting t: %s\n", t->name);

                for(uint64_t i = 0; i < t->nelem; i++) {
                    float current_data = ffml_get_data_flat(t, i);
                    float grad = ffml_get_grad_flat(t, i);
                    float update = -LR * grad;
                    float new_data = current_data + update;

                    ffml_set_data_flat(t, i, new_data);
                }

            }
        }

    }

    //     evaluate

    // save model

    // print hello world:
    printf("Hello, World!\n");

    // free memory pool/context
    ffml_memory_pool_destroy(pool);
    // todo: destroy tensors too

}

#endif