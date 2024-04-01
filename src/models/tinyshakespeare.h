#ifndef TINYSHAKESPEARE_H
#define TINYSHAKESPEARE_H

#include "../ffml/ffml.h"
#include "../engine/Layer.h"
#include "../engine/MLP.h"

#include <vector>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <map>

std::string shakespeare_path() {
    // check either 'datasets/' or '../datasets/'
    std::ifstream f("datasets/tinyshakespeare.txt");
    if(f.good()) {
        return "datasets/tinyshakespeare.txt";
    } else {
        std::ifstream f("../datasets/tinyshakespeare.txt");
        if(f.good()) {
            return "../datasets/tinyshakespeare.txt";
        } else {
            printf("ERROR: could not find datasets/tinyshakespeare.txt\n");
            exit(1);
        }
    }
}

void inference(ffml_cgraph * cgraph, std::map<char, int> char_to_index, std::map<int, char> index_to_char, std::vector<ffml_tensor*> input_tensors, std::vector<ffml_tensor*> logits_tensors, int PREV_WINDOW, const int ALPHABET_SIZE) {
    std::string text = "First Citi";

    const int HOW_MANY_TIMES = 60;


    for(int i = 0; i < HOW_MANY_TIMES; i++) {
        if (i > 0) {
            // exit(1);
        }

        // load data
        std::string minibatch_entry_text = text.substr(text.size() - PREV_WINDOW, PREV_WINDOW);

        // printf("minibatch_start_charidx: %d\n", minibatch_start_charidx);

        // first clear

        for(uint64_t j = 0; j < PREV_WINDOW; j++) {
            for(uint64_t k = 0; k < ALPHABET_SIZE; k++) {
                ffml_set_data(input_tensors[0], {k, j, 0, 0}, 0.0f);
            }
        }

        // print alphabet
        // printf("alphabet: ");
        // for(auto it = char_to_index.begin(); it != char_to_index.end(); it++) {
        //     // output character and its index
        //     printf("%c:%d ", it->first, it->second);
        // }
        // printf("\n");

        for(int j = 0; j < PREV_WINDOW; j++) {
            char c = minibatch_entry_text[j];
            int charidx = char_to_index[c];
            ffml_set_data(input_tensors[0], {charidx, j, 0, 0}, 1.0f);
        }

        //     forward pass

        ffml_cgraph_forward(cgraph);

        // ffml_debug_print_cgraph_data(cgraph);
        // exit(1);

        // take logits out
        std::vector<float> logits;
        logits.resize(ALPHABET_SIZE);
        for(int j = 0; j < logits_tensors[0]->ne[0]; j++) {
            logits[j] = ffml_get_data(logits_tensors[0], {j,0,0,0});
        }

        // // print logits
        // printf("logits: ");
        // for(int j = 0; j < ALPHABET_SIZE; j++) {
        //     printf("%f ", logits[j]);
        // }

        // get maximum value for numerical stability
        float max = logits[0];
        for(int j = 1; j < ALPHABET_SIZE; j++) {
            if(logits[j] > max) {
                max = logits[j];
            }
        }

        // softmax
        std::vector<float> softmax;
        float sum = 0.0f;
        for(int j = 0; j < ALPHABET_SIZE; j++) {
            softmax.push_back(exp(logits[j] - max));
            sum += softmax[j];
        }
        for(int j = 0; j < ALPHABET_SIZE; j++) {
            softmax[j] /= sum;
        }

        // // print softmax
        // printf("softmax: ");
        // for(int j = 0; j < ALPHABET_SIZE; j++) {
        //     printf("%f ", softmax[j]);
        // }

        // sample from softmax
        float r = (float)rand() / (float)RAND_MAX;
        float sum2 = 0.0f;
        int out_charidx = 0;
        for(int j = 0; j < ALPHABET_SIZE; j++) {
            sum2 += softmax[j];
            if(r < sum2) {
                out_charidx = j;
                break;
            }
        }

        // char with max probability
        int max_charidx = 0;
        float max_prob = softmax[0];
        for(int j = 1; j < ALPHABET_SIZE; j++) {
            if(softmax[j] > max_prob) {
                max_prob = softmax[j];
                max_charidx = j;
            }
        }

        // append to text
        // text += index_to_char[out_charidx];
        text += index_to_char[max_charidx];

        // printf("text: %s\n", text.c_str());

        // free memory

        // printf("text iteration %d: %s\n", i, text.c_str());
    }

    printf("text: %s\n", text.c_str());



}

void tinyshakespeare() {
    const int EMB_WIDTH = 4;
    
    const int N_EPOCHS = 100;

    // load datasets/tinyshakespeare.txt into a string
    std::ifstream f(shakespeare_path());
    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string text = buffer.str();

    // print first 100 chars
    printf("text: %s\n", text.substr(0, 100).c_str());

    // create a set of unique characters
    std::set<char> chars;
    for(int i = 0; i < text.size(); i++) {
        chars.insert(text[i]);
    }

    // create a map from char to index
    std::map<char, int> char_to_index;
    int i = 0;
    for(auto it = chars.begin(); it != chars.end(); it++) {
        char_to_index[*it] = i;
        i++;
    }

    // print char_to_index
    printf("char_to_index: ");
    for(auto it = chars.begin(); it != chars.end(); it++) {
        // output character and its index
        printf("%c:%d ", *it, char_to_index[*it]);
    }

    // create a map from index to char
    std::map<int, char> index_to_char;
    i = 0;
    for(auto it = chars.begin(); it != chars.end(); it++) {
        index_to_char[i] = *it;
        i++;
    }

    // output alphabet
    // printf("alphabet: ");
    // for(auto it = chars.begin(); it != chars.end(); it++) {
    //     // output character and its index
    //     printf("%c:%d ", *it, char_to_index[*it]);
    // }
    // printf("\n");

    const int PREV_WINDOW = 5;

    const int TEXT_SIZE = text.size();

    const int ALPHABET_SIZE = chars.size();

    const int MINI_BATCH_SIZE = 100;

    const int MINI_BATCHES_PER_BATCH = 40;

    // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(1 * GB);

    // configure a model

    auto mlp = new MLP(pool, PREV_WINDOW * EMB_WIDTH, {20,20,20, ALPHABET_SIZE});

    std::vector<ffml_tensor*> input_tensors;
    std::vector<ffml_tensor*> y_tensors;
    std::vector<ffml_tensor*> logits_tensors;

    std::vector<ffml_tensor*> losses_tensors;



    ffml_tensor * C = ffml_tensor_create(2, {ALPHABET_SIZE,EMB_WIDTH,0,0}, "C");
    C->op = FFML_OP_INIT_RND_UNIFORM;


    for(int i = 0; i < MINI_BATCH_SIZE; i++) {
        // std::string minibatch_entry_text = text.substr(minibatch_start_charidx + i, PREV_WINDOW);

        //

        ffml_tensor * inputs = ffml_tensor_create(2, {ALPHABET_SIZE,PREV_WINDOW,0,0}, "inputs");
        input_tensors.push_back(inputs);

        ffml_tensor * inputs_transposed = ffml_unary_op(FFML_OP_TRANSPOSE, inputs);

        // ffml_debug_print_tensor_metadata(inputs_transposed);

        ffml_tensor * emb = ffml_op(FFML_OP_MATMUL, inputs_transposed, C);

        // ffml_debug_print_tensor_metadata(emb);

        ffml_tensor * emb_reshaped = ffml_reshape(emb, 1, {PREV_WINDOW * EMB_WIDTH, 1, 0, 0});

        ffml_tensor * emb_reshaped_tanh = ffml_unary_op(FFML_OP_TANH, emb_reshaped);

        auto logits = mlp->call(emb_reshaped_tanh);
        logits_tensors.push_back(logits);



        ffml_tensor * y = ffml_tensor_create(1, {ALPHABET_SIZE,0,0,0}, "y");
        // ffml_debug_print_tensor_metadata(y);
        y_tensors.push_back(y);

        // auto ypred = ffml_op(FFML_OP_SOFTMAX, logits);

        // auto y_diff = ffml_op(FFML_OP_SUB, y, ypred);
        // ffml_set_name(y_diff, "y_diff");

        auto loss = ffml_op(FFML_OP_SOFTMAX_CROSS_ENTROPY, logits, y);
        ffml_set_name(loss, "loss");

        losses_tensors.push_back(loss);
    }

    // sum losses
    ffml_tensor* total_loss = nullptr;
    for (int i = 0; i < losses_tensors.size(); i++) {
        if (total_loss == nullptr) {
            total_loss = losses_tensors[i];
        } else {
            total_loss = ffml_op(FFML_OP_ADD, total_loss, losses_tensors[i]);
        }
    }

    ffml_tensor* mini_batch_size_tensor = ffml_tensor_create(1, {1,0,0,0}, "mini_batch_size");

    total_loss = ffml_op(FFML_OP_DIV, total_loss, mini_batch_size_tensor);

    ffml_set_name(total_loss, "crossentropy_loss");

    // todo: do the average!
//     // sum losses
// ffml_tensor* total_loss = nullptr;
// int total_samples = 0;

// for (int i = 0; i < losses.size(); i++) {
//     if (total_loss == nullptr) {
//         total_loss = losses[i];
//     } else {
//         total_loss = ffml_op(FFML_OP_ADD, total_loss, losses[i]);
//     }
//     total_samples += MINI_BATCH_SIZE;
// }

// // Compute the scaling factor for averaging the loss
// float scale_factor = 1.0f / total_samples;

// // Scale the total_loss by the scaling factor to get the average loss
// total_loss = ffml_op(FFML_OP_MUL, total_loss, ffml_tensor_create_scalar(scale_factor));

// // Store the total_loss tensor in the computation graph
// ffml_cgraph_set_output(cgraph, total_loss);


    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(total_loss);

    ffml_cgraph_alloc(cgraph, pool);

    ffml_set_data(mini_batch_size_tensor, {0,0,0,0}, MINI_BATCH_SIZE);

    // ffml_debug_print_cgraph_data(cgraph);

    uint64_t counter = 0;

    for(int epoch = 0; epoch < N_EPOCHS; epoch++) {
    for(int minibatch_start_charidx = 0; minibatch_start_charidx < TEXT_SIZE; minibatch_start_charidx+=MINI_BATCH_SIZE) {

        counter++;

        if (counter > 100) {
            // break;
        }

        // load data
        // for(uint64_t i = 0; i < xs.size(); i++) {
        //     //     load data
        //     for(uint64_t j = 0; j < xs[i].size(); j++) {
        //         ffml_set_data(input_tensors[i], {j, 0, 0, 0}, xs[i][j]);
        //     }
        //     ffml_set_data(y_tensors[i], {0, 0, 0, 0}, ys[i]);
        // }
        for(int i = 0; i < MINI_BATCH_SIZE; i++) {
            if(minibatch_start_charidx + i >= TEXT_SIZE) {
                break;
            }

            std::string minibatch_entry_text = text.substr(minibatch_start_charidx + i, PREV_WINDOW);

            // printf("minibatch_start_charidx: %d\n", minibatch_start_charidx);
            // printf("i: %d\n", i);
            // printf("minibatch_entry_text: %s\n", minibatch_entry_text.c_str());

            // first clear
            for(uint64_t j = 0; j < PREV_WINDOW; j++) {
                for(uint64_t k = 0; k < ALPHABET_SIZE; k++) {
                    ffml_set_data(input_tensors[i], {k, j, 0, 0}, 0.0f);
                }
            }
            for(int j = 0; j < PREV_WINDOW; j++) {
                char c = minibatch_entry_text[j];
                int charidx = char_to_index[c];
                ffml_set_data(input_tensors[i], {charidx, j, 0, 0}, 1.0f);
            }

            char c = text[minibatch_start_charidx + i + PREV_WINDOW];
            int charidx = char_to_index[c];
            // clear y
            for(uint64_t j = 0; j < ALPHABET_SIZE; j++) {
                ffml_set_data(y_tensors[i], {j, 0, 0, 0}, 0.0f);
            }
            ffml_set_data(y_tensors[i], {charidx, 0, 0, 0}, 1.0f);
            // printf("y: %c\n", c);
        }

        // TRAINING LOOP

        //     forward pass
        ffml_cgraph_forward(cgraph);

        // ffml_debug_print_cgraph_data(cgraph);

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

        // run inference
        if (counter % 1 == 0) {
            inference(cgraph, char_to_index, index_to_char, input_tensors, logits_tensors, PREV_WINDOW, ALPHABET_SIZE);

            //    print loss
            printf(" epoch: %d ", epoch);
            printf(" counter: %lu ", counter);
            printf(" total loss: %f", ffml_get_data(total_loss, {0,0,0,0}));

            // print ypred
            // printf("ypred: ");
            // for(uint64_t i = 0; i < ypred_tensors.size(); i++) {
            //     printf("%f ", ffml_get_data(ypred_tensors[i], {0,0,0,0}));
            // }
            printf("\n");
        }

        if (counter > 100) {
            counter = 0;
            break;
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