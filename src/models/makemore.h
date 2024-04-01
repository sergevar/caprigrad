#ifndef MAKEMORE_H
#define MAKEMORE_H

#include "../ffml/ffml.h"
#include "../helpers.h"

#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <cmath>

#include "../optim/SGD.h"
#include "../optim/SGDWithMomentum.h"
#include "../optim/RMSProp.h"
#include "../optim/Adam.h"
#include "../optim/AdaBelief.h"
#include "../optim/AdamW.h"
#include "../optim/LearningRateScheduler.h"

#include "../keyboard_events.h"

void makemore() {
    const int EMB_WIDTH = 4;
    
    const int N_EPOCHS = 100;

    printf("Loading makemore dataset...\n");
    std::string names = file_get_contents("../datasets/makemore_names.txt");
    std::vector<std::string> words = explode('\n', names);

    // print first 5 words
    for(int i = 0; i < 5; i++) {
        printf("%s\n", words[i].c_str());
    }

    // print len(words)
    printf("len(words): %lu\n", words.size());

    // build the vocabulary of characters and mappings to/from integers
    std::set<char> chars;
    for(int i = 0; i < words.size(); i++) {
        for(int j = 0; j < words[i].size(); j++) {
            chars.insert(words[i][j]);
        }
    }
    auto stoi = std::map<char, int>();
    auto itos = std::map<int, char>();
    // push '.' as the first
    chars.insert('.');
    stoi['.'] = 0;
    itos[0] = '.';
    int i = 0;
    for(auto it = chars.begin(); it != chars.end(); it++) {
        stoi[*it] = i;
        itos[i] = *it;
        i++;
    }
    uint64_t vocab_size = chars.size();
    
    // print stoi
    printf("stoi: ");
    for(auto it = stoi.begin(); it != stoi.end(); it++) {
        printf("%c:%d ", it->first, it->second);
    }

    // print itos
    printf("\nitos: ");
    for(auto it = itos.begin(); it != itos.end(); it++) {
        printf("%d:%c ", it->first, it->second);
    }
    printf("\n");

    // print vocab_size
    printf("vocab_size: %lu\n", vocab_size);

    // build the dataset
    const int BLOCK_SIZE = 3;
    auto build_dataset = [&words, &stoi, &itos, &vocab_size, &BLOCK_SIZE](int start, int end) {
        std::vector<std::vector<int>> xs;
        std::vector<int> ys;

        for(int i = start; i < end; i++) {
            std::vector<int> context;
            context.resize(BLOCK_SIZE);
            for(int j = 0; j < BLOCK_SIZE; j++) context[j] = 0;

            // for each char in the word + '.':
            std::string word = words[i] + ".";
            for(int j=0; j < word.size(); j++) {
                // get the char
                char ch = word[j];
                // get the index
                int ix = stoi[ch];
                // add context to xs
                xs.push_back(context);
                // add the index to ys
                ys.push_back(ix);
                // shift context - crop and append (context = context[1:] + [ix])
                for(int k = 0; k < BLOCK_SIZE - 1; k++) {
                    context[k] = context[k + 1];
                }
                context[BLOCK_SIZE - 1] = ix;
            }
        }

        return std::make_pair(xs, ys);
    };

    // seed random
    srand(0);

    // shuffle words
    std::random_shuffle(words.begin(), words.end());

    // build datasets
    uint64_t n1 = words.size() * 0.8f;
    uint64_t n2 = words.size() * 0.9f;

    auto train_set = build_dataset(0, n1);
    auto valid_set = build_dataset(n1, n2);
    auto test_set = build_dataset(n2, words.size());

    auto Xtr = train_set.first;
    auto Ytr = train_set.second;
    auto Xdev = valid_set.first;
    auto Ydev = valid_set.second;
    auto Xte = test_set.first;
    auto Yte = test_set.second;

    // MLP revisited
    const uint64_t n_embd = 10; // the dimensionality of the character embedding vectors
    const uint64_t n_hidden = 200; // the number of neurons in the hidden layer of the MLP
    const uint64_t batch_size = 320;

    // create a memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(1 * GB);

    // configure a model
    auto C = ffml_tensor_create(2, {vocab_size, n_embd, 0, 0}, "C");
    C->op = FFML_OP_INIT_RND_NORMAL;
    auto W1 = ffml_tensor_create(2, {n_embd * BLOCK_SIZE, n_hidden, 0, 0}, "W1");
    W1->op = FFML_OP_INIT_RND_NORMAL_KAIMING;
    W1->op_metadata->insert({"init_kaiming.fan_in", (float)(n_embd * BLOCK_SIZE)});
    W1->op_metadata->insert({"init_kaiming.gain", (5.0f/3.0f)});
    auto b1 = ffml_tensor_create(1, {n_hidden, 0, 0, 0}, "b1");
    b1->op = FFML_OP_INIT_RND_NORMAL;
    auto W2 = ffml_tensor_create(2, {n_hidden, vocab_size, 0, 0}, "W2");
    W2->op = FFML_OP_INIT_RND_NORMAL;    

    auto scale = ffml_tensor_create(1, {1, 0, 0, 0}, "scale"); // *0.01 to get better values at init
    scale->op = FFML_OP_INIT_FILL;
    scale->op_metadata->insert({"init_fill.value", 0.01f});
    
    auto b2 = ffml_tensor_create(1, {vocab_size, 0, 0, 0}, "b2");
    b2->op = FFML_OP_INIT_ZEROES;

    auto Xb = ffml_tensor_create(2, {batch_size, BLOCK_SIZE, 0, 0}, "Xb");
    auto Yb_onehot = ffml_tensor_create(2, {batch_size, vocab_size, 0, 0}, "Yb_onehot");

    auto embcat = ffml_op(FFML_OP_SELECT, Xb, C, "embcat");

    auto embcat_reshaped = ffml_reshape(embcat, 2, {batch_size, n_embd * BLOCK_SIZE, 0, 0});

    // F scaled
    auto scaled = [scale](ffml_tensor * to_scale) {
        return ffml_op(FFML_OP_MUL, to_scale, scale, (std::string(to_scale->name) + "_scaled").c_str());
    };

    auto hpreact = ffml_op(FFML_OP_ADD, ffml_op(FFML_OP_MATMUL, embcat_reshaped, W1, "embreshapedxxw1"), scaled(b1), "hpreact");
    auto h = ffml_unary_op(FFML_OP_TANH, hpreact, "h");
    auto logits = ffml_op(FFML_OP_ADD, ffml_op(FFML_OP_MATMUL, h, scaled(W2), "hxW2"), b2, "logits");
    auto loss = ffml_op(FFML_OP_SOFTMAX_CROSS_ENTROPY, logits, Yb_onehot, "loss");
    auto final_loss = ffml_unary_op(FFML_OP_MEAN, loss, "final_loss");

    std::vector<ffml_tensor*> parameters = {C, W1, b1, W2, b2};

    // print total parameters
    int total_params = 0;
    for(auto param: parameters) {
        total_params += param->nelem;
    }
    printf("total_params: %d\n", total_params);

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(final_loss);
    ffml_cgraph_alloc(cgraph, pool, true);

    KeyboardListenerGrad * keyboardListenerGrad = new KeyboardListenerGrad(cgraph);

    keyboardListenerGrad->attach('h', [&]() {
        print_histogram_of_tensor(h, 30);
        print_histogram_of_tensor(hpreact, 30);
        keyboardListenerGrad->pause();
    });

    keyboardListenerGrad->attach('t', [&]() {
        print_threshold_map(h, 0.99f, 10000);
        keyboardListenerGrad->pause();
    });

    keyboardListenerGrad->attach('y', [&]() {
        print_two_threshold_map(h, -0.99f, 0.99f, 70000);
        keyboardListenerGrad->pause();
    });


    // same optimization as last time
    const int max_steps = 200000;
    std::vector<float> lossi;

    // auto optim = new SGDWithMomentum(0.9f);
    // auto optim = new Adam(0.9f, 0.999f, 1e-8);
    // auto optim = new AdaBelief(0.9f, 0.999f, 1e-8);
    auto optim = new AdamW();
    optim->addParameters(parameters);

    auto lrs = new LearningRateScheduler();

    for (int i=0; i<max_steps; i++) {
        // minibatch construct
        std::vector<std::vector<int>> Xbatch;
        std::vector<int> Ybatch;
        for(int j = 0; j < batch_size; j++) {
            int ix = rand() % Xtr.size();
            Xbatch.push_back(Xtr[ix]);
            Ybatch.push_back(Ytr[ix]);
        }

        for(uint64_t j = 0; j < batch_size; j++) {
            for(uint64_t k = 0; k < BLOCK_SIZE; k++) {
                ffml_set_data(Xb, {j, k, 0, 0}, (float)Xbatch[j][k]);
            }

            // first clean
            for(uint64_t k = 0; k < vocab_size; k++) {
                ffml_set_data(Yb_onehot, {j, k, 0, 0}, 0.0f);
            }
            ffml_set_data(Yb_onehot, {j, (uint64_t)Ybatch[j], 0, 0}, 1.0f);
        }

        // forward pass
        ffml_cgraph_forward(cgraph);

        // backward pass
        ffml_zerograd(cgraph);
        ffml_cgraph_backward(cgraph);

        if (test_for_dead_neurons({h})) {
            keyboardListenerGrad->pause();
        }

        // ffml_debug_print_cgraph_data(cgraph);
        // exit(1);

        lossi.push_back(ffml_get_data(final_loss, {0,0,0,0}));

        // track stats
        if (i % 1 == 0) {
            printf("%d/%d: loss: %f ", i, max_steps, ffml_get_data(final_loss, {0,0,0,0}));

            // val loss
            int val_batch_size = batch_size;
            for(uint64_t j = 0; j < val_batch_size; j++) {
                for(uint64_t k = 0; k < BLOCK_SIZE; k++) {
                    ffml_set_data(Xb, {j, k, 0, 0}, (float)Xdev[j][k]);
                }

                // first clean
                for(uint64_t k = 0; k < vocab_size; k++) {
                    ffml_set_data(Yb_onehot, {j, k, 0, 0}, 0.0f);
                }
                ffml_set_data(Yb_onehot, {j, (uint64_t)Ydev[j], 0, 0}, 1.0f);
            }

            // forward pass
            ffml_cgraph_forward(cgraph);

            // get loss
            float val_loss = ffml_get_data(final_loss, {0,0,0,0});

            printf("val_loss: %f ", val_loss);
        }

        // sample
        if (i % 100 == 0) {
            printf("\n");

            std::vector<std::string> contexts;
            for(int sample=0; sample < batch_size; sample++) {
                std::string context = "";
                for(int q=0; q<BLOCK_SIZE; q++) context += ".";
                contexts.push_back(context);
            }

            // use the model to generate some text
            for(int q=0; q<30; q++) {
                for(uint64_t sample=0; sample < batch_size; sample++) {
                    // get the context
                    std::string context = contexts[sample].substr(contexts[sample].size() - BLOCK_SIZE, BLOCK_SIZE);
                    std::vector<int> context_ix;
                    for(int j = 0; j < BLOCK_SIZE; j++) {
                        context_ix.push_back(stoi[context[j]]);
                    }

                    // set the context
                    for(uint64_t j = 0; j < BLOCK_SIZE; j++) {
                        ffml_set_data(Xb, {sample, j, 0, 0}, (float)context_ix[j]);
                    }
                }

                // forward pass
                ffml_cgraph_forward(cgraph);

                for(uint64_t sample=0; sample < batch_size; sample++) {
                    // get the logits
                    std::vector<float> logits;
                    for(uint64_t j = 0; j < vocab_size; j++) {
                        logits.push_back(ffml_get_data(ffml_get_tensor_by_name(cgraph, "logits"), {sample, j, 0, 0}));
                    }

                    // turn logits into probabilities
                    std::vector<float> probabilities;
                    float sum = 0.0f;
                    for(int j = 0; j < vocab_size; j++) {
                        float p = std::exp(logits[j]);
                        probabilities.push_back(p);
                        sum += p;
                    }

                    // sample from the logits - multinomial
                    int ix = sample_multinomial(probabilities);

                    // get the char
                    char ch = itos[ix];

                    // append the char to the context
                    contexts[sample] += ch;
                }
            }

            // print the generated text
            const int max_words = 10;
            for(int sample=0; sample < std::min(batch_size, (uint64_t)max_words); sample++) {
                int start = 0;
                while (start < contexts[sample].size() && contexts[sample][start] == '.') start++;
                for (int j = start; j < contexts[sample].size(); j++) {
                    char ch = contexts[sample][j];
                    printf("%c", ch);
                    if (ch == '.') break;
                }
                printf("\n");
            }
        }

        // update
        // lrs->setSameBasedOnBatchSize((i < 500) ? 0.001f : 0.001f / 10.0f, batch_size);
        lrs->setSame((i < 500) ? 0.01f : 0.001f);
        const float LR = lrs->getLR();
        printf(" lr: %f", LR);
        optim->step(LR);

        // batch size
        printf(" batch_size: %d", batch_size);

        printf("\n");

        // keyboard listener
        keyboardListenerGrad->check();
    }


}

#endif