#ifndef KARPATHYGPT_H
#define KARPATHYGPT_H

#include <vector>
#include <string>
#include <iostream>

#include "../ffml/ffml.h"
#include "../helpers.h"
#include "../dataloaders/tokenizers/CharacterTokenizer.h"
#include "../engine/Module.h"
#include "../dataloaders/transform_functions.h"
#include "../dataloaders/DataSource.h"
#include "../dataloaders/DataLoader.h"
#include "../optim/Optimizer.h"
#include "../optim/AdamW.h"
#include "../keyboard_events.h"
#include "../optim/LearningRateScheduler.h"
// #include "../ui/app.h"
#include "../sockets/server.h"

ffml_tensor * scaled(ffml_tensor * src, float scale) {
    ffml_tensor * scalar = ffml_tensor_create(1, {1, 0, 0, 0}, ("scalar:" + std::to_string(scale)).c_str());
    scalar->op = FFML_OP_INIT_FILL;
    scalar->op_metadata->insert({"init_fill.value", scale});

    ffml_tensor * result = ffml_op(FFML_OP_MUL, src, scalar, (std::string(src->name) + "_scaled").c_str());
    return result;
}

class LinearLayer: public Module {
public:
    ffml_tensor* w;
    ffml_tensor* b;
    ffml_tensor* w_scaled;
    ffml_tensor* b_scaled;

    LinearLayer(const int in_features, const int out_features, float w_scaled = 1.0f, float b_scaled = 1.0f) {
        this->w = ffml_tensor_create(2, {in_features, out_features, 0, 0}, "w"); // (in_features, out_features)
        // this->w->op = FFML_OP_INIT_RND_NORMAL;
        this->w->op = FFML_OP_INIT_RND_NORMAL_KAIMING;
        this->w->op_metadata->insert({"init_kaiming.fan_in", (float)(in_features)});
        this->w->op_metadata->insert({"init_kaiming.gain", (5.0f/3.0f)});

        this->b = ffml_tensor_create(1, {out_features, 0, 0, 0}, "b"); // (out_features)
        this->b->op = FFML_OP_INIT_RND_NORMAL;

        this->w_scaled = scaled(this->w, w_scaled);

        this->b_scaled = scaled(this->b, b_scaled);

        this->parameters.push_back(this->w);
        this->parameters.push_back(this->b);
    }

    ffml_tensor* call(ffml_tensor* input) {
        ffml_tensor* matmul = ffml_op(FFML_OP_MATMUL, input, this->w_scaled, "matmul"); // (BATCH_SIZE, out_features)        

        ffml_tensor* matmul_plus_b = ffml_op(FFML_OP_ADD, matmul, this->b_scaled, "matmul_plus_b"); // (BATCH_SIZE, out_features)

        return matmul_plus_b;
    }
};

// class LayerNorm: public Module {
// public:
//     ffml_tensor* w;
//     ffml_tensor* b;

//     LayerNorm(const int n_hidden): Module() {
//         this->w = ffml_tensor_create(1, {n_hidden, 0, 0, 0}, "w"); // (n_hidden)
//         this->w->op = FFML_OP_INIT_RND_NORMAL;
//         this->w->op_metadata->insert({"init_rnd_normal.stddev", 0.02f});

//         this->b = ffml_tensor_create(1, {n_hidden, 0, 0, 0}, "b"); // (n_hidden)
//         this->b->op = FFML_OP_INIT_RND_NORMAL;
//         this->b->op_metadata->insert({"init_rnd_normal.stddev", 0.02f});

//         this->parameters.push_back(this->w);
//         this->parameters.push_back(this->b);
//     }

//     ffml_tensor* call(ffml_tensor* input) {
//         ffml_tensor* mean = ffml_unary_op(FFML_OP_MEAN_BATCHED
//         ffml_tensor* mean_reshaped = ffml_reshape(mean, 2, {mean->nelem, 1, 0, 0}); // (BATCH_SIZE * n_hidden, 1)

//         ffml_tensor* mean_subbed = ffml_op(FFML_OP_SUB, input, mean_reshaped, "mean_subbed"); // (BATCH_SIZE, n_hidden)

//         ffml_tensor* mean_subbed_squared = ffml_unary_op(FFML_OP_SQUARE, mean_subbed, "mean_subbed_squared"); // (BATCH_SIZE, n_hidden)

//         ffml_tensor* mean_subbed_squared_mean = ffml_unary_op(FFML_OP_MEAN, mean_subbed_squared, "mean_subbed_squared_mean"); // (BATCH_SIZE, n_hidden)

//         ffml_tensor* mean_subbed_squared_mean_sqrt = ffml_unary_op(FFML_OP_SQRT, mean_subbed_squared_mean, "mean_subbed_squared_mean_sqrt"); // (BATCH_SIZE, n_hidden)

//         ffml_tensor* mean_subbed_squared_mean_sqrt_reciprocal = ffml_unary_op(FFML_OP_RECIPROCAL, mean_subbed_squared_mean_sqrt, "mean_subbed_squared_mean_sqrt_rec

ffml_tensor* tanh_activation(ffml_tensor* input) {
    return input;
    return ffml_unary_op(FFML_OP_TANH, input);
    // return ffml_unary_op(FFML_OP_LEAKY_RELU, input);
}

// class AttentionLayer: public Module {
// public:
//     ffml_tensor* wq;
//     ffml_tensor* wk;
//     ffml_tensor* wv;
//     ffml_tensor* wo;

//     int n_embd;
//     int head_size;
//     int n_heads;

//     AttentionLayer(const int n_embd, const int head_size, const int n_heads): Module() {
//         this->n_embd = n_embd;
//         this->head_size = head_size;
//         this->n_heads = n_heads;

//         this->wq = ffml_tensor_create(2, {n_embd, head_size, 0, 0}, "wq"); // (n_embd, n_hidden)
//         this->wq->op = FFML_OP_INIT_RND_NORMAL;
//         this->wq->op_metadata->insert({"init_rnd_normal.stddev", 0.02f});

//         this->wk = ffml_tensor_create(2, {n_embd, head_size, 0, 0}, "wk"); // (n_embd, n_hidden)
//         this->wk->op = FFML_OP_INIT_RND_NORMAL;
//         this->wk->op_metadata->insert({"init_rnd_normal.stddev", 0.02f});

//         this->wv = ffml_tensor_create(2, {n_embd, head_size, 0, 0}, "wv"); // (n_embd, n_hidden)
//         this->wv->op = FFML_OP_INIT_RND_NORMAL;
//         this->wv->op_metadata->insert({"init_rnd_normal.stddev", 0.02f});

//         this->wo = ffml_tensor_create(2, {n_hidden, n_embd, 0, 0}, "wo"); // (n_hidden, n_embd)
//         this->wo->op = FFML_OP_INIT_RND_NORMAL;
//         this->wo->op_metadata->insert({"init_rnd_normal.stddev", 0.02f});

//         this->bq = ffml_tensor_create(1, {n_hidden, 0, 0, 0}, "bq"); // (n_hidden)
//         this->bq->op =
//     }
// }

class TransformerModel: public Module {
public:
    ffml_tensor* embedding_table;
    ffml_tensor* position_embedding_table;

    int vocab_size;
    int n_embd;
    int batch_size;
    int context_len;
    int n_hidden;
    int head_size;
    int n_layers;
    int n_heads;

    ffml_tensor* w;
    ffml_tensor* b;

    // std::vector<AttentionLayer*> attention_layers;

    LinearLayer* ff_layer;
    LinearLayer* classifier_layer;

    TransformerModel(const int batch_size, const int context_len, const int vocab_size, const int n_embd, const int n_hidden, const int head_size, const int n_layers, const int n_heads): Module() {
        this->vocab_size = vocab_size;
        this->n_embd = n_embd;
        this->batch_size = batch_size;
        this->context_len = context_len;
        this->n_hidden = n_hidden;
        this->head_size = head_size;
        this->n_layers = n_layers;
        this->n_heads = n_heads;

        this->embedding_table = ffml_tensor_create(2, {vocab_size, n_embd, 0, 0}, "embd"); // (vocab_size, n_embd)
        this->embedding_table->op = FFML_OP_INIT_RND_UNIFORM;
        this->parameters.push_back(this->embedding_table);

        // this->position_embedding_table = ffml_tensor_create(1, {context_len, 0, 0, 0}, "position_embd"); // (context_len, n_embd)
        // this->position_embedding_table->op = FFML_OP_INIT_ARANGE;
        // this->position_embedding_table->op_metadata->insert({"init_arange.start", 0.0f});
        // this->position_embedding_table->op_metadata->insert({"init_arange.end", 0.1f});

        // for(int i = 0; i < n_layers; i++) {
        //     this->attention_layers.push_back(new AttentionLayer(n_embd, n_hidden, head_size, n_heads));
        //     this->addParametersFromModule(this->attention_layers[i]);
        // }

        this->ff_layer = new LinearLayer(n_embd, n_hidden, 0.1f, 0.01f);
        this->addParametersFromModule(this->ff_layer);

        this->classifier_layer = new LinearLayer(this->context_len * this->n_hidden, this->vocab_size, 0.01f, 0.01f);
        this->addParametersFromModule(this->classifier_layer);
    }

    ffml_tensor* call(ffml_tensor* Xb) {
        printf("Don't call TransformerModel here.\n");
        exit(1);
    }

    std::pair<ffml_tensor*, ffml_tensor*> call(ffml_tensor* Xb, ffml_tensor* Yb_onehot) {
        ffml_tensor* embedded = ffml_op(FFML_OP_SELECT, Xb, this->embedding_table, "embedded"); // (BATCH_SIZE, CONTEXT_LEN, N_EMBD)

        // reshape for matmuls
        ffml_tensor* embedded_flattened = ffml_reshape(embedded, 2, {this->batch_size * this->context_len, this->n_embd, 0, 0}); // (BATCH_SIZE * CONTEXT_LEN, N_EMBD)

        // ffml_tensor* roped = ffml_op(FFML_OP_ADD, embedded_flattened, this->position_embedding_table, "roped"); // (BATCH_SIZE * CONTEXT_LEN, N_EMBD)

        ffml_tensor* ff_layer_out = this->ff_layer->call(embedded_flattened); // (BATCH_SIZE * CONTEXT_LEN, N_EMBD) @ (N_EMBD, N_HIDDEN) = (BATCH_SIZE * CONTEXT_LEN, N_HIDDEN)

        // ffml_tensor* ff_layer_norm = layernorm(ff_layer_out);

        ffml_tensor* ff_layer_out_tanh = tanh_activation(ff_layer_out); // (BATCH_SIZE * CONTEXT_LEN, N_HIDDEN)
        ffml_set_name(ff_layer_out_tanh, "ff_layer_out_tanh");

        ffml_tensor* ff_layer_out_reshaped = ffml_reshape(ff_layer_out_tanh, 2, {this->batch_size, this->context_len * this->n_hidden, 0, 0}); // (BATCH_SIZE, CONTEXT_LEN * N_HIDDEN)

        auto classifier_tensor = this->classifier_layer->call(ff_layer_out_reshaped); // (BATCH_SIZE, VOCAB_SIZE)

        // auto logits = tanh_activation(classifier_tensor); // (BATCH_SIZE, VOCAB_SIZE)
        auto logits = classifier_tensor;
        ffml_set_name(logits, "logits");

        auto loss = ffml_op(FFML_OP_SOFTMAX_CROSS_ENTROPY, logits, Yb_onehot, "loss");

        auto final_loss = ffml_unary_op(FFML_OP_MEAN, loss, "final_loss");
        
        return {logits, final_loss};
    }
};

void karpathygpt() {
    // ----------- load the data -----------------
    printf("Loading TinyShakespeare data...\n");
    std::string text = file_get_contents("../datasets/tinyshakespeare.txt");

    // print the first 100 characters just to see what it looks like
    printf("First 100 characters of the dataset:\n");
    for(int i = 0; i < 100; i++) {
        printf("%c", text[i]);
    }
    printf("\n-----\n");

    auto tokenizer = CharacterTokenizer(text, 1); // offset == 1, for padding character
    uint64_t VOCAB_SIZE = (uint64_t)tokenizer.vocab_size;

    // ---------- hyperparameters ---------------
    const uint64_t GRAB_BATCH_SIZE = 4 * 2;
    const uint64_t CONTEXT_LEN = 8;
    const uint64_t BATCH_SIZE = GRAB_BATCH_SIZE * CONTEXT_LEN;
    const uint64_t N_EMBD = VOCAB_SIZE;
    const uint64_t N_HIDDEN = N_EMBD;
    const uint64_t MAX_STEPS = 2000000;
    const uint64_t EPOCHS = 200;
    const uint64_t HEAD_SIZE = 16;
    const uint64_t N_HEADS = 2;
    const uint64_t N_LAYERS = 2;

    uint64_t MAX_IDX = 100000;
    MAX_IDX = std::min((int)text.size() - CONTEXT_LEN - 1, MAX_IDX);

    // ----------- build the dataset -------------
    printf("Building the dataset...\n");
    auto xs = std::vector<std::vector<int>>();
    auto ys = std::vector<int>();
    for (int i = 0; i < MAX_IDX; i++) {
        std::string block = text.substr(i, CONTEXT_LEN + 1); // +1 because we want to grab the next character too

        for (int pad = 0; pad < CONTEXT_LEN; pad++) { // if CONTEXT_LEN is 8, pad goes from 0 to 7
            std::vector<int> x;
            for (int j = 0; j < pad; j++) {
                x.push_back(0);
            }
            for (int j = 0; j < CONTEXT_LEN - pad; j++) {
                x.push_back(tokenizer.stoi[block[j]]);
            }
            xs.push_back(x);
            char y = block[CONTEXT_LEN - pad];
            ys.push_back(tokenizer.stoi[y]);
        }
    }

    auto transformX = std::bind(float_direct_pass, std::placeholders::_1);
    auto transformY = std::bind(one_hot, VOCAB_SIZE, std::placeholders::_1);
    auto dataSource = new DataSource<std::vector<int>, int, std::vector<float>, std::vector<float>>(&xs, &ys, transformX, transformY);
    auto dataLoader = new DataLoader<std::vector<int>, int, std::vector<float>, std::vector<float>>(dataSource, BATCH_SIZE, DataSplitConfig({0.8, 0.1}));

    printf("Loaded.\n");

    // print the first 20 examples just to see what it looks like
    printf("First 20 examples of the dataset:\n");
    for(int i = 0; i < 20; i++) {
        printf("x: ");
        for(int j = 0; j < xs[i].size(); j++) {
            printf("%d ", xs[i][j]);
        }
        printf("y: %d\n", ys[i]);
    }

    // ---------------------- configure cgraph ------------------------

    // create a memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(1 * GB);

    // configure a model
    auto model = new TransformerModel(BATCH_SIZE, CONTEXT_LEN, VOCAB_SIZE, N_EMBD, N_HIDDEN, HEAD_SIZE, N_LAYERS, N_HEADS);

    auto Xb = ffml_tensor_create(2, {BATCH_SIZE, CONTEXT_LEN, 0, 0}, "Xb");
    auto Yb_onehot = ffml_tensor_create(2, {BATCH_SIZE, VOCAB_SIZE, 0, 0}, "Yb_onehot");

    auto model_out = model->call(Xb, Yb_onehot);
    auto logits = model_out.first;
    auto final_loss = model_out.second;

    // print total parameters
    int total_params = 0;
    for(auto param: model->parameters) {
        total_params += param->nelem;
    }
    printf("total_params: %d\n", total_params);

    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(final_loss);
    ffml_cgraph_alloc(cgraph, pool, true);

    // ------------------------ ui ----------------------------
    // app(cgraph);

    server_start(cgraph);

    // ---------------------- train ------------------------

    KeyboardListenerGrad * keyboardListenerGrad = new KeyboardListenerGrad(cgraph);

    // keyboardListenerGrad->attach('h', [&]() {
    //     print_histogram_of_tensor(h, 30);
    //     print_histogram_of_tensor(hpreact, 30);
    //     keyboardListenerGrad->pause();
    // });

    // keyboardListenerGrad->attach('t', [&]() {
    //     print_threshold_map(h, 0.99f, 10000);
    //     keyboardListenerGrad->pause();
    // });

    // keyboardListenerGrad->attach('y', [&]() {
    //     print_two_threshold_map(h, -0.99f, 0.99f, 70000);
    //     keyboardListenerGrad->pause();
    // });

    std::vector<float> lossi;

    auto optim = new AdamW();
    optim->addParameters(model->parameters);

    auto lrs = new LearningRateScheduler();
    for (int i=0; i<MAX_STEPS; i++) {
        bool print_on_this_step = (i % 10 == 0);

        // minibatch construct
        dataLoader->loadBatch(SplitType::TRAIN, i, Xb, Yb_onehot);

        // forward pass
        ffml_cgraph_forward(cgraph);

        // backward pass
        ffml_zerograd(cgraph);
        ffml_cgraph_backward(cgraph);

        // if (test_for_dead_neurons({h})) {
        //     keyboardListenerGrad->pause();
        // }

        // ffml_debug_print_cgraph_data(cgraph);
        // exit(1);

        lossi.push_back(ffml_get_data(final_loss, {0,0,0,0}));
        if (print_on_this_step) {
            float final_loss_number = ffml_get_data(final_loss, {0,0,0,0});
            server_push_event("training_step", {
                {"step", i},
                {"loss", final_loss_number}
            });

            printf("step: %d loss: %f", i, final_loss_number);
        }

        // update params
        // lrs->setSameBasedOnBatchSize((i < 500) ? 0.001f : 0.001f / 10.0f, batch_size);
        // LEARNING RATE LRATE
        const float start_lr = 0.03f;
        lrs->setSame((i < 500) ? start_lr : start_lr/10.0f);
        const float LR = lrs->getLR();
        if (print_on_this_step) printf(" lr: %f", LR);
        optim->step(LR);

        // more info
        if (print_on_this_step) printf(" batch_size: %lu", BATCH_SIZE);

        if (print_on_this_step) printf("\n");

        // keyboard listener 
        if (print_on_this_step) keyboardListenerGrad->check();

        if (print_on_this_step) server_loop_interrupt();

        // // **************** validation **************
        // if (i % 20 == 0) {
        //     // val loss
        //     dataLoader->loadBatch(SplitType::VAL, i, Xb, Yb_onehot);

        //     // forward pass
        //     ffml_cgraph_forward(cgraph);

        //     // get loss
        //     float val_loss = ffml_get_data(final_loss, {0,0,0,0});

        //     printf("val_loss: %f ", val_loss);
        // }

        // ************** sample inference ************
        if (i % 100 == 0) {
            printf("\n");

            std::vector<std::string> contexts;
            for(int sample=0; sample < BATCH_SIZE; sample++) {
                std::string context = "";
                for(int q=0; q<CONTEXT_LEN; q++) context += (char)0x00;
                contexts.push_back(context);
            }

            // use the model to generate some text
            for(int q=0; q<10; q++) {
                for(uint64_t sample=0; sample < BATCH_SIZE; sample++) {
                    // get the context
                    std::string context = contexts[sample].substr(contexts[sample].size() - CONTEXT_LEN, CONTEXT_LEN);
                    std::vector<int> context_ix;
                    for(int j = 0; j < CONTEXT_LEN; j++) {
                        context_ix.push_back(tokenizer.stoi[context[j]]);
                    }

                    // set the context
                    for(uint64_t j = 0; j < CONTEXT_LEN; j++) {
                        ffml_set_data(Xb, {sample, j, 0, 0}, (float)context_ix[j]);
                    }
                }

                // forward pass
                ffml_cgraph_forward(cgraph);

                for(uint64_t sample=0; sample < BATCH_SIZE; sample++) {
                    // get the logits
                    std::vector<float> inference_logits;
                    for(uint64_t j = 0; j < VOCAB_SIZE; j++) {
                        inference_logits.push_back(ffml_get_data(logits, {sample, j, 0, 0}));
                    }

                    // turn logits into probabilities
                    std::vector<float> probabilities;
                    float sum = 0.0f;
                    for(int j = 0; j < VOCAB_SIZE; j++) {
                        float p = std::exp(inference_logits[j]);
                        probabilities.push_back(p);
                        sum += p;
                    }

                    // sample from the logits - multinomial
                    int ix = sample_multinomial(probabilities);

                    // get the char
                    char ch = tokenizer.itos[ix];

                    // append the char to the context
                    contexts[sample] += ch;
                }
            }

            // print the generated text
            const int max_words = 10;
            for(int sample=0; sample < std::min(BATCH_SIZE, (uint64_t)max_words); sample++) {
                int start = 0;
                while (start < contexts[sample].size() && contexts[sample][start] == 0x00) start++;
                for (int j = start; j < contexts[sample].size(); j++) {
                    char ch = contexts[sample][j];
                    printf("%c", ch);
                    if (ch == 0x00) break;
                }
                printf("\n");
            }
        }
    }

}

#endif // KARPATHYGPT_H