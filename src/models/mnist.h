#ifndef MNIST_H
#define MNIST_H

#include "../ffml/ffml.h"
#include "../engine/Layer.h"
#include "../engine/MLP.h"
#include "../engine/conv/ConvMLP.h"
#include "../optim/BasicOptimizer.h"

#include <vector>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <map>
#include <iostream>




#include <thread>
#include <atomic>
#include <iostream>
#include <termios.h>
#include <unistd.h>

struct KeyboardEvent {
    char key;
    bool pressed;
    int id;
};

std::atomic<KeyboardEvent> kb_event({'\0', false, 0});    // flag to control when to exit the application
const float LR_INITIAL = 0.001f;
std::atomic<float> LR_atomic(LR_INITIAL);

void changemode(int);
int  kbhit(void);

// run your keyboard listening in this thread
void keyboard_thread() 
{
    char c;
    while(true)
    {
        if(kbhit()) 
        {
            c = getchar();

            int random_id = rand() % 1000000;

            kb_event.store({
                c, true, random_id
            });

            // printf("key: %c\n", c);

            switch(c) {
                case ']':
                    LR_atomic.store(LR_atomic.load() + LR_INITIAL);
                    printf("LR: %f\n", LR_atomic.load());
                    break;
                case '[':
                    LR_atomic.store(LR_atomic.load() - LR_INITIAL);
                    printf("LR: %f\n", LR_atomic.load());
                    break;
                default:
                    break;
            }
        }
    }
}

void changemode(int dir)
{
    static struct termios oldt, newt;
                    
    if ( dir == 1 )
    {
        tcgetattr( STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~( ICANON | ECHO );
        tcsetattr( STDIN_FILENO, TCSANOW, &newt);
    }
    else
        tcsetattr( STDIN_FILENO, TCSANOW, &oldt);
}

int kbhit (void)
{
    struct timeval tv;
    fd_set rdfs;

    tv.tv_sec = 0;
    tv.tv_usec = 0;

    FD_ZERO(&rdfs);
    FD_SET (STDIN_FILENO, &rdfs);

    select(STDIN_FILENO+1, &rdfs, NULL, NULL, &tv);
    return FD_ISSET(STDIN_FILENO, &rdfs);
}



std::string mnist_path() {
    // check either 'datasets/' or '../datasets/'
    std::ifstream f("datasets/mnist/train.csv");
    if(f.good()) {
        return "datasets/mnist/train.csv";
    } else {
        std::ifstream f("../datasets/mnist/train.csv");
        if(f.good()) {
            return "../datasets/mnist/train.csv";
        } else {
            printf("ERROR: could not find datasets/mnist/train.csv\n");
            exit(1);
        }
    }
}


void mnist() {
    changemode(1);
    
    // create the thread for keyboard listening
    std::thread t2(keyboard_thread);



    const int EMB_WIDTH = 4;
    
    const int N_EPOCHS = 100;

    printf("Loading MNIST dataset...\n");

    std::vector<std::vector<float>> xs;
    std::vector<int> ys;

    // load dataset
    int row = -1;
  	std::fstream file (mnist_path(), std::ios::in);
    std::string line, word;
	if(file.is_open())
	{
        // skip first line
        std::getline(file, line);

		while(std::getline(file, line))
		{
            row++;

			std::stringstream str(line);
 
            std::getline(str, word, ',');
            // std::cout << word << "\t";
            ys.push_back(std::stoi(word));

            std::vector<float> x;

			while(std::getline(str, word, ','))
            {
                x.push_back(std::stof(word));
            }

            xs.push_back(x);

            // std::cout << std::endl;
		}
	}
	else {
		std::cout<<"Could not open the file\n";
        exit(1);
    }

    printf("Done loading MNIST dataset.\n");

    // print last mnist image to make sure we loaded it correctly
    const int _LOAD_IDX = xs.size() - 3;
    printf("ys[0]: %d\n", ys[_LOAD_IDX]);
    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            // printf("%f ", xs[0][i * 28 + j]);
            if (xs[_LOAD_IDX][i * 28 + j] > 0.5f) {
                // unicode
                // printf("\u2588");
                printf("*");
            } else {
                printf(" ");
            }
        }
        printf("\n");
    }
    printf("-----");

    // // iterate over all im  
    // for(int i = 0; i < xs.size(); i++) {
    //     for(int j = 0; j < 784; j++) {
    //         float x = xs[i][j];
    //         if (x > 0.5f) {
    //             xs[i][j] = 255.0f;
    //         } else {
    //             xs[i][j] = 0.0f;
    //         }
    //     }
    // }

//     // print first 100 chars
//     printf("text: %s\n", text.substr(0, 100).c_str());

//     // create a set of unique characters
//     std::set<char> chars;
//     for(int i = 0; i < text.size(); i++) {
//         chars.insert(text[i]);
//     }

//     // create a map from char to index
//     std::map<char, int> char_to_index;
//     int i = 0;
//     for(auto it = chars.begin(); it != chars.end(); it++) {
//         char_to_index[*it] = i;
//         i++;
//     }

//     // print char_to_index
//     printf("char_to_index: ");
//     for(auto it = chars.begin(); it != chars.end(); it++) {
//         // output character and its index
//         printf("%c:%d ", *it, char_to_index[*it]);
//     }

//     // create a map from index to char
//     std::map<int, char> index_to_char;
//     i = 0;
//     for(auto it = chars.begin(); it != chars.end(); it++) {
//         index_to_char[i] = *it;
//         i++;
//     }

//     // output alphabet
//     // printf("alphabet: ");
//     // for(auto it = chars.begin(); it != chars.end(); it++) {
//     //     // output character and its index
//     //     printf("%c:%d ", *it, char_to_index[*it]);
//     // }
//     // printf("\n");

//     const int PREV_WINDOW = 5;

//     const int TEXT_SIZE = text.size();

//     const int ALPHABET_SIZE = chars.size();

    float L1_lambda = 0.000f;
    float L2_lambda = 0.001f;


    const int MINI_BATCH_SIZE = 320;
    // const int MINI_BATCH_SIZE = 1;

//     const int MINI_BATCHES_PER_BATCH = 40;

//     // create memory pool/context
    ffml_memory_pool * pool = ffml_memory_pool_create(1 * GB);

//     // configure a model

    // auto mlp = new MLP(pool, 784, {128, 64, 10});
    auto convmlp = new ConvMLP(pool, 1, 28, 28, {
        {15, 3, 3}, // # of filters, kernel side, pool side
        {9, 3, 3}, // # of filters, kernel side, pool side
    });

    auto mlp = new MLP(pool, 324, {20, 10});

    std::vector<ffml_tensor*> input_tensors;
    std::vector<ffml_tensor*> y_tensors;
    std::vector<ffml_tensor*> logits_tensors;

    std::vector<ffml_tensor*> losses_tensors;



//     ffml_tensor * C = ffml_tensor_create(2, {ALPHABET_SIZE,EMB_WIDTH,0,0}, "C");
//     C->op = FFML_OP_INIT_RND_UNIFORM;

    for(int i = 0; i < MINI_BATCH_SIZE; i++) {
//         // std::string minibatch_entry_text = text.substr(minibatch_start_charidx + i, PREV_WINDOW);

//         //

        ffml_tensor * inputs = ffml_tensor_create(2, {28,28,0,0}, "inputs");
        input_tensors.push_back(inputs);

//         ffml_tensor * inputs_transposed = ffml_unary_op(FFML_OP_TRANSPOSE, inputs);

//         // ffml_debug_print_tensor_metadata(inputs_transposed);

//         ffml_tensor * emb = ffml_op(FFML_OP_MATMUL, inputs_transposed, C);

//         // ffml_debug_print_tensor_metadata(emb);

//         ffml_tensor * emb_reshaped = ffml_reshape(emb, 1, {PREV_WINDOW * EMB_WIDTH, 1, 0, 0});

//         ffml_tensor * emb_reshaped_tanh = ffml_unary_op(FFML_OP_TANH, emb_reshaped);

        // auto inputs_flattened = ffml_reshape(inputs, 1, {784, 0, 0, 0});

        auto inputs_unsqueezed = ffml_unary_op(FFML_OP_UNSQUEEZE, inputs);
        ffml_set_name(inputs_unsqueezed, "inputs_unsqueezed");

        auto conv_logits = convmlp->call(inputs_unsqueezed);

        // ffml_debug_print_tensor_metadata(conv_logits);

        auto conv_logits_flattened = ffml_reshape(conv_logits, 1, {conv_logits->nelem, 0, 0, 0});

        auto logits = mlp->call(conv_logits_flattened);

        // ffml_debug_print_tensor_metadata(logits);

        logits_tensors.push_back(logits);


        ffml_tensor * y = ffml_tensor_create(1, {10,0,0,0}, "y");
//         ffml_tensor * y = ffml_tensor_create(1, {ALPHABET_SIZE,0,0,0}, "y");
//         // ffml_debug_print_tensor_metadata(y);
        y_tensors.push_back(y);

//         // auto ypred = ffml_op(FFML_OP_SOFTMAX, logits);

//         // auto y_diff = ffml_op(FFML_OP_SUB, y, ypred);
//         // ffml_set_name(y_diff, "y_diff");

        auto loss = ffml_op(FFML_OP_SOFTMAX_CROSS_ENTROPY, logits, y);
        ffml_set_name(loss, "loss");

        losses_tensors.push_back(loss);
    }

    auto optim = new BasicOptimizer();
    optim->addParameters(convmlp->parameters);
    optim->addParameters(mlp->parameters);

    // sum losses
    ffml_tensor* total_loss = nullptr;
    for (int i = 0; i < losses_tensors.size(); i++) {
        if (total_loss == nullptr) {
            total_loss = losses_tensors[i];
        } else {
            total_loss = ffml_op(FFML_OP_ADD, total_loss, losses_tensors[i]);
        }
    }

    ffml_tensor* zero = ffml_tensor_create(1, {1,0,0,0}, "zero");

    ffml_tensor* accum_weights_summed = zero;
    ffml_tensor* accum_weights_squared_summed = zero;

    for(ffml_tensor* param: optim->parameters) {
        if (L1_lambda != 0.0f) {
            accum_weights_summed = ffml_op(FFML_OP_ADD, accum_weights_summed, ffml_unary_op(FFML_OP_SUM, ffml_unary_op(FFML_OP_ABS, ffml_flatten(param))));
        }
        if (L2_lambda != 0.0f) {
            accum_weights_squared_summed = ffml_op(FFML_OP_ADD, accum_weights_summed, ffml_unary_op(FFML_OP_SUM, ffml_unary_op(FFML_OP_SQUARE, ffml_flatten(param))));
        }
    }

    ffml_tensor* mini_batch_size_tensor = ffml_tensor_create(1, {1,0,0,0}, "mini_batch_size");

    ffml_tensor* lambda_l1 = ffml_tensor_create(1, {1,0,0,0}, "lambda_l1");
    ffml_tensor* lambda_l2 = ffml_tensor_create(1, {1,0,0,0}, "lambda_l2");

    ffml_tensor* l1_loss = ffml_op(FFML_OP_MUL, lambda_l1, accum_weights_summed);
    ffml_tensor* l2_loss = ffml_op(FFML_OP_MUL, lambda_l2, accum_weights_squared_summed);

    total_loss = ffml_op(FFML_OP_DIV, total_loss, mini_batch_size_tensor);

    total_loss = ffml_op(FFML_OP_ADD, total_loss, l1_loss);
    total_loss = ffml_op(FFML_OP_ADD, total_loss, l2_loss);

    ffml_set_name(total_loss, "crossentropy_loss");

//     // todo: do the average!
// //     // sum losses
// // ffml_tensor* total_loss = nullptr;
// // int total_samples = 0;

// // for (int i = 0; i < losses.size(); i++) {
// //     if (total_loss == nullptr) {
// //         total_loss = losses[i];
// //     } else {
// //         total_loss = ffml_op(FFML_OP_ADD, total_loss, losses[i]);
// //     }
// //     total_samples += MINI_BATCH_SIZE;
// // }

// // // Compute the scaling factor for averaging the loss
// // float scale_factor = 1.0f / total_samples;

// // // Scale the total_loss by the scaling factor to get the average loss
// // total_loss = ffml_op(FFML_OP_MUL, total_loss, ffml_tensor_create_scalar(scale_factor));

// // // Store the total_loss tensor in the computation graph
// // ffml_cgraph_set_output(cgraph, total_loss);


    // create computation graph/memory for tensors
    ffml_cgraph * cgraph = ffml_cgraph_create(total_loss);

    ffml_cgraph_alloc(cgraph, pool);

    // ffml_debug_print_cgraph_shapes(cgraph);

    ffml_set_data(mini_batch_size_tensor, {0,0,0,0}, MINI_BATCH_SIZE);

    printf("Starting training loop...\n");

    // ffml_debug_print_cgraph_data(cgraph);

    uint64_t counter = -1;

    const float train_pct = 0.8f;

    const uint64_t TRAIN_MAX_IDX = xs.size() * train_pct;

    int lastKbEventId = 0;

    // float LR = 0.001f;

    for(int epoch = 0; epoch < N_EPOCHS; epoch++) {
        for (int batch_start = 0; batch_start < TRAIN_MAX_IDX - MINI_BATCH_SIZE; batch_start += MINI_BATCH_SIZE) {
//     for(int minibatch_start_charidx = 0; minibatch_start_charidx < TEXT_SIZE; minibatch_start_charidx+=MINI_BATCH_SIZE) {

        counter++;

        if (epoch == 8) {
            ffml_save(cgraph, "mnist_model.bin");
            exit(0);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        auto ev = kb_event.load();
        if (ev.id != lastKbEventId) {
            lastKbEventId = ev.id;

            if (ev.pressed) {
                switch(ev.key) {
                    case 'q':
                        printf("Quitting...\n");
                        exit(0);
                        break;
                    case 'd':
                        printf("Dumping the graph...");
                        ffml_debug_print_cgraph_shapes(cgraph);
                        ffml_debug_print_cgraph_data(cgraph);
                        exit(0);
                        break;
                    default:
                        break;
                }
            }
        }


        // if (counter > 1) {
        //     exit(1);
        // }

        // load data
//         // for(uint64_t i = 0; i < xs.size(); i++) {
//         //     //     load data
//         //     for(uint64_t j = 0; j < xs[i].size(); j++) {
//         //         ffml_set_data(input_tensors[i], {j, 0, 0, 0}, xs[i][j]);
//         //     }
//         //     ffml_set_data(y_tensors[i], {0, 0, 0, 0}, ys[i]);
//         // }
        for(int i = 0; i < MINI_BATCH_SIZE; i++) {
            int idx = batch_start + i;

            // No! Let's load a random index between 0 and TRAIN_MAX_IDX
            idx = rand() % TRAIN_MAX_IDX;

            for(int j = 0; j < 784; j++) {
                int _x = j / 28;
                int _y = j % 28;
                // ffml_set_data(input_tensors[i], {j, 0, 0, 0}, xs[idx][j] / 255.0f);
                ffml_set_data(input_tensors[i], {_x, _y, 0, 0}, xs[idx][j] / 256.0f);
            }

            // first clear y
            for(uint64_t j = 0; j < 10; j++) {
                ffml_set_data(y_tensors[i], {j, 0, 0, 0}, 0.0f);
            }
            ffml_set_data(y_tensors[i], {ys[idx], 0, 0, 0}, 1.0f);
//             if(minibatch_start_charidx + i >= TEXT_SIZE) {
//                 break;
//             }

//             std::string minibatch_entry_text = text.substr(minibatch_start_charidx + i, PREV_WINDOW);

//             // printf("minibatch_start_charidx: %d\n", minibatch_start_charidx);
//             // printf("i: %d\n", i);
//             // printf("minibatch_entry_text: %s\n", minibatch_entry_text.c_str());

//             // first clear
//             for(uint64_t j = 0; j < PREV_WINDOW; j++) {
//                 for(uint64_t k = 0; k < ALPHABET_SIZE; k++) {
//                     ffml_set_data(input_tensors[i], {k, j, 0, 0}, 0.0f);
//                 }
//             }
//             for(int j = 0; j < PREV_WINDOW; j++) {
//                 char c = minibatch_entry_text[j];
//                 int charidx = char_to_index[c];
//                 ffml_set_data(input_tensors[i], {charidx, j, 0, 0}, 1.0f);
//             }

//             char c = text[minibatch_start_charidx + i + PREV_WINDOW];
//             int charidx = char_to_index[c];
//             // clear y
//             for(uint64_t j = 0; j < ALPHABET_SIZE; j++) {
//                 ffml_set_data(y_tensors[i], {j, 0, 0, 0}, 0.0f);
//             }
//             ffml_set_data(y_tensors[i], {charidx, 0, 0, 0}, 1.0f);
//             // printf("y: %c\n", c);
        }

        // TRAINING LOOP

        //     forward pass
        ffml_cgraph_forward(cgraph);

        ffml_zerograd(cgraph);

        //     backward pass
        ffml_cgraph_backward(cgraph);

        // ffml_debug_print_cgraph_data(cgraph);
        // ffml_debug_print_cgraph_data(cgraph);
        // ffml_debug_print_cgraph_data(cgraph);

        printf(" epoch: %d ", epoch);
        printf(" counter: %lu ", counter);
        printf(" records processed: %d / %lu ", batch_start, TRAIN_MAX_IDX);
        printf(" total loss: %f", ffml_get_data(total_loss, {0,0,0,0}));
        printf("   ||   ");

        //     update weights (optimizer)

        float LR = LR_atomic.load();

        optim->step(LR);


        // run inference to see manually how many are predicted
        if (counter % 1 == 0) {

            for(int record = 0; record < MINI_BATCH_SIZE; record++) {

                // test on last record we processed
                int _idx = TRAIN_MAX_IDX + record;
                // int _idx = xs.size() - 1;

                for(int j = 0; j < 784; j++) {
                    int _x = j / 28;
                    int _y = j % 28;
                    ffml_set_data(input_tensors[record], {_x, _y, 0, 0}, xs[_idx][j] / 256.0f);
                }
                for(uint64_t j = 0; j < 10; j++) {
                    ffml_set_data(y_tensors[record], {j, 0, 0, 0}, 0.0f);
                }
                ffml_set_data(y_tensors[record], {ys[_idx], 0, 0, 0}, 1.0f);
            }

            ffml_cgraph_forward(cgraph);

            int correct = 0;

            for(int record = 0; record < MINI_BATCH_SIZE; record++) {
                int _idx = TRAIN_MAX_IDX + record;

                // get ypred
                std::vector<float> logits;
                logits.resize(logits_tensors[record]->ne[0]);
                for(int j = 0; j < logits_tensors[record]->ne[0]; j++) {
                    logits[j] = ffml_get_data(logits_tensors[record], {j,0,0,0});
                }

                // softmax manually
                float sum = 0.0f;
                for(int j = 0; j < logits.size(); j++) {
                    sum += exp(logits[j]);
                }
                for(int j = 0; j < logits.size(); j++) {
                    logits[j] = exp(logits[j]) / sum;
                }

                // get y from input array
                float correct_y;
                correct_y = ys[_idx];

                
                // ffml_debug_print_cgraph_data(cgraph);


                // printf("logits: ");
                // for(int j = 0; j < logits.size(); j++) {
                //     printf("%f ", logits[j]);
                // }
                // // printf("\n");

                // printf("correct_y: %f ", correct_y);

                int correct_y_int = (int) correct_y;

                // get maximum
                float max = -1.0f;
                int max_idx = -1;
                for(int j = 0; j < logits.size(); j++) {
                    if (logits[j] > max) {
                        max = logits[j];
                        max_idx = j;
                    }
                }

                if (correct_y_int == max_idx) {
                    correct++;
                }

                // printf("predicted: %d ", max_idx);

                // print loss

                // if (correct_y_int == max_idx) {
                //     printf("CORRECT!!!!!!!");
                // } else {
                //     printf("---");
                // }


                //    print loss

                // print ypred
                // printf("ypred: ");
                // for(uint64_t i = 0; i < ypred_tensors.size(); i++) {
                //     printf("%f ", ffml_get_data(ypred_tensors[i], {0,0,0,0}));
                // }

            }

            printf("correct: %d/%d (%f%%) ", correct, MINI_BATCH_SIZE, (float) correct / (float) MINI_BATCH_SIZE * 100.0f);

            printf("loss: %f ", ffml_get_data(total_loss, {0,0,0,0}));


        }

        // if (counter > 10) {
        //     exit(1);
        //     counter = 0;
        //     break;
        // }

        printf("\n");
    }

    // waiting for the keyboard thread to finish
    t2.join();

    changemode(0);
    exit(0);

    }

//     //     evaluate

//     // save model

    // // print hello world:
    // printf("Hello, World!\n");

    // // free memory pool/context
    // ffml_memory_pool_destroy(pool);
    // // todo: destroy tensors too

}

#endif