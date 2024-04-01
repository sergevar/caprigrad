#ifndef MNIST_LOAD_H
#define MNIST_LOAD_H

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
#include <iostream>

using namespace std;


std::vector<std::string> test_image() {
    vector<string> lines;

    string line;
    ifstream file("../datasets/misc/test_digit.txt");
    if (file.is_open()) {
        while (getline(file, line)) {
            lines.push_back(line + "              ");
        }
        lines.push_back("                            ");
        lines.push_back("                            ");
        lines.push_back("                            ");
        lines.push_back("                            ");
        lines.push_back("                            ");
        lines.push_back("                            ");
        lines.push_back("                            ");
        file.close();
    } else {
        printf("Unable to open file\n");
        exit(1);
    }

    return lines;

    return {
        "0000000000000000000000000000",
        "0000000000000000000000000000",
        "0000000000000000000000000000",
        "0000000000000000000000000000",
        "0000000000000000000000000000",
        "0000000000000000000000000000",
        "0000000000000000000000000000",
        "0000000111111111000000000000",
        "0000000000000011000000000000",
        "0000000000000011000000000000",
        "0000000000000011000000000000",
        "0000000000000011000000000000",
        "0000000000000011000000000000",
        "0000000000111111111100000000",
        "0000000000000011000000000000",
        "0000000000000011000000000000",
        "0000000000000011000000000000",
        "0000000000000011000000000000",
        "0000000000000011000000000000",
        "0000000000000110000000000000",
        "0000000000000110000000000000",
        "0000000000001110000000000000",
        "0000000000000000000000000000",
        "0000000000000000000000000000",
        "0000000000000000000000000000",
        "0000000000000000000000000000",
        "0000000000000000000000000000",
        "0000000000000000000000000000"
    };
}


void mnist_load() {
    ffml_memory_pool* pool = ffml_memory_pool_create(1 * GB);

    // create computation graph/memory for tensors
    std::string model_path = "mnist_model.bin";
    printf("Loading model from %s\n", model_path.c_str());
    ffml_cgraph * cgraph = ffml_load(model_path.c_str(), pool);
    printf("Loaded model from %s\n", model_path.c_str());

    ffml_tensor * inputs = ffml_get_tensor_by_name(cgraph, "inputs");

    // fill from test_image
    printf("Filling inputs\n");
    printf("inputs->ne[0] = %d\n", inputs->ne[0]);
    for(int i = 0; i < inputs->ne[0]; i++) {
        int x = i / 28;
        int y = i % 28;
        char ch = test_image()[x][y];
        int image_bit_01 = (ch != '0' && ch != ' ') ? 1 : 0;
        float image_bit = image_bit_01 ? 1.0f : 0.0f;

        printf("%d ", image_bit_01);
        if (y == 27) {
            printf("\n");
        }

        ffml_set_data_flat(inputs, i, image_bit);
    }

    // forward pass
    ffml_cgraph_forward(cgraph);

    // ffml_debug_print_cgraph_data(cgraph);

    // get output
    ffml_tensor * outputs = ffml_get_tensor_by_name(cgraph, "layer1_act");

    // get ypred
    std::vector<float> logits;
    logits.resize(outputs->ne[0]);
    for(int j = 0; j < outputs->ne[0]; j++) {
        logits[j] = ffml_get_data(outputs, {j,0,0,0});
    }

    // print logits
    printf("logits: ");
    for(int j = 0; j < logits.size(); j++) {
        printf("%f ", logits[j]);
    }

    // softmax manually
    float sum = 0.0f;
    for(int j = 0; j < logits.size(); j++) {
        sum += exp(logits[j]);
    }
    for(int j = 0; j < logits.size(); j++) {
        logits[j] = exp(logits[j]) / sum;
    }

    // get maximum
    float max = -1.0f;
    int max_idx = -1;
    for(int j = 0; j < logits.size(); j++) {
        if (logits[j] > max) {
            max = logits[j];
            max_idx = j;
        }
    }

    // print prediction
    printf("\n");
    printf("prediction: %d\n", max_idx);

    // ffml_debug_print_cgraph_data(cgraph);

    exit(0);

}

#endif // MNIST_H