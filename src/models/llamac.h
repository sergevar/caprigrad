#ifndef LLAMAC_H
#define LLAMAC_H

#include "../dataloaders/tokenizers/LlamaModel.h"

void llamac() {
    LlamaModel * model = new LlamaModel("/home/user/Applications/experiments/llama.cpp/models/7B/ggml-model-f16.bin");

    std::vector<llama_vocab::id> tokenized;
    std::vector<llama_vocab::id> output;

    std::string prompt = "Why I like apples:";

    // Add a space in front of the first character to match OG llama tokenizer behavior
    prompt.insert(0, 1, ' ');

    model->tokenizer->tokenize(prompt, tokenized);

    // print tokens
    for (int i = 0; i < tokenized.size(); i++) {
        printf("%i ", tokenized[i]);
    }

    model->inference(tokenized, output, 100);
}

#endif