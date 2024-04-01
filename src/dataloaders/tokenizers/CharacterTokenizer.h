#ifndef CHARACTER_TOKENIZER_H
#define CHARACTER_TOKENIZER_H

#include <string>
#include <algorithm>
#include <unordered_map>

class CharacterTokenizer {
public:
    std::string itos = "";
    std::unordered_map<char, int> stoi;
    int vocab_size;

    CharacterTokenizer(std::string &text, int shift = 0) {
        // alphabet
        for (int i = 0; i < text.size(); i++) {
            if (this->itos.find(text[i]) == std::string::npos) {
                this->itos += text[i];
            }
        }

        // sort alphabet
        std::sort(this->itos.begin(), this->itos.end());

        // add extra characters
        for (int i = 0; i < shift; i++) {
            this->itos = (char)i + this->itos;
        }
        this->vocab_size = this->itos.size();

        // print alphabet
        printf("alphabet: %s\n", this->itos.c_str());

        // print alphabet size
        printf("alphabet size: %lu\n", this->itos.size());

        // stoi
        for (int i = 0; i < this->itos.size(); i++) {
            this->stoi[this->itos[i]] = i;
        }
        printf("\n");
    }
};

#endif