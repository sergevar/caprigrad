#ifndef HELPERS_H
#define HELPERS_H

#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include "./colors.h"

std::random_device rd;
std::mt19937 helpers_gen(rd());

std::string file_get_contents(std::string filename) {
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (in) {
        std::string contents;
        in.seekg(0, std::ios::end);
        contents.resize(in.tellg());
        in.seekg(0, std::ios::beg);
        in.read(&contents[0], contents.size());
        in.close();
        return(contents);
    }
    printf("Error reading file %s\n", filename.c_str());
    throw(errno);
}

std::vector<std::string> explode(char delim, std::string s) {
    std::vector<std::string> result;
    std::istringstream iss(s);

    for (std::string token; std::getline(iss, token, delim); ) {
        result.push_back(std::move(token));
    }

    return result;
}

int sample_multinomial(std::vector<float> probabilities) {
    std::discrete_distribution<> d(probabilities.begin(), probabilities.end());
    return d(helpers_gen);
}

void print_histogram_of_tensor(ffml_tensor * h, int buckets = 30) {
    printf("Histogram of tensor %s\n", h->name);

    // print histogram of 'h'
    std::vector<float> h_data;
    std::vector<int> histo;
    for(uint64_t i = 0; i < h->nelem; i++) {
        FFML_TYPE value = ffml_get_data_flat(h, i);
        h_data.push_back(value);
    }

    std::sort(h_data.begin(), h_data.end());

    float min = h_data[0];
    float max = h_data[h_data.size() - 1];
    float range = max - min;
    float bucket_size = range / buckets;

    for(int i = 0; i < buckets; i++) {
        histo.push_back(0.0f);
    }

    for(uint64_t i = 0; i < h_data.size(); i++) {
        int bucket = (int)((h_data[i] - min) / bucket_size);
        if (bucket >= buckets) bucket = buckets - 1;
        histo[bucket]++;
    }

    for(int i = 0; i < buckets; i++) {
        printf("(%f..%f) %d\n", min + i * bucket_size, min + (i + 1) * bucket_size, histo[i]);
    }

    printf("\n");
}

void print_threshold_map(ffml_tensor * h, float threshold, int max_elems = 0, bool greater_than = true) {
    printf("Threshold map of tensor %s, threshold: value > %f\n", h->name, threshold);

    if (max_elems > 0 && max_elems < h->nelem) {
        printf("Showing only first %d elements\n", max_elems);
    }

    // print shape
    printf("Shape: ");
    for(int i = 0; i < h->n_dims; i++) {
        printf("%lu ", h->ne[i]);
    }
    printf("\n");

    // print visualization of the tensor, with white/black depending of whether the element exceeds the threshold
    uint64_t c = 0;
    for(uint64_t row = 0; row < (h->n_dims > 1) ? h->ne[0] : 1; row++) {
        for(uint64_t col = 0; col < (h->n_dims > 1) ? h->nelem / h->ne[0] : h->nelem; col++) {
            const uint64_t index = (h->n_dims > 1) ? row * (h->nelem / h->ne[0]) + col : col;
            FFML_TYPE value = ffml_get_data_flat(h, index);
            // use pseudographic
            if ((greater_than && value > threshold) || (!greater_than && value < threshold)) {
                printf("\u2588"); // full block
            } else {
                printf("\u2591"); // light shade
            }

            c++;
            if (max_elems > 0 && c >= max_elems) break;
        }
        printf("\n");
        if (max_elems > 0 && c >= max_elems) break;
    }
}

void print_two_threshold_map(ffml_tensor * h, float threshold1, float threshold2, int max_elems = 0) {
    // make sure thresholds are in the right order
    if (threshold1 > threshold2) {
        float tmp = threshold1;
        threshold1 = threshold2;
        threshold2 = tmp;
    }

    printf("Two threshold map of tensor %s, thresholds: value < %f, value >= %f but < %f, value >= %f\n", h->name, threshold1, threshold1, threshold2, threshold2);

    if (max_elems > 0 && max_elems < h->nelem) {
        printf("Showing only first %d elements\n", max_elems);
    }

    // print shape
    printf("Shape: ");
    for(int i = 0; i < h->n_dims; i++) {
        printf("%lu ", h->ne[i]);
    }
    printf("\n");

    // print visualization of the tensor, with white/black depending of where the element is
    uint64_t c = 0;
    for(uint64_t row = 0; row < (h->n_dims > 1) ? h->ne[0] : 1; row++) {
        for(uint64_t col = 0; col < (h->n_dims > 1) ? h->nelem / h->ne[0] : h->nelem; col++) {
            const uint64_t index = (h->n_dims > 1) ? row * (h->nelem / h->ne[0]) + col : col;
            FFML_TYPE value = ffml_get_data_flat(h, index);
            // use pseudographic
            if (value < threshold1) {
                // // full block
                // printf("\u2588");
                printWithBackground(" ", Color::BG_RED);
            } else if (value >= threshold1 && value < threshold2) {
                // // medium shade
                // printf("\u2592");
                printf(" ");
            } else {
                // // completely black
                // printf(" ");
                printWithBackground(" ", Color::BG_BLUE);
            }

            c++;
            if (max_elems > 0 && c >= max_elems) break;
        }

        resetColor();
        printf("\n");
        if (max_elems > 0 && c >= max_elems) break;
    }
}

bool _is_normal_value(FFML_TYPE value, ffml_op_type op) {
    switch(op) {
        case FFML_OP_TANH:
        case FFML_OP_SIGMOID:
            return value > -0.99f && value < 0.99f;
        case FFML_OP_RELU:
            return value > 0.0f;
        default:
            printf("Cannot test for normal values on this tensor op\n");
            return false;
    }    
}

bool test_for_dead_neurons(ffml_tensor * h) {
    bool found = false;

    assert(h->n_dims == 2);

    for(uint64_t i = 0; i < h->ne[1]; i++) {
        int normal_values = 0;

        for(uint64_t j = 0; j < h->ne[0]; j++) {
            const uint64_t index = i * h->ne[0] + j;
            FFML_TYPE value = ffml_get_data_flat(h, index);

            if (_is_normal_value(value, h->op)) {
                normal_values++;
                break;
            }
        }

        if (normal_values == 0) {
            found = true;
            printf("Neuron %lu in %s seems to be dead - none of the %lu examples activated it\n", i, h->name, h->ne[0]);
        }
    }

    if (!found) {
        // printf("No dead neurons found in %s\n", std::string(h->name).c_str());
    }

    return found;
}

bool test_for_dead_neurons(std::vector<ffml_tensor *> hs) {
    bool found = false;

    for(int i = 0; i < hs.size(); i++) {
        ffml_tensor * h = hs[i];
        bool dead = test_for_dead_neurons(h);
        if (dead) found = true;
    }

    return found;
}

#endif // HELPERS_H