#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "../../ffml/ffml.h"

class MaxPoolLayer {
public:
    ffml_memory_pool* pool;
    int n_in_channels;
    int n_in_x;
    int n_in_y;
    int pool_size;
    int out_x;
    int out_y;
    std::string name;

    uint64_t n_neurons;

    MaxPoolLayer(ffml_memory_pool* _pool, int _n_in_channels, int _n_in_x, int _n_in_y, int _pool_size, std::string _name);

    ffml_tensor* call(ffml_tensor* inputs);

    std::vector<ffml_tensor*> parameters;

    std::vector<ffml_tensor*>::const_iterator begin();
    std::vector<ffml_tensor*>::const_iterator end();
};

#endif