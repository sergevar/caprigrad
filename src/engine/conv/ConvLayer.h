#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "../../ffml/ffml.h"
#include "ConvLayerConfiguration.h"

class ConvLayer {
public:
    ffml_memory_pool* pool;
    ffml_tensor* kernels;
    int n_in_x;
    int n_in_y;
    int n_in_channels;
    int n_filters;
    int kernel_size;
    int out_x;
    int out_y;
    std::string name;

    uint64_t n_neurons;

    ConvLayer(ffml_memory_pool* _pool, int _n_in_channels, int _n_in_x, int _n_in_y, int _n_filters, int _kernel_size, std::string _name);

    ffml_tensor* call(ffml_tensor* inputs);

    std::vector<ffml_tensor*> parameters;

    std::vector<ffml_tensor*>::const_iterator begin();
    std::vector<ffml_tensor*>::const_iterator end();
};

#endif