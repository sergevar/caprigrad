#ifndef CONV_MLP_H
#define CONV_MLP_H

#include "../../ffml/ffml.h"
#include "ConvLayer.h"
#include "MaxPoolLayer.h"

#include <vector>

class ConvMLP {
public:
    ffml_memory_pool* pool;
    int n_in_channels;
    int n_in_x;
    int n_in_y;
    int kernel_size;
    int n_filters;
    std::vector<ConvLayerConfiguration> layer_configurations;
    std::vector<ConvLayer*> convlayers;
    std::vector<MaxPoolLayer*> maxpoollayers;

    std::vector<ffml_tensor*> parameters;

    ConvMLP(ffml_memory_pool* _pool, int _n_in_channels, int _n_in_x, int _n_in_y, std::vector<ConvLayerConfiguration> _layer_configurations);

    ffml_tensor* call(ffml_tensor* inputs);

    // std::vector<ConvLayer*>::iterator begin();

    // std::vector<ConvLayer*>::iterator end();
};


#endif