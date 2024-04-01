#ifndef LAYER_H
#define LAYER_H

#include "../ffml/ffml.h"

class Layer {
public:
    ffml_memory_pool* pool;
    ffml_tensor* w;
    ffml_tensor* b;
    ffml_tensor* wt;
    int n_in;
    int n_out;
    std::string name;

    uint64_t n_neurons;

    Layer(ffml_memory_pool* _pool, int _n_in, int _n_out, std::string _name);
    ffml_tensor* call(ffml_tensor* inputs);

    std::vector<ffml_tensor*> parameters;

    std::vector<ffml_tensor*>::const_iterator begin();
    std::vector<ffml_tensor*>::const_iterator end();
};

#endif