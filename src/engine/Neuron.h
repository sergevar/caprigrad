#ifndef NEURON_H
#define NEURON_H

#include "../ffml/ffml.h"

class Neuron {
public:
    ffml_memory_pool* pool;
    ffml_tensor* inputs;
    ffml_tensor* w;
    ffml_tensor* b;

    Neuron(ffml_memory_pool* _pool, ffml_tensor* _inputs);
    ffml_tensor* call();
};

#endif