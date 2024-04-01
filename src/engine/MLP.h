#ifndef MLP_H
#define MLP_H

#include "../ffml/ffml.h"
#include "Layer.h"

#include <vector>

class MLP {
public:
    ffml_memory_pool* pool;
    int n_in;
    std::vector<int> n_hidden;
    std::vector<Layer*> layers;

    MLP(ffml_memory_pool* _pool, int _n_in, std::vector<int> _n_hidden);

    ffml_tensor* call(ffml_tensor* inputs);

    // std::vector<Layer*>::iterator begin();

    // std::vector<Layer*>::iterator end();

    std::vector<ffml_tensor*> parameters;
};


#endif