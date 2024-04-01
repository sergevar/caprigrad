#ifndef BASIC_OPTIMIZER_H
#define BASIC_OPTIMIZER_H

#include <vector>
#include "Optimizer.h"
#include "../ffml/ffml.h"

class BasicOptimizer: public Optimizer {
public:
    BasicOptimizer() {
    }

    virtual void step(float LR) {
        for(auto t_iter = this->parameters.begin(); t_iter != this->parameters.end(); t_iter++) {
            auto t = *t_iter;

            // printf("adjusting t: %s\n", t->name);

            for(uint64_t i = 0; i < t->nelem; i++) {
                float current_data = ffml_get_data_flat(t, i);
                float grad = ffml_get_grad_flat(t, i);
                float update = -LR * grad;
                float new_data = current_data + update;

                ffml_set_data_flat(t, i, new_data);
            }
        }
    }
};

#endif