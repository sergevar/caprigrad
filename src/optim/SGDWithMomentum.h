#ifndef OPTIM_SGD_WITH_MOMENTUM_H
#define OPTIM_SGD_WITH_MOMENTUM_H

#include <vector>
#include "Optimizer.h"
#include "../ffml/ffml.h"

class SGDWithMomentum: public Optimizer {
public:
    float momentum = 0.0;

    std::unordered_map<uint64_t, std::vector<float>> optimStates;

    SGDWithMomentum(float momentum = 0.9) {
        // validate
        if(momentum < 0.0 || momentum > 1.0) {
            printf("momentum must be between 0.0 and 1.0\n");
            exit(1);
        }
        this->momentum = momentum;
    }

    virtual void step(float LR) {
        for(auto t_iter = this->parameters.begin(); t_iter != this->parameters.end(); t_iter++) {
            auto t = *t_iter;

            // printf("adjusting t: %s\n", t->name);

            if(this->optimStates.find(t->key) == this->optimStates.end()) {
                this->optimStates[t->key] = std::vector<float>(t->nelem, 0.0);
            }

            for(uint64_t i = 0; i < t->nelem; i++) {
                float current_data = ffml_get_data_flat(t, i);
                float grad = ffml_get_grad_flat(t, i);

                this->optimStates[t->key][i] = this->momentum * this->optimStates[t->key][i] - LR * grad;

                // float update = -LR * grad;
                float new_data = current_data + this->optimStates[t->key][i];

                ffml_set_data_flat(t, i, new_data);
            }
        }
    }
};

#endif