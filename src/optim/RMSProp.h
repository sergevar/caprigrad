#ifndef OPTIM_RMSPROP_H
#define OPTIM_RMSPROP_H

#include <vector>
#include "Optimizer.h"
#include "../ffml/ffml.h"

class RMSProp: public Optimizer {
public:
    float decayRate = 0.0;
    float epsilon = 0.0;

    std::unordered_map<uint64_t, std::vector<float>> optimStates;

    RMSProp(float decayRate = 0.99f, float epsilon = 1e-8) {
        // validate
        if(decayRate < 0.0 || decayRate > 1.0) {
            printf("decayRate must be between 0.0 and 1.0\n");
            exit(1);
        }
        this->decayRate = decayRate;
        this->epsilon = epsilon;
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

                this->optimStates[t->key][i] = this->decayRate * this->optimStates[t->key][i] + (1.0f - this->decayRate) * grad * grad;

                float new_data = current_data - LR * grad / (sqrt(this->optimStates[t->key][i]) + this->epsilon);
    
                ffml_set_data_flat(t, i, new_data);
            }
        }
    }
};

#endif