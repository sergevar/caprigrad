#ifndef OPTIM_ADAM_H
#define OPTIM_ADAM_H

#include <vector>
#include "Optimizer.h"
#include "../ffml/ffml.h"

class Adam: public Optimizer {
public:
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8;

    std::unordered_map<uint64_t, std::vector<float>> gradStates;
    std::unordered_map<uint64_t, std::vector<float>> optimStates;
    uint64_t t = 0; // timestep

    Adam(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8) {
        this->beta1 = beta1;
        this->beta2 = beta2;
        this->epsilon = epsilon;
    }

    virtual void step(float LR) {
        this->t += 1; // Increase the timestep
        for(auto t_iter = this->parameters.begin(); t_iter != this->parameters.end(); t_iter++) {
            auto t = *t_iter;

            if(this->gradStates.find(t->key) == this->gradStates.end()) {
                    this->gradStates[t->key] = std::vector<float>(t->nelem, 0.0);
                    this->optimStates[t->key] = std::vector<float>(t->nelem, 0.0);
            }

            for(uint64_t i = 0; i < t->nelem; i++) {
                float current_data = ffml_get_data_flat(t, i);
                float grad = ffml_get_grad_flat(t, i);

                // Update Biased first moment estimate
                this->gradStates[t->key][i] = this->beta1 * this->gradStates[t->key][i] + (1 - this->beta1) * grad;

                // Update Biased second raw moment estimate
                this->optimStates[t->key][i] = this->beta2 * this->optimStates[t->key][i] + (1 - this->beta2) * grad * grad; 

                // Compute bias-corrected first moment estimate
                float grad_corrected = this->gradStates[t->key][i] / (1 - std::pow(this->beta1, this->t)); 

                // Compute bias-corrected second raw moment estimate
                float optim_corrected = this->optimStates[t->key][i] / (1 - std::pow(this->beta2, this->t)); 

                float new_data = current_data - LR * grad_corrected / (sqrt(optim_corrected) + this->epsilon);

                ffml_set_data_flat(t, i, new_data);
            }
        }
    }
};

#endif