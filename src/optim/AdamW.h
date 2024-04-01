#ifndef OPTIM_ADAMW_H
#define OPTIM_ADAMW_H

#include <vector>
#include "Optimizer.h"
#include "../ffml/ffml.h"

class AdamW: public Optimizer {
public:
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8;
    float weight_decay = 0.01f;

    std::unordered_map<uint64_t, std::vector<float>> m;
    std::unordered_map<uint64_t, std::vector<float>> v;
    uint64_t t = 0;

    AdamW(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8, float weight_decay = 0.01f) {
        this->beta1 = beta1;
        this->beta2 = beta2;
        this->epsilon = epsilon;
        this->weight_decay = weight_decay;
    }

    virtual void step(float LR) {
        this->t += 1; 
        for(auto t_iter = this->parameters.begin(); t_iter != this->parameters.end(); t_iter++) {
            auto t = *t_iter;

            // Initialize m and v for each parameter
            if(this->m.find(t->key) == this->m.end()) {
                this->m[t->key] = std::vector<float>(t->nelem, 0.0);
                this->v[t->key] = std::vector<float>(t->nelem, 0.0);
            }

            for(uint64_t i = 0; i < t->nelem; i++) {
                float current_data = ffml_get_data_flat(t, i);
                float grad = ffml_get_grad_flat(t, i);

                // Update biased first moment estimate
                this->m[t->key][i] = this->beta1 * this->m[t->key][i] + (1 - this->beta1) * grad;
                
                // Update biased second moment estimate
                this->v[t->key][i] = this->beta2 * this->v[t->key][i] + (1 - this->beta2) * grad * grad;

                // Compute bias-corrected first and second moment estimates
                float m_hat = this->m[t->key][i] / (1 - std::pow(this->beta1, this->t)); 
                float v_hat = this->v[t->key][i] / (1 - std::pow(this->beta2, this->t));
                
                // Add weight decay to the gradients
                float new_data = current_data - LR * (m_hat / (sqrt(v_hat) + this->epsilon) + this->weight_decay * current_data);
                
                ffml_set_data_flat(t, i, new_data);
            }
        }
    }
};

#endif