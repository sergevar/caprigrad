#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include "../ffml/ffml.h"

class Optimizer {
public:
    std::vector<ffml_tensor*> parameters;

    void addParameter(ffml_tensor* parameter) {
        this->parameters.push_back(parameter);
    }

    void addParameters(std::vector<ffml_tensor*> parameters) {
        for (auto parameter : parameters) {
            this->parameters.push_back(parameter);
        }

    }

    virtual void step(float LR) = 0;
};

#endif