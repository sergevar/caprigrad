#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include "../ffml/ffml.h"

class Module {
public:
    std::vector<ffml_tensor *> parameters;
    Module() {}
    virtual ffml_tensor* call(ffml_tensor* input) = 0;

    void addParametersFromModule(Module* module) {
        for (auto parameter : module->parameters) {
            this->parameters.push_back(parameter);
        }
    }
};

#endif