#include "ConvLayer.h"

#include <assert.h>

ConvLayer::ConvLayer(ffml_memory_pool* _pool, int _n_in_channels, int _n_in_x, int _n_in_y, int _n_filters, int _kernel_size, std::string _name) {
    this->pool = _pool;
    this->n_in_x = _n_in_x;
    this->n_in_y = _n_in_y;
    this->n_filters = _n_filters;
    this->n_in_channels = _n_in_channels;
    this->kernel_size = _kernel_size;

    this->name = _name;

    this->out_x = this->n_in_x - this->kernel_size + 1;
    this->out_y = this->n_in_y - this->kernel_size + 1;

    this->kernels = ffml_tensor_create(4, { this->n_filters, this->n_in_channels, this->kernel_size, this->kernel_size }, (name + "_filter").c_str());
    this->kernels->op = FFML_OP_INIT_RND_UNIFORM;
    this->parameters.push_back(this->kernels);
}

ffml_tensor* ConvLayer::call(ffml_tensor* inputs) {
    assert(inputs->n_dims == 3);

    ffml_tensor* convolved = ffml_op(FFML_OP_CONV2D, inputs, this->kernels);
    ffml_set_name(convolved, this->name + "_conv");

    return convolved;
}

std::vector<ffml_tensor*>::const_iterator ConvLayer::begin() {
    return this->parameters.begin();
}

std::vector<ffml_tensor*>::const_iterator ConvLayer::end() {
    return this->parameters.end();
}