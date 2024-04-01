#include "MaxPoolLayer.h"

#include <assert.h>
#include <iostream>

MaxPoolLayer::MaxPoolLayer(ffml_memory_pool* _pool, int _n_in_channels, int _n_in_x, int _n_in_y, int _pool_size, std::string _name) {
    this->pool = _pool;
    this->n_in_channels = _n_in_channels;
    this->n_in_x = _n_in_x;
    this->n_in_y = _n_in_y;
    this->pool_size = _pool_size;

    this->name = _name;

    this->out_x = ceil(this->n_in_x*1.0 / 2.0);
    this->out_y = ceil(this->n_in_y*1.0 / 2.0);

    // // print in and out x and y
    // std::cout << "in_x: " << this->n_in_x << std::endl;
    // std::cout << "in_y: " << this->n_in_y << std::endl;
    // std::cout << "out_x: " << this->out_x << std::endl;
    // std::cout << "out_y: " << this->out_y << std::endl;
}

ffml_tensor* MaxPoolLayer::call(ffml_tensor* inputs) {
    assert(inputs->n_dims == 3);

    ffml_tensor* pooled = ffml_unary_op(FFML_OP_MAXPOOL2D, inputs);
    ffml_set_name(pooled, this->name + "_pooled");

    assert(pooled->n_dims == 3);
    assert(pooled->ne[0] == this->n_in_channels);
    assert(pooled->ne[1] == this->out_x);
    assert(pooled->ne[2] == this->out_y);

    return pooled;
}

std::vector<ffml_tensor*>::const_iterator MaxPoolLayer::begin() {
    return this->parameters.begin();
}

std::vector<ffml_tensor*>::const_iterator MaxPoolLayer::end() {
    return this->parameters.end();
}