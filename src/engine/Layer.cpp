#include "Layer.h"

#include <assert.h>

Layer::Layer(ffml_memory_pool* _pool, int _n_in, int _n_out, std::string _name) {
    this->pool = _pool;
    this->n_in = _n_in;
    this->n_out = _n_out;
    this->name = _name;

    this->w = ffml_tensor_create(2, { this->n_out, this->n_in, 0, 0 }, (name + "_w").c_str());
    this->w->op = FFML_OP_INIT_RND_UNIFORM;
    this->b = ffml_tensor_create(1, {this->n_out, 0, 0, 0}, (name + "_b").c_str());
    this->b->op = FFML_OP_INIT_RND_UNIFORM;

    this->wt = ffml_unary_op(FFML_OP_TRANSPOSE, this->w);
    ffml_set_name(wt, this->name + "_wt");

    this->parameters.push_back(this->w);
    this->parameters.push_back(this->b);
}

ffml_tensor* Layer::call(ffml_tensor* inputs) {
    assert(inputs->n_dims == 1);

    // matmul
    ffml_tensor* inputs_unsqueezed = ffml_unary_op(FFML_OP_UNSQUEEZE, inputs);
    ffml_set_name(inputs_unsqueezed, this->name + "_iu");
    ffml_tensor* matmul = ffml_op(FFML_OP_MATMUL, inputs_unsqueezed, wt);
    ffml_set_name(matmul, this->name + "_mm");
    ffml_tensor* matmul_plus_b = ffml_op(FFML_OP_ADD, matmul, this->b);
    ffml_set_name(matmul_plus_b, this->name + "_mm_b");

    ffml_tensor* squeezed = ffml_unary_op(FFML_OP_SQUEEZE, matmul_plus_b);
    ffml_set_name(squeezed, this->name + "_lout");

    return squeezed;
}

std::vector<ffml_tensor*>::const_iterator Layer::begin() {
    return this->parameters.begin();
}

std::vector<ffml_tensor*>::const_iterator Layer::end() {
    return this->parameters.end();
}