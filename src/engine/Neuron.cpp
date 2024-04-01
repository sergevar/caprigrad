#include "Neuron.h"

Neuron::Neuron(ffml_memory_pool* _pool, ffml_tensor* _inputs) {
    this->pool = _pool;
    this->inputs = _inputs;
    this->w = ffml_tensor_create(this->inputs->n_dims, this->inputs->ne, "w");
    this->b = ffml_tensor_create(1, {1,0,0,0}, "b"); // todo: shouldn't be scalar? 0dim?
}

ffml_tensor* Neuron::call() {
    // matmul
    ffml_tensor* wt = ffml_unary_op(FFML_OP_TRANSPOSE, this->w);
    ffml_tensor* matmul = ffml_op(FFML_OP_MATMUL, this->inputs, wt);
    ffml_tensor* matmul_plus_b = ffml_op(FFML_OP_ADD, matmul, this->b);

    return matmul_plus_b;
}