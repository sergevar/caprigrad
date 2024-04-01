#include "MLP.h"

MLP::MLP(ffml_memory_pool* _pool, int _n_in, std::vector<int> _n_hidden) {
    this->pool = _pool;
    this->n_in = _n_in;
    this->n_hidden = _n_hidden;

    int layer_in = 0;
    int layer_out = this->n_in;
    for(int i = 0; i < this->n_hidden.size(); i++) {
        layer_in = layer_out;
        layer_out = this->n_hidden[i];

        auto layer = new Layer(pool, layer_in, layer_out, "layer" + std::to_string(i));
        this->layers.push_back(layer);

        for (auto parameter : layer->parameters) {
            this->parameters.push_back(parameter);
        }
    }

    // auto layer1 = new Layer(pool, inputs, 4, "layer1");
    // auto layer2 = new Layer(pool, layer1_out, 4, "layer2");
    // auto layer3 = new Layer(pool, layer2_out, 1, "layer3");

}

ffml_tensor* MLP::call(ffml_tensor* inputs) {
    ffml_tensor* l = inputs;
    for(int i = 0; i < this->layers.size(); i++) {
        l = this->layers[i]->call(l);

        // ffml_tensor* act = ffml_unary_op(FFML_OP_RELU, l);
        // ffml_set_name(act, this->layers[i]->name + "_relu");
        // ffml_tensor* act = ffml_unary_op(FFML_OP_LEAKY_RELU, l);
        // ffml_set_name(act, this->layers[i]->name + "_lrelu");
        // ffml_tensor* act = ffml_unary_op(FFML_OP_TANH, l);
        // ffml_set_name(act, this->layers[i]->name + "_tanh");
        // ffml_tensor* act = ffml_unary_op(FFML_OP_SIGMOID, l);
        // ffml_set_name(act, this->layers[i]->name + "_sigmoid");
        // l = act;

        if (i != this->layers.size() - 1) {
            ffml_tensor* relu = ffml_unary_op(FFML_OP_TANH, l);
            // ffml_tensor* relu = ffml_unary_op(FFML_OP_RELU, l);
            // ffml_tensor* relu = ffml_unary_op(FFML_OP_LEAKY_RELU, l);
            // ffml_tensor* relu = ffml_unary_op(FFML_OP_SIGMOID, l);
            ffml_set_name(relu, this->layers[i]->name + "_act");
            l = relu;
        }

    }
    return l;
}

// std::vector<Layer*>::iterator MLP::begin() {
//     return this->layers.begin();
// }

// std::vector<Layer*>::iterator MLP::end() {
//     return this->layers.end();
// }
