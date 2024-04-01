#include "ConvMLP.h"

/*

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

*/

ConvMLP::ConvMLP(ffml_memory_pool* _pool, int _n_in_channels, int _n_in_x, int _n_in_y, std::vector<ConvLayerConfiguration> _layer_configurations) {
    this->pool = _pool;

    this->n_in_channels = _n_in_channels;
    this->n_in_x = _n_in_x;
    this->n_in_y = _n_in_y;
    this->layer_configurations = _layer_configurations;

    int layer_in_channels = this->n_in_channels;
    int layer_in_x = this->n_in_x;
    int layer_in_y = this->n_in_y;
    for(int i = 0; i < this->layer_configurations.size(); i++) {
        // layer_in = layer_out;
        // layer_out = this->n_hidden[i];
        auto layer_configuration = this->layer_configurations[i];

        auto convlayer = new ConvLayer(pool, layer_in_channels, layer_in_x, layer_in_y, layer_configuration.n_filters, layer_configuration.kernel_size, "convlayer" + std::to_string(i));
        this->convlayers.push_back(convlayer);

        for (auto parameter : convlayer->parameters) {
            this->parameters.push_back(parameter);
        }

        auto maxpool = new MaxPoolLayer(pool, convlayer->n_filters, convlayer->out_x, convlayer->out_y, layer_configuration.pool_size, "maxpool" + std::to_string(i));
        this->maxpoollayers.push_back(maxpool);

        for (auto parameter : maxpool->parameters) {
            this->parameters.push_back(parameter);
        }

        layer_in_channels = convlayer->n_filters;
        layer_in_x = maxpool->out_x;
        layer_in_y = maxpool->out_y;
    }

    // auto layer1 = new Layer(pool, inputs, 4, "layer1");
    // auto layer2 = new Layer(pool, layer1_out, 4, "layer2");
    // auto layer3 = new Layer(pool, layer2_out, 1, "layer3");

}

ffml_tensor* ConvMLP::call(ffml_tensor* inputs) {
    ffml_tensor* l = inputs;
    for(int i = 0; i < this->layer_configurations.size(); i++) {

        // printf("57\n");

        // ffml_debug_print_tensor_metadata(l);

        l = this->convlayers[i]->call(l);

        // printf("63\n");

        // ffml_debug_print_tensor_metadata(l);

        l = this->maxpoollayers[i]->call(l);

        // printf("69\n");

        // ffml_debug_print_tensor_metadata(l);

        // ffml_tensor* act = ffml_unary_op(FFML_OP_LEAKY_RELU, l);
        // ffml_set_name(act, this->layers[i]->name + "_lrelu");
        // ffml_tensor* act = ffml_unary_op(FFML_OP_TANH, l);
        // ffml_set_name(act, this->convlayers[i]->name + "_tanh");
        ffml_tensor* act = ffml_unary_op(FFML_OP_SIGMOID, l);
        ffml_set_name(act, this->convlayers[i]->name + "_sigmoid");
        // l = act;

        // if (i != this->layers.size() - 1) {
            // ffml_tensor* relu = ffml_unary_op(FFML_OP_TANH, l);
            // ffml_tensor* relu = ffml_unary_op(FFML_OP_RELU, l);
            // ffml_tensor* relu = ffml_unary_op(FFML_OP_LEAKY_RELU, l);
            // ffml_tensor* relu = ffml_unary_op(FFML_OP_SIGMOID, l);
            // ffml_set_name(relu, this->layers[i]->name + "_act");
            // l = relu;
        // }

    }
    return l;
}

// std::vector<ConvLayer*>::iterator ConvMLP::begin() {
//     return this->layers.begin();
// }

// std::vector<ConvLayer*>::iterator ConvMLP::end() {
//     return this->layers.end();
// }
