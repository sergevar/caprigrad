# Caprigrad

An experiment to write an autograd engine (mini Pytorch clone) purely in C++, to play with training AI models and backpropagation on a low level

![Caprigrad](doc/screenshot.jpg)

## Progress

- [x] Tensor manipulation library
    - [x] dimensions, coordinates
	- [x] memory management, pool, allocator
    - [x] strides, broadcasting
    - [x] tensor ops - forward and backward pass
    	- [x] FFML_OP_LOOKUP
    	- [x] FFML_OP_DUP
    	- [x] FFML_OP_ADD
    	- [x] FFML_OP_SUB
    	- [x] FFML_OP_MUL
    	- [x] FFML_OP_DIV
    	- [x] FFML_OP_POW
    	- [x] FFML_OP_SQUARE
    	- [x] FFML_OP_NEG
    	- [x] FFML_OP_EXP
    	- [x] FFML_OP_ABS
    	- [x] FFML_OP_MATMUL
    	- [x] FFML_OP_MEAN
    	- [x] FFML_OP_MEAN_BATCHED
    	- [x] FFML_OP_SUM
    	- [x] FFML_OP_TANH
    	- [x] FFML_OP_SIGMOID
    	- [x] FFML_OP_RELU
    	- [x] FFML_OP_GELU
    	- [x] FFML_OP_GELU_APPROX_TANH
    	- [x] FFML_OP_LEAKY_RELU
    	- [x] FFML_OP_TRANSPOSE
    	- [x] FFML_OP_SQUEEZE
    	- [x] FFML_OP_UNSQUEEZE
    	- [x] FFML_OP_SOFTMAX
    	- [x] FFML_OP_SOFTMAX_CROSS_ENTROPY
    	- [x] FFML_OP_INIT_ZEROES
    	- [x] FFML_OP_INIT_ONES
    	- [x] FFML_OP_INIT_FILL
    	- [x] FFML_OP_INIT_RND_UNIFORM
    	- [x] FFML_OP_INIT_RND_NORMAL
    	- [x] FFML_OP_INIT_RND_NORMAL_KAIMING
    	- [x] FFML_OP_INIT_CONSTANT
    	- [x] FFML_OP_INIT_ARANGE
    	- [x] FFML_OP_CONV2D
    	- [x] FFML_OP_MAXPOOL2D
    	- [x] FFML_OP_SELECT
    	- [x] FFML_OP_REPEAT
    	- [x] FFML_OP_RMS_NORM
    - [x] save/load model
    - [x] initializers (see in ops)
    - [x] optimizers
        - [x] default
        - [x] SGD
        - [x] SGD with Momentum
        - [x] AdaBelief
        - [x] Adam
        - [x] AdamW
        - [x] RMSProp
- [x] models
    - [x] toy linear models
    - [x] MNIST
    - [x] MNIST CNN
    - [x] makemore
    - [x] makemore_wavenet
    - [x] language model trained on TinyShakespeare dataset
    - [ ] RNN (a-la RWKV)
    - [ ] GPT-based transformer
    - [ ] MoE
- [x] cli interface
- [x] QT UI
- [x] web server for visualization/debugging/UI
- [x] tests
- [ ] Llama inference (build close to llama.cpp original implementation for reference)
- [ ] CUBL acceleration
- [ ] GPU inference (CUDA/Metal acceleration)
