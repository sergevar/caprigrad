#include "../tests/common.h"

#include "../tests/test_karpathy_1.h"
#include "../tests/test_karpathy_2.h"
#include "../tests/test_karpathy_3.h"
#include "../tests/test_karpathy_4_exp.h"
#include "../tests/test_all_ops.h"
#include "../tests/test_2d.h"
#include "../tests/test_matmul.h"
#include "../tests/test_broadcasting.h"
#include "../tests/test_broadcasting2.h"
#include "../tests/test_broadcasting3.h"
#include "../tests/test_transpose.h"
#include "../tests/test_softmax.h"
#include "../tests/test_softmax_crossentropy.h"
#include "../tests/test_softmax_crossentropy_batched.h"
#include "../tests/test_conv2d.h"
#include "../tests/test_convolution.h"
#include "../tests/test_uneven_pooling.h"
#include "../tests/test_embedding.h"
#include "../tests/test_matmul_higher_dim.h"
#include "../tests/test_reshape.h"
#include "../tests/test_gelu.h"
#include "../tests/test_gelu_approx.h"
#include "../tests/test_rmsnorm.h"
#include "../tests/test_repeat.h"

int main() {
    std::cout << "Hello, Test!" << std::endl;

    test_rmsnorm();
    test_reshape();
    test_embedding();
    test_broadcasting();
    test_broadcasting2();
    test_broadcasting3();
    test_uneven_pooling();
    test_conv2d();
    test_convolution();
    test_karpathy_1();
    test_karpathy_2();
    test_karpathy_3();
    test_karpathy_4_exp();
    test_all_ops();
    test_2d();
    test_matmul();
    test_matmul_higher_dim();
    test_transpose();
    test_softmax();
    test_softmax_crossentropy();
    test_softmax_crossentropy_batched();
    test_gelu();
    test_gelu_approx();
    test_repeat();

    test_print_results();
}
