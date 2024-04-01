#ifndef TESTS_COMMON
#define TESTS_COMMON

#include <iostream>
#include <cmath>
#include "../src/ffml/ffml.h"

int total_tests = 0;
int fails = 0;

void test_assert(bool condition, const char* message) {
    if (!condition) {
        printf("❌  Assertion failed: %s\n", message);
        fails++;
    }
}

template <typename T>
void test_equal(T a, T b) {
    if (a != b) {
        std::cout << "❌  Assertion failed: " << a << " != " << b << std::endl;
        fails++;
    } else {
        std::cout << "✔  " << a << " == " << b << std::endl;
    }

    total_tests++;
}

float almost_round(float a) {
    return std::round(a * 10000.0f) / 10000.0f;
}

bool almost_same(float a, float b) {
    float max = std::max(std::fabs(a), std::fabs(b));
    float margin = max * 10.0f * 0.001f;
    if (margin < 0.00000000001f) margin = 0.00000000001f;
    // std::cout << "a: " << a << " b: " << b << std::endl;
    // std::cout << "fabs(a - b): " << std::fabs(a - b) << std::endl;
    // std::cout << "fabs(a - b) < margin: " << (std::fabs(a - b) < margin) << std::endl;
    // std::cout << "max: " << max << std::endl;
    // std::cout << "margin: " << margin << std::endl;
    return (std::fabs(a - b)) < margin;
}

void test_tensor_data_flat_almost_equal(ffml_tensor * t, std::vector<float> data) {
    for (uint64_t i = 0; i < data.size(); i++) {
        float tensor_data = ffml_get_data_flat(t, i);
        float expected_data = data[i];
        if (! almost_same((float)tensor_data, (float)expected_data)) {
            std::cout << std::endl;
            std::cout << "❌  Assertion failed in test_tensor_data_flat_equal: " << tensor_data << " != " << expected_data << " at index " << i << std::endl;
            fails++;
        } else {
            std::cout << "✔  " << tensor_data << " == " << expected_data << " ";
        }

        total_tests++;
    }
    std::cout << std::endl;
}

void test_tensor_grad_flat_almost_equal(ffml_tensor * t, std::vector<float> data) {
    for (uint64_t i = 0; i < data.size(); i++) {
        float tensor_data = ffml_get_grad_flat(t, i);
        float expected_data = data[i];
        if (! almost_same((float)tensor_data, (float)expected_data)) {
            std::cout << std::endl;
            std::cout << "❌  Assertion failed in test_tensor_grad_flat_equal: " << tensor_data << " != " << expected_data << " at index " << i << std::endl;
            fails++;
        } else {
            std::cout << "✔  " << tensor_data << " == " << expected_data << " ";
        }

        total_tests++;
    }
    std::cout << std::endl;
}

void test_almost_equal(float a, float b) {
    if (! almost_same(a, b)) {
        std::cout << "❌  Assertion failed: " << a << " != " << b << std::endl;
        fails++;
    } else {
        std::cout << "✔  " << a << " ~ " << b << std::endl;
    }

    total_tests++;
}

void test_almost_equal_grad(struct ffml_tensor * tensor, Coord coord, float grad) {
    if (! almost_same(ffml_get_grad(tensor, coord), grad)) {
        std::cout << "❌  Assertion failed: Tensor '" << tensor->name << "'{" << coord.dim[0] << "," << coord.dim[1] << "," << coord.dim[2] << "," << coord.dim[3] << "} grad " << ffml_get_grad(tensor, coord) << " != " << grad << std::endl;
        fails++;
    } else {
        std::cout << "✔ T<" << tensor->name << ">{" << coord.dim[0] << "," << coord.dim[1] << "," << coord.dim[2] << "," << coord.dim[3] << "} grad " << ffml_get_grad(tensor, coord) << " ~ " << grad << std::endl;
    }

    total_tests++;
}

void test_name(const char * name) {
    std::cout << std::endl << name << std::endl;
}

void test_print_results() {
    std::cout << std::endl << "Total tests: " << total_tests << std::endl;
    if (fails == 0) {
        std::cout << "✅︎  All tests passed!" << std::endl;
    } else {
        std::cout << "❌  Fails: " << fails << std::endl;
    }
}

#endif