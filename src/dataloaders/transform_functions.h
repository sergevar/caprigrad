#ifndef TRANSFORM_FUNCTIONS_H
#define TRANSFORM_FUNCTIONS_H

#include <vector>
#include <functional>

std::function<std::vector<float>(int, int)> one_hot = [](int width, int x) {
    std::vector<float> result;
    result.resize(width);
    for(int i = 0; i < width; i++) {
        result[i] = 0.0f;
    }
    result[x] = 1.0f;
    return result;
};

std::function<std::vector<std::vector<float>>(int, std::vector<int> &)> one_hot_several = [](int width, std::vector<int> &xs) {
    std::vector<std::vector<float>> result;
    result.resize(xs.size());
    for(int i = 0; i < xs.size(); i++) {
        result[i] = one_hot(width, xs[i]);
    }
    return result;
};

std::function<std::vector<float>(std::vector<int> &)> float_direct_pass = [](std::vector<int> &x) {
    std::vector<float> result;
    result.resize(x.size());
    for(int i = 0; i < x.size(); i++) {
        result[i] = (float)x[i];
    }
    return result;
};

#endif