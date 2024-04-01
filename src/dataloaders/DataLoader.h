#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <cassert>
#include <cstdint>
#include "../ffml/ffml.h"
#include "./DataSource.h"

struct DataSplitConfig {
    float train = 0.8;
    float test = 0.1;
};

struct Split {
    uint64_t start;
    uint64_t end;
};

enum class SplitType {
    TRAIN,
    TEST,
    VAL
};

template <typename TX, typename TY, typename RX, typename RY>
class DataLoader {
protected:
    void setTensorData(ffml_tensor * tensor, uint64_t batch_row, std::vector<float> data) {
        assert(tensor->n_dims == 2);
        assert(tensor->ne[0] == this->batch_size);
        assert(tensor->ne[1] == data.size());

        for(uint64_t i = 0; i < data.size(); i++) {
            ffml_set_data(tensor, {batch_row, i, 0, 0}, data[i]);
        }
    }

    void setTensorData(ffml_tensor * tensor, uint64_t batch_row, std::vector<std::vector<float>> data) {
        assert(tensor->n_dims == 3);
        assert(tensor->ne[0] == this->batch_size);
        assert(tensor->ne[1] == data.size());
        assert(tensor->ne[2] == data[0].size());

        for(uint64_t i = 0; i < data.size(); i++) {
            for(uint64_t j = 0; j < data[i].size(); j++) {
                ffml_set_data(tensor, {batch_row, i, j, 0}, data[i][j]);
            }
        }
    }

    uint64_t getRandomIndexWithinSplit(SplitType splitType) {
        uint64_t split_range = this->splits[(int)splitType].end - this->splits[(int)splitType].start;
        uint64_t idx = this->splits[(int)splitType].start + (rand() % split_range); // todo: allow seed
        return idx;
    }

public:
    uint64_t batch_size;
    DataSplitConfig splitConfig;
    std::vector<Split> splits;
    DataSource<TX, TY, RX, RY> * dataSource;

    uint64_t size() {
        return this->dataSource->size;
    }

    DataLoader(DataSource<TX, TY, RX, RY> * dataSource, uint64_t batch_size, DataSplitConfig splitConfig = DataSplitConfig()) {
        this->dataSource = dataSource;

        this->batch_size = batch_size;
        assert(this->batch_size > 0);
        assert(this->batch_size <= this->dataSource->size);

        this->setSplitConfig(splitConfig);
    }

    void setSplitConfig(DataSplitConfig splitConfig) {
        assert(splitConfig.train + splitConfig.test <= 1.0);
        this->splitConfig = splitConfig;

        this->splits.clear();

        uint64_t train_size = this->size() * this->splitConfig.train;
        uint64_t test_size = this->size() * this->splitConfig.test;
        uint64_t val_size = this->size() - train_size - test_size;

        uint64_t start = 0;
        uint64_t end = train_size;
        this->splits.push_back({start, end});

        start = end;
        end = start + test_size;
        this->splits.push_back({start, end});

        start = end;
        end = start + val_size;
        this->splits.push_back({start, end});

        assert(end == this->size());

        assert(this->splits.size() == 3);
    }

    void loadBatch(SplitType splitType, uint64_t batch_index, ffml_tensor * X_tensor, ffml_tensor * Y_tensor) {
        for(uint64_t i = 0; i < this->batch_size; i++) {
            uint64_t idx = this->getRandomIndexWithinSplit(splitType);

            RX X_data = this->dataSource->getX(idx);
            RY Y_data = this->dataSource->getY(idx);

            this->setTensorData(X_tensor, i, X_data);
            this->setTensorData(Y_tensor, i, Y_data);

            idx++;
        }
    }
};

#endif