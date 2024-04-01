#ifndef LEARNING_RATE_SCHEDULER_H
#define LEARNING_RATE_SCHEDULER_H

class LearningRateScheduler {
public:
    float LR;

    void setSame(const float _lr) {
        this->LR = _lr;
    }

    void setSameBasedOnBatchSize(const float per_one, const int batch_size) {
        this->LR = per_one * batch_size;
    }

    const float getLR() {
        return this->LR;
    }
};

#endif