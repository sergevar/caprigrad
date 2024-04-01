#ifndef DATA_SOURCE_H
#define DATA_SOURCE_H

template <typename TX, typename TY, typename RX, typename RY>
class DataSource {
public:
    uint64_t size;

    std::vector<TX> * Xs;
    std::vector<TY> * Ys;

    // transform Fs (for x and y)
    // std::function<std::vector<std::vector<float>>(TX &)> transformX;
    std::function<RX(TX &)> transformX;
    std::function<RY(TY &)> transformY;

    DataSource(std::vector<TX> * Xs, std::vector<TY> * Ys, std::function<RX(TX &)> transformX, std::function<RY(TY &)> transformY) {
        this->Xs = Xs;
        this->Ys = Ys;
        this->size = Xs->size();
        assert(this->size == Ys->size());

        this->transformX = transformX;
        this->transformY = transformY;
    }

    RX getX(uint64_t idx) {
        assert(idx < this->size);
        TX x = this->Xs->at(idx);
        return this->transformX(x);
    }

    RY getY(uint64_t idx) {
        assert(idx < this->size);
        TY y = this->Ys->at(idx);
        return this->transformY(y);
    }
};

#endif