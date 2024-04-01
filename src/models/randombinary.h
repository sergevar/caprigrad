// #include <stdio.h>
// #include "../dataloaders/DataSource.h"
// #include "../dataloaders/DataLoader.h"
// #include <functional>

// void randombinary() {
//     auto transformX = std::bind(float_direct_pass, std::placeholders::_1);
//     auto transformY = std::bind(one_hot, VOCAB_SIZE, std::placeholders::_1);
//     auto dataSource = new DataSource<std::vector<int>, int, std::vector<float>, std::vector<float>>(&xs, &ys, transformX, transformY);
//     auto dataLoader = new DataLoader<std::vector<int>, int, std::vector<float>, std::vector<float>>(dataSource, BATCH_SIZE, DataSplitConfig({0.8, 0.1}));
// }