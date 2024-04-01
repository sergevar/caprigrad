#include <assert.h>

#define N 1024

#define A_SIDE N
#define INNER_SIDE N
#define C_SIDE N

#include <stdint.h>
#include <time.h>

// 3x4 @ 4x5 = 3x5

#include <iostream>

#define BLOCK 4

uint64_t nanos() {
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

float A[A_SIDE][INNER_SIDE];
float B[INNER_SIDE][C_SIDE];
float C[A_SIDE][C_SIDE];

int main() {
    assert(N%BLOCK == 0);

    uint64_t start = nanos();

    for (int x = 0; x < A_SIDE; x++) {
	    for (int y = 0; y < INNER_SIDE; y++) {
	        A[x][y] = x*INNER_SIDE+y;
	    }
    }
    for (int x = 0; x < INNER_SIDE; x++) {
	    for (int y = 0; y < C_SIDE; y++) {
            B[x][y] = y*C_SIDE+x;
	    }
    }

    for (int by = 0; by < N; by+=BLOCK) { // for each independent row of the output
        for (int bx = 0; bx < N; bx+=BLOCK) { // for each independent column of the output
            
            for(int y = by; y < by+BLOCK; y++) {
                for (int x = bx; x < bx+BLOCK; x++) {
                    float acc = 0;
            
                    for (int k = 0; k < INNER_SIDE; k++) {  // looping through inner glue, which is [ ;y] for first and [x; ] for second
                        acc += A[y][k] * B[x][k];
                    }

                    C[y][x] = acc;
                }
            }

        }
    }

    uint64_t end = nanos();

    double flop = (2.0*N*N*N);
    double gflop = flop*1e-9;
    double s = (end-start)*1e-9;

    std::cout << "GFLOP/S: " << gflop/s << std::endl;
    std::cout << "Time: " << (end - start) / 1000000 << "ms" << std::endl;

    // std::cout << "A:" << std::endl;
    // for (int x = 0; x < A_SIDE; x++) {
	//     for (int y = 0; y < INNER_SIDE; y++) {
    //         std::cout << A[x][y] << " ";
	//     }
	//     std::cout << std::endl;
    // }
    // std::cout << "B:" << std::endl;
    // for (int x = 0; x < INNER_SIDE; x++) {
    //     for (int y = 0; y < C_SIDE; y++) {
    //         std::cout << B[x][y] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "C:" << std::endl;
    // for (int x = 0; x < A_SIDE; x++) {
    //     for (int y = 0; y < C_SIDE; y++) {
    //         std::cout << C[x][y] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}
