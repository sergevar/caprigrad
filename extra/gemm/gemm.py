#!/usr/bin/env python3
import time
import numpy as np

N = 4096
if __name__ == "__main__":
    A = np.random.randn(N, N).astype(np.float64)
    B = np.random.randn(N, N).astype(np.float64)

    flop = N * N * 2 * N
    print(f"{flop / 1e9:.2f} GFLOP")

    for i in range(100):
        st = time.monotonic()
        C = A @ B
        et = time.monotonic()

        s = et - st

        print(f"{flop/s/1e12:.2f} TFLOP/S")

