import numpy as np

N = 3


for i in range(N):
    for j in range(N):
        print(np.cos((2 * i + 1) * j * np.pi / (2 * N)), end=', ')
        if j == N - 1:
            print()
