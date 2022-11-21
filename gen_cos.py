import numpy as np

N = 4


for i in range(N):
    for j in range(N):
        print("{0:.52f}".format(np.cos((2 * i + 1) * j * np.pi / (2 * N))), end=', ')
        if j == N - 1:
            print()
