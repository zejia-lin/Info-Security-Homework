import numpy as np

cosines = np.zeros((9, 9))

for i in range(9):
    for j in range(9):
        print(np.cos((2 * i + 1) * j * np.pi / (2 * 3)), end=', ')
        if j == 8:
            print()
