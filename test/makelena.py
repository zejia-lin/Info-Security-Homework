import cv2
import numpy as np

N = 3

img = cv2.imread('../pic/lena.png')
out = np.array([np.zeros_like(img)] * N * N).reshape(img.shape[0] * N, img.shape[1] * N, img.shape[2])

rows, cols = img.shape[:2]

for i in range(N):
    for j in range(N):
        out[rows * i: rows * (i + 1), cols * j: cols * (j + 1)] = img

# out[:img.shape[0], :img.shape[1]] = img
# out[img.shape[0]:, :img.shape[1]] = img
# out[:img.shape[0], img.shape[1]:] = img
# out[img.shape[0]:, img.shape[1]:] = img

cv2.imwrite(f'../pic/lena{N * N}.png', out)
