import numpy as np
import cv2
import subprocess
import time
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

np.set_printoptions(3, suppress=True)

N = 1200

a = np.array(range(N * N)).reshape(N, N).astype(np.float32)
cpu = np.ones_like(a)

st_time = time.time()
for i in range(0, N, 3):
    for j in range(0, N, 3):
        cpu[i:i+3, j:j+3] = cv2.dct(a[i:i+3, j:j+3])
ed_time = time.time()

print("Finish CPU")
# print(f"CPU time: {(ed_time - st_time) * 1000} ms")

subprocess.run(f"nvcc mydct.cu -o build/mydct".split()).check_returncode()
subprocess.run(f"build/mydct {N}".split()).check_returncode()

print(f"CPU time: {(ed_time - st_time) * 1000} ms")

gpu = np.fromfile('./out/gpu_9.bin', dtype=np.float32).reshape(N, N)

print('\nA\n', a)
print('\ncpu\n', cpu)
print('\ngpu\n', gpu)

print("MSE: ", mean_squared_error(gpu, cpu))
# print("SSIM: ", ssim(gpu, cpu))

# print(a[-3:, -3:])
