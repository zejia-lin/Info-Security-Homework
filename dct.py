import numpy as np
import cv2
import subprocess
import time
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

np.set_printoptions(3, suppress=True)

N = 6402

a = np.array(range(N * N)).reshape(N, N).astype(np.float32)
cpu = np.zeros_like(a)

st_time = time.time()
for i in range(N // 3):
    for j in range(N // 3):
        cpu[i*3:i*3+3, j*3:j*3+3] = cv2.dct(a[i*3:i*3+3, j*3:j*3+3])
ed_time = time.time()

print("Finish CPU")
# print(f"CPU time: {(ed_time - st_time) * 1000} ms")

subprocess.run(f"nvcc mydct.cu -o build/mydct".split()).check_returncode()
subprocess.run(f"build/mydct {N}".split()).check_returncode()

print(f"CPU time: {(ed_time - st_time) * 1000} ms")

gpu = np.fromfile('./out/gpu_9.bin', dtype=np.float32).reshape(N, N)

# print('\nA\n', a)
# print('\ncpu\n', cpu)
# print('\ngpu\n', gpu)

print("MSE: ", mean_squared_error(gpu, cpu))
print("SSIM: ", ssim(gpu, cpu))
print(pd.DataFrame(np.sqrt((gpu - cpu) ** 2).flatten()).describe())

gpu_inv = cv2.idct(gpu)
print("MSE: ", mean_squared_error(gpu_inv, a))
print("SSIM: ", ssim(gpu_inv, a))
print(pd.DataFrame(np.sqrt((gpu_inv - a) ** 2).flatten()).describe())


# print(a[-3:, -3:])
