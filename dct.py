import numpy as np
import cv2
import subprocess
import time
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

np.set_printoptions(3, suppress=True)

N = 6

a = np.array(range(N * N)).reshape(N, N).astype(np.float32)
cpu = np.zeros_like(a)

st_time = time.time()
for i in range(N // 3):
    for j in range(N // 3):
        cpu[i*3:i*3+3, j*3:j*3+3] = cv2.dct(a[i*3:i*3+3, j*3:j*3+3])
ed_time = time.time()

print("Finish CPU dct")

subprocess.run(f"nvcc mydct.cu -o build/mydct".split()).check_returncode()
subprocess.run(f"build/mydct {N}".split()).check_returncode()

print(f"CPU time: {(ed_time - st_time) * 1000} ms")

gpu = np.fromfile('./out/gpu_9.bin', dtype=np.float32).reshape(N, N)

# print('\nA\n', a)
# print('\ncpu\n', cpu)
# print('\ngpu\n', gpu)

print("MSE: ", mean_squared_error(gpu, cpu))
print("SSIM: ", ssim(gpu, cpu))
# print(pd.DataFrame(np.sqrt((gpu - cpu) ** 2).flatten()).describe())


def blocked_idct(A, res):
    for i in range(N // 3):
        for j in range(N // 3):
            res[i*3:i*3+3, j*3:j*3+3] = cv2.idct(A[i*3:i*3+3, j*3:j*3+3])

print("======================================================")
gpu_inv = np.zeros_like(gpu)
st_time = time.time()
blocked_idct(gpu, gpu_inv)
ed_time = time.time()
print("Finish CPU inv")
print(f"CPU time: {(ed_time - st_time) * 1000} ms")
print("MSE: ", mean_squared_error(gpu_inv, a))
print("SSIM: ", ssim(gpu_inv, a))

print("======================================================")
cpu_inv = np.zeros_like(gpu)
st_time = time.time()
blocked_idct(cpu, cpu_inv)
ed_time = time.time()
print("Finish CPU inv")
print(f"CPU time: {(ed_time - st_time) * 1000} ms")
print("MSE: ", mean_squared_error(cpu_inv, a))
print("SSIM: ", ssim(cpu_inv, a))


# print(pd.DataFrame(np.sqrt((gpu_inv - a) ** 2).flatten()).describe())


# print('\nA\n', a)
# print('\ngpu_inv\n', gpu_inv)
