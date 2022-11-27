import numpy as np
import cv2
import subprocess
import time
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

np.set_printoptions(3, suppress=True)

N = 2400
TILE = 4

a = np.array(range(N * N)).reshape(N, N).astype(np.float32).T
cpu = np.zeros_like(a)

st_time = time.time()
for i in range(N // TILE):
    for j in range(N // TILE):
        cpu[i*TILE:i*TILE+TILE, j*TILE:j*TILE+TILE] = cv2.dct(a[i*TILE:i*TILE+TILE, j*TILE:j*TILE+TILE])
ed_time = time.time()

print("Finish CPU dct")

subprocess.run(f"sh ../script/run.sh mydct.cu {N}".split()).check_returncode()

print(f"CPU time: {(ed_time - st_time) * 1000} ms")

gpu = np.fromfile('../out/gpu_dct.bin', dtype=np.float32).reshape(N, N + 1)[:N, :N]

# print('\nA\n', a)
# print('\ncpu\n', cpu)
# print('\ngpu\n', gpu)

print("MSE: ", mean_squared_error(gpu, cpu))
print("SSIM: ", ssim(gpu, cpu))
# print(pd.DataFrame(np.sqrt((gpu - cpu) ** 2).flatten()).describe())


def blocked_idct(A, res):
    for i in range(N // TILE):
        for j in range(N // TILE):
            res[i*TILE:i*TILE+TILE, j*TILE:j*TILE+TILE] = cv2.idct(A[i*TILE:i*TILE+TILE, j*TILE:j*TILE+TILE])

print("======================================================")
gpu_inv = np.zeros_like(gpu)
st_time = time.time()
blocked_idct(gpu, gpu_inv)
ed_time = time.time()
print("Finish CPU inv for GPU dct")
print(f"CPU time: {(ed_time - st_time) * 1000} ms")
print("MSE: ", mean_squared_error(gpu_inv, a))
print("SSIM: ", ssim(gpu_inv, a))

print("======================================================")
cpu_inv = np.zeros_like(gpu)
st_time = time.time()
blocked_idct(cpu, cpu_inv)
ed_time = time.time()
print("Finish CPU inv for CPU dct")
print(f"CPU time: {(ed_time - st_time) * 1000} ms")
print("MSE: ", mean_squared_error(cpu_inv, a))
print("SSIM: ", ssim(cpu_inv, a))

print("======================================================")
gpugpu_inv = np.fromfile('../out/gpu_idct.bin', dtype=np.float32).reshape(N, N + 1)[:N, :N]
print("Finish GPU inv for GPU dct")
print("MSE vs CPU: ", mean_squared_error(gpugpu_inv, gpu_inv))
print("MSE vs Origin: ", mean_squared_error(gpugpu_inv, a))
print("SSIM vs CPU: ", ssim(gpugpu_inv, gpu_inv))
print("SSIM vs Origin: ", ssim(gpugpu_inv, a))

# print(gpugpu_inv)


# print(pd.DataFrame(np.sqrt((gpu_inv - a) ** 2).flatten()).describe())


# print('\nA\n', a)
# print('\ngpu_inv\n', gpu_inv)
