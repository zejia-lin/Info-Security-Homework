import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import subprocess


np.set_printoptions(3, suppress=True)

TILE = 4
rows = 4
cols = 4


def tiled_svd(A, U, S, VT):
    for i in range(rows // TILE):
        for j in range(cols // TILE):
            si = slice(i*TILE, i*TILE+TILE)
            sj = slice(j*TILE, j*TILE+TILE)
            U[si, sj], S[si], VT[si, sj] = np.linalg.svd(A[si, sj])



np.random.seed(42)
A = np.array(range(rows * cols)).reshape(rows, cols).astype(np.float32)
cpu_U = np.zeros_like(A)
cpu_VT = np.zeros_like(A)
cpu_S = np.zeros(rows * cols // TILE)
A.tofile('../out/A.bin')

print(A)

subprocess.run("sh ../script/bwm.sh bwm.cu".split()).check_returncode()
subprocess.run(f"../build/bwm {rows}".split())

tiled_svd(A, cpu_U, cpu_S, cpu_VT)

gpu_U = np.fromfile("../out/U.bin", dtype=np.float32).reshape(rows, cols)
gpu_VT = np.fromfile("../out/V.bin", dtype=np.float32).reshape(rows, cols).T
gpu_S = np.fromfile("../out/S.bin", dtype=np.float32)

print("="*20)
print(cpu_U, end='\n\n')
print(gpu_U, end='\n\n')
print(cpu_VT, end='\n\n')
print(gpu_VT, end='\n\n')
print(cpu_S, end='\n\n')
print(gpu_S, end='\n\n')

gpu_inv = gpu_U @ np.diag(gpu_S) @ gpu_VT
cpu_inv = cpu_U @ np.diag(cpu_S) @ cpu_VT

print(gpu_inv)

print(f"GPU vs CPU: {mean_squared_error(gpu_inv, cpu_inv)}")
print(f"GPU vs Origin: {mean_squared_error(gpu_inv, A)}")
print(f"CPU vs Origin: {mean_squared_error(cpu_inv, A)}")
