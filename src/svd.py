import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import subprocess


np.set_printoptions(3, suppress=True)

TILE = 4
rows = 4
cols = 4


def tiled_svd(_A, _U, _S, _VT):
    for i in range(rows // TILE):
        for j in range(cols // TILE):
            si = slice(i*TILE, i*TILE+TILE)
            sj = slice(j*TILE, j*TILE+TILE)
            ss = slice((i * (rows // TILE) + j) * TILE, (i * (rows // TILE) + j) * TILE + TILE)
            # print(_A[si, sj])
            _U[si, sj], _S[ss], _VT[si, sj] = np.linalg.svd(_A[si, sj])


def tiled_gemm(A, U, S, VT, trans):
    for i in range(rows // TILE):
        for j in range(cols // TILE):
            si = slice(i*TILE, i*TILE+TILE)
            sj = slice(j*TILE, j*TILE+TILE)
            ss = slice((i * (rows // TILE) + j) * TILE, (i * (rows // TILE) + j) * TILE + TILE)
            if trans:
                A[si, sj] = U[si, sj] @ np.diag(S[ss]) @ VT[si, sj].T
            else:
                A[si, sj] = U[si, sj] @ np.diag(S[ss]) @ VT[si, sj]



np.random.seed(42)
A = np.array(np.arange(rows * cols)).reshape(rows, cols).astype(np.float32)
cpu_U = np.zeros_like(A)
cpu_VT = np.zeros_like(A)
cpu_S = np.zeros(rows * cols // TILE)
A.tofile('../out/A.bin')

print(A)

subprocess.run("sh ../script/bwm.sh bwm.cu".split()).check_returncode()
subprocess.run(f"../build/bwm {rows} {cols}".split())

tiled_svd(A, cpu_U, cpu_S, cpu_VT)

gpu_U = np.fromfile("../out/U.bin", dtype=np.float32).reshape(rows, cols)
gpu_V = np.fromfile("../out/V.bin", dtype=np.float32).reshape(rows, cols)
gpu_S = np.fromfile("../out/S.bin", dtype=np.float32)

print("CPU U")
print(cpu_U, end='\n\n')
print("GPU U")
print(gpu_U, end='\n\n')
print("CPU V")
print(cpu_VT, end='\n\n')
print("GPU V")
print(gpu_V, end='\n\n')
print(cpu_S, end='\n\n')
print(gpu_S, end='\n\n')


gpu_inv = np.zeros_like(A)
cpu_inv = np.zeros_like(A)
tiled_gemm(gpu_inv, gpu_V, gpu_S, gpu_U, True)
tiled_gemm(cpu_inv, cpu_U, cpu_S, cpu_VT, False)

print("Inverse GPU")
print(gpu_inv)

print("Inverse CPU")
print(cpu_inv)


slc = slice(0, 4)
# print()

# print(A[slc, slc])
# print(gpu_U[slc, slc] @ np.diag(gpu_S[slc]) @ gpu_V[slc, slc])
# print(cpu_U[slc, slc] @ np.diag(cpu_S[slc]) @ cpu_VT[slc, slc])

print(f"GPU vs CPU: {mean_squared_error(gpu_inv[slc, slc], cpu_inv[slc, slc])}")
print(f"GPU vs Origin: {mean_squared_error(gpu_inv, A)}")
print(f"CPU vs Origin: {mean_squared_error(cpu_inv, A)}")




