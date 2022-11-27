import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import subprocess
import time


np.set_printoptions(3, suppress=True)

TILE = 4
rows = 2400
cols = 2400

wmlen = (rows // TILE) * (cols // TILE)


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
    if trans:
        np.copyto(A, A.T)


rng = np.random.RandomState(1024)
A = np.array(rng.random(rows * cols)).reshape(rows, cols).astype(np.float32) * 100 + 200
# A = np.fromfile("../out/haar.bin", dtype=np.float32).reshape(rows, cols)
wm = np.array(rng.randint(2, size=wmlen)).astype(np.uint8)
cpu_U = np.zeros_like(A)
cpu_VT = np.zeros_like(A)
cpu_S = np.zeros(rows * cols // TILE)
A.tofile('../out/A.bin')
wm.tofile('../out/wm.bin')

wm = np.fromfile("../out/wm.bin", dtype=np.uint8)
print(wm)
print(A)

subprocess.run("sh ../script/bwm.sh ../test/test_svd.cu".split(), cwd='../test').check_returncode()
subprocess.run(f"../build/svd {rows} {cols} {wmlen}".split())

st = time.time()
tiled_svd(A, cpu_U, cpu_S, cpu_VT)
ed = time.time()
print(f"CPU end to end {(ed - st) * 1000} ms")

gpu_inv = np.fromfile('../out/inv.bin', dtype=np.float32).reshape(rows, cols)
cpu_inv = np.zeros_like(A)
tiled_gemm(cpu_inv, cpu_U, cpu_S, cpu_VT, False)

print("Inverse GPU")
print(gpu_inv)

print("Inverse CPU")
print(cpu_inv)

slc = slice(0, 4)

print(f"GPU vs CPU: {mean_squared_error(gpu_inv[slc, slc], cpu_inv[slc, slc])}")
print(f"GPU vs Origin: {mean_squared_error(gpu_inv, A)}")
print(f"CPU vs Origin: {mean_squared_error(cpu_inv, A)}")

wmget = np.fromfile('../out/wmget.bin', dtype=np.uint8)
print(wmget, wm)
print(f"Extracted vs Origin: {mean_squared_error(wmget, wm)}")


