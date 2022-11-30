import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import subprocess
import time
from pywt import dwt2, idwt2


img = cv2.imread('../pic/violet-big.png')
wm = (cv2.cvtColor(cv2.imread('../pic/wm.png'), cv2.COLOR_BGR2GRAY) > 128).astype(np.uint8)
ca, hvd = [np.array([])] * 3, [np.array([])] * 3

img[img > 245] = 245

print("Image shape", img.shape)
img_shape = img.shape[:2]
rd_shape = (32 * (img.shape[0] // 32), 32 * (img.shape[1] // 32))
wmlen = wm.shape[0] * wm.shape[1]
print("Rd shape", rd_shape)

img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
print("YUV shape", img_YUV.shape)
img_YUV[:, :, 0].astype(np.float32).tofile('../out/img.bin')

for channel in range(3):
    ca[channel], hvd[channel] = dwt2(img_YUV[:rd_shape[0], :rd_shape[1], channel], 'haar')

ca[0].astype(np.float32).tofile('../out/haar.bin')
wm.tofile('../out/wm.bin')
print("CA[0]\n========================\n", ca[0])

subprocess.run("sh ../script/bwm.sh ../test/embed.cu".split(), cwd='../test').check_returncode()
subprocess.run(f"../build/embed {ca[0].shape[1]} {ca[0].shape[0]} {wmlen}".split())

embeded = np.fromfile("../out/embeded.bin", dtype=np.float32).reshape(ca[0].shape)
img_YUV[:rd_shape[0], :rd_shape[1], 0] = idwt2((embeded, hvd[0]), "haar")

embed_img_YUV = img_YUV[:img_shape[0], :img_shape[1]]
embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
embed_img = np.clip(embed_img, a_min=0, a_max=255)

wmget = np.fromfile('../out/wmget.bin', dtype=np.float32).reshape(wm.shape)
wmget_bin = (wmget > 0.5).astype(np.uint8) * 255

cv2.imwrite("../out/embeded.png", embed_img)
cv2.imwrite("../out/origin.png", wm * 255)
cv2.imwrite("../out/wmget.png", wmget * 255)
cv2.imwrite("../out/wmget_bin.png", wmget_bin)

print(f"MSE: {mean_squared_error(wmget_bin, wm * 255)}")
print(f"SSIM: {ssim(wmget_bin, wm * 255)}")
