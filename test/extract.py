import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import subprocess
import time
from pywt import dwt2, idwt2

img = cv2.imread('../out/embeded.png')
wm = (cv2.cvtColor(cv2.imread('../pic/wm2.png'), cv2.COLOR_BGR2GRAY) > 128).astype(np.uint8)
ca, hvd = [np.array([])] * 3, [np.array([])] * 3

img_shape = img.shape[:2]
wmlen = wm.shape[0] * wm.shape[1]

img_YUV = cv2.copyMakeBorder(cv2.cvtColor(img, cv2.COLOR_BGR2YUV),
                                    0, img.shape[0] % 2, 0, img.shape[1] % 2,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
ca_shape = (img_YUV.shape[0] / 2, img_YUV.shape[1] / 2)
for channel in range(3):
    ca[channel], hvd[channel] = dwt2(img_YUV[:, :, channel], 'haar')

ca[0].astype(np.float32).tofile('../out/haar.bin')
wm.tofile('../out/wm.bin')
print("CA[0]\n========================\n", ca[0])

subprocess.run("sh ../script/bwm.sh ../test/extract.cu".split(), cwd='../test').check_returncode()
subprocess.run(f"../build/extract {ca_shape[0]} {ca_shape[1]} {wmlen}".split())

wmget = np.fromfile('../out/wmget.bin', dtype=np.float32).reshape(wm.shape)

cv2.imwrite("../out/origin.png", wm * 255)
cv2.imwrite("../out/wmget.png", wmget * 255)

print(wmget.shape, wm.shape)

print(f"Extract vs origin: {mean_squared_error(wmget, wm)}")