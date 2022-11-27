import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import subprocess
import time
from pywt import dwt2, idwt2

img = cv2.imread('../pic/lena.png')
wm = cv2.imread('../pic/wm2.png').astype(np.float32)
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
print("CA[0]\n========================\n", ca[0])

subprocess.run("sh ../script/bwm.sh ../test/embed.cu".split()).check_returncode()
subprocess.run(f"../build/embed {ca_shape[0]} {ca_shape[1]} {wmlen}".split())

embeded = np.fromfile("../out/embeded.bin", dtype=np.float32).reshape(hvd[0][0].shape)
invhaar = idwt2((embeded, hvd[0]), "haar")
img_YUV[:, :, 0] = invhaar
print(invhaar.shape, img_YUV.shape)

embed_img_YUV = img_YUV[:img_shape[0], :img_shape[1]]
embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
embed_img = np.clip(embed_img, a_min=0, a_max=255)

cv2.imwrite("../out/embeded.png", embed_img)
