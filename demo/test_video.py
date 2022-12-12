import numpy as np
import cv2
import subprocess
import os
import sys
import tqdm
import time
from skimage.metrics import mean_squared_error
import pandas as pd
from test_tool import *
from make_report import make_report

basedir = os.path.dirname(__file__)
picdir = os.path.join(basedir, 'pic')
outdir = os.path.join(basedir, 'out')
tmpdir = '/dev/shm/'
wmdir = os.path.join(basedir, 'wm')
video_path = os.path.join(picdir, 'bbb-short.mp4')
output_path = os.path.join(outdir, 'embeded.mp4')
watermark_name = 'logo.jpg'
wmname = os.path.join(wmdir, watermark_name)
wmmat = cv2.imread(wmname)
wmrows, wmcols = wmmat.shape[:2]

############################################################
# Start CLI
############################################################
server_st = time.time()
servers = []
server = subprocess.Popen(f'../build/WmCLI', shell=True,
                    stdin=subprocess.PIPE, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT)
for _ in range(2):
    print(server.stdout.readline().decode('utf-8'), end='')
print(f'Startup time: {(time.time() - server_st) * 1000} ms')


############################################################
# Embed
############################################################
cap = cv2.VideoCapture(video_path)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_size = (int(width), int(height))
fourcc = cap.get(cv2.CAP_PROP_FOURCC)
fps =  cap.get(cv2.CAP_PROP_FPS)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_writer = cv2.VideoWriter(output_path, int(fourcc), fps, frame_size)

count = 0
pbar = tqdm.tqdm(total=length)
e2est = time.time()
emtimeacc = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    pbar.update(1)
    picname = os.path.join(tmpdir, 'lzj_tmp.jpg')
    outname = os.path.join(tmpdir, 'lzj_emb.jpg')
    cv2.imwrite(picname, frame)
    tmpst = time.time()
    cmd = f"embed {picname} {wmname} {outname}\n".encode('utf-8')
    server.stdin.write(cmd)
    server.stdin.flush()
    echo = server.stdout.readline()
    tmped = time.time()
    emtimeacc += tmped - tmpst
    wmed_frame = cv2.imread(outname)
    video_writer.write(wmed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
video_writer.release()
pbar.close()
e2eed = time.time()

print("End to end", e2eed - e2est)
print("Embed", emtimeacc)
