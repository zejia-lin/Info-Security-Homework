import numpy as np
import cv2
import subprocess
import os
import sys
import tqdm
import time
import heapq
import multiprocessing
from skimage.metrics import mean_squared_error
import pandas as pd
from test_tool import *
from make_report import make_report


def embed_async(wid):
    picname = os.path.join(tmpdir, f'lzj_tmp_{wid}.jpg')
    outname = os.path.join(tmpdir, f'lzj_emb_{wid}.jpg')
    cmd = f"embed {picname} {wmname} {outname}\n".encode('utf-8')
    servers[wid].stdin.write(cmd)
    servers[wid].stdin.flush()
    echo = servers[wid].stdout.readline()
    pbar.update(1)


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
WORKERS = 2


############################################################
# Start CLI
############################################################
server_st = time.time()
servers = []
for _ in range(WORKERS):
    servers.append(subprocess.Popen(f'../build/WmCLI', shell=True,
                    stdin=subprocess.PIPE, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT))
    for i in range(2):
        print(servers[_].stdout.readline().decode('utf-8'), end='')
print(f'Startup time: {(time.time() - server_st) * 1000} ms')
print(f"Created {WORKERS} workers")

############################################################
# Init video capture
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
pool = multiprocessing.Pool(WORKERS)
futures = []


while cap.isOpened():
    for wid in range(WORKERS):
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        picname = os.path.join(tmpdir, f'lzj_tmp_{wid}.jpg')
        outname = os.path.join(tmpdir, f'lzj_emb_{wid}.jpg')
        cv2.imwrite(picname, frame)
        future = pool.apply_async(embed_async, (wid,))
        futures.append(future)
    for future in futures:
        future.wait()
    futures = []
    for wid in range(WORKERS):
        img = cv2.imread(os.path.join(tmpdir, f'lzj_emb_{wid}.jpg'))
        video_writer.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
video_writer.release()
pbar.close()
e2eed = time.time()
os.system('rm /tmp/shm/lzj_*')

print("End to end", e2eed - e2est)
print("Embed", emtimeacc)
