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
atkdir = os.path.join(basedir, 'out/attack')
emdir = os.path.join(basedir, 'out/embeded')
exdir = os.path.join(basedir, 'out/extracted')
wmdir = os.path.join(basedir, 'wm')
watermark_name = 'logo.jpg'
wmname = os.path.join(wmdir, watermark_name)
wmmat = cv2.imread(wmname)
wmrows, wmcols = wmmat.shape[:2]


############################################################
# Start CLI
############################################################
server_st = time.time()
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
picnames = [os.path.join(picdir, i) for i in os.listdir(picdir)]
time_df = pd.DataFrame(columns=['pic', 'resolution', 'time'])
embed_st = time.time()
for picname in tqdm.tqdm(picnames):
    bn = os.path.basename(picname).split('.')
    ob = f"{bn[0]}-out.{bn[1]}"
    outname = os.path.join(emdir, ob)
    cmd = f"embed {picname} {wmname} {outname}\n".encode('utf-8')
    st = time.time()
    server.stdin.write(cmd)
    server.stdin.flush()
    echo = server.stdout.readline()
    ed = time.time()
    img = cv2.imread(picname)
    time_df.loc[len(time_df)] = [os.path.basename(picname), img.shape[0] * img.shape[1], ed - st]
print(f'Embed time: {(time.time() - embed_st) * 1000} ms')
print(time_df)
time_df.to_csv(os.path.join(outdir, 'time.csv'))


############################################################
# Attack
############################################################
picnames = [os.path.join(emdir, i) for i in os.listdir(emdir)]
for picname in tqdm.tqdm(picnames):
    img = cv2.imread(picname)
    for atype in attack_list:
        atk = attack(img, atype)
        outname = add_suffix(picname, atype, atkdir)
        cv2.imwrite(outname, atk)


############################################################
# Extract
############################################################
picnames = [os.path.join(atkdir, i) for i in os.listdir(atkdir)]
extract_st = time.time()
for picname in tqdm.tqdm(picnames):
    outname = add_suffix(picname, 'wm', exdir)
    cmd = f"extract {wmrows} {wmcols} {picname} {outname}\n".encode('utf-8')
    server.stdin.write(cmd)
    server.stdin.flush()
    echo = server.stdout.readline()
stdout_data, stderr_data = server.communicate()
print(f'Extract time: {(time.time() - extract_st) * 1000} ms')

make_report(watermark_name)
print("Everything saved at", outdir)
