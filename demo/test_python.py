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
sys.path.insert(1, os.path.join(basedir, '../3rd_party'))

from blind_watermark import WaterMark


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
bwm = WaterMark(password_wm=1, password_img=1, processes=64)
bwm.read_wm(wmname)


############################################################
# Embed
############################################################
picnames = [os.path.join(picdir, i) for i in os.listdir(picdir)]
time_df = pd.DataFrame(columns=['pic', 'resolution', 'time'])
embed_st = time.time()
for picname in tqdm.tqdm(picnames):
    try:
        bn = os.path.basename(picname).split('.')
        ob = f"{bn[0]}-out.{bn[1]}"
        outname = os.path.join(emdir, ob)
        st = time.time()
        bwm.read_img(filename=picname)
        bwm.embed(outname)
        ed = time.time()
        img = cv2.imread(picname)
        time_df.loc[len(time_df)] = [os.path.basename(picname), img.shape[0] * img.shape[1], ed - st]
    except Exception as e:
        print(e)
print(f'Embed time: {(time.time() - embed_st) * 1000} ms')
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
    bwm.extract(picname, wm_shape=wmmat.shape[:2], out_wm_name=outname, mode='img')
print(f'Extract time: {(time.time() - extract_st) * 1000} ms')


make_report(wmname)
