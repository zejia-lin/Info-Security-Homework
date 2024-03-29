import numpy as np
import pandas as pd
import cv2
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity
import tqdm
import sys

from test_tool import *

def make_report(watermark_name):
    basedir = os.path.dirname(__file__)
    picdir = os.path.join(basedir, 'pic')
    outdir = os.path.join(basedir, 'out')
    bindir = os.path.join(basedir, 'out/thresh')
    emdir = os.path.join(basedir, 'out/embeded')
    exdir = os.path.join(basedir, 'out/extracted')
    wmdir = os.path.join(basedir, 'wm')
    wmname = os.path.join(wmdir, f'{watermark_name}')
    wmmat = cv2.imread(wmname)
    wmmat = cv2.cvtColor(wmmat, cv2.COLOR_BGR2GRAY)
    cv2.threshold(wmmat, 127, 255, cv2.THRESH_OTSU, wmmat)
    cv2.imwrite(os.path.join(bindir, '000AAAorigin.jpg'), wmmat)
    psnr_df = pd.DataFrame(columns=['method', 'pic', 'psnr', 'ssim'])
    mse_df = pd.DataFrame(columns=['method', 'pic', 'attack', 'ssim'])

    for emname in tqdm.tqdm(os.listdir(emdir)):
        em = cv2.imread(os.path.join(emdir, emname))
        bn, ext = os.path.basename(emname).split('.')
        pname = '-'.join(bn.split('-')[:-1])
        pname = f"{pname}.{ext}"
        ori = cv2.imread(os.path.join(picdir, pname))
        psnr_df.loc[len(psnr_df)] = ['ours', emname, psnr(em, ori), structural_similarity(em, ori, channel_axis=2)]

    psnr_df.sort_values(['pic', 'psnr'], inplace=True)
    psnr_df.to_csv(os.path.join(outdir, 'psnr.csv'))


    for exname in tqdm.tqdm(os.listdir(exdir)):
        ex = cv2.imread(os.path.join(exdir, exname))
        ex = cv2.cvtColor(ex, cv2.COLOR_BGR2GRAY)
        cv2.threshold(ex, 127, 255, cv2.THRESH_OTSU, ex)
        cv2.imwrite(os.path.join(bindir, exname), ex)
        bn, ext = os.path.basename(exname).split('.')
        atk = bn.split('-')[-2]
        pname = '-'.join(bn.split('-')[:-3])
        pname = f"{pname}.{ext}"
        mse_df.loc[len(mse_df)] = ['ours', pname, atk, structural_similarity(ex, wmmat)]


    mse_df.sort_values(['pic', 'attack', 'ssim'], inplace=True)
    mse_df.to_csv(os.path.join(outdir, 'ssim.csv'))
    
    print(mse_df.describe())
    print(psnr_df.describe())


if __name__ == '__main__':
    make_report(sys.argv[1])
