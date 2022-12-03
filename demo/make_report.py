import numpy as np
import pandas as pd
import cv2
from skimage.metrics import mean_squared_error
from test_tool import *

basedir = os.path.dirname(__file__)
picdir = os.path.join(basedir, 'pic')
atkdir = os.path.join(basedir, 'out/attack')
emdir = os.path.join(basedir, 'out/embeded')
exdir = os.path.join(basedir, 'out/extracted')
wmdir = os.path.join(basedir, 'wm')
wmname = os.path.join(wmdir, 'logo-big.jpg')
wmmat = cv2.imread(wmname)
wmrows, wmcols = wmmat.shape[:2]
psnr_df = pd.DataFrame(columns=['method', 'pic', 'psnr'])
mse_df = pd.DataFrame(columns=['method', 'pic', 'attack', 'mse'])


