import numpy as np
import cv2
import subprocess
import os
import sys
import tqdm
import time


basedir = os.path.dirname(__file__)
picdir = os.path.join(basedir, 'pic')
atkdir = os.path.join(basedir, 'attack')
outdir = os.path.join(basedir, 'out')
wmdir = os.path.join(basedir, 'wm')

picnames = [os.path.join(picdir, i) for i in os.listdir(picdir)]
wmname = os.path.join(wmdir, 'wm.png')

batchSize = 8
server = subprocess.Popen(f'../build/WmCLI {batchSize}', shell=True,
                    stdin=subprocess.PIPE, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT)

end2end_st = time.time()
for _ in range(batchSize + 1):
    print(server.stdout.readline().decode('utf-8'), end='')

compute_st = time.time()
for picname in picnames:
    bn = os.path.basename(picname).split('.')
    ob = f"{bn[0]}-out.{bn[1]}"
    outname = os.path.join(outdir, ob)
    cmd = f"embed {picname} {wmname} {outname}\n".encode('utf-8')
    server.stdin.write(cmd)
    server.stdin.flush()

stdout_data, stderr_data = server.communicate()
end2end_ed = time.time()
print(stdout_data.decode('utf-8'))
print(f'End to end time: {(end2end_ed - end2end_st) * 1000} ms')
print(f'Computation time: {(end2end_ed - compute_st) * 1000} ms')