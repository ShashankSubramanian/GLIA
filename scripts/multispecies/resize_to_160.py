import numpy as np

import os,sys


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path + '/../helpers/')

from resize_image import resample 

path = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/syndata/'
out = os.path.join(path, '160')
if not os.path.exists(out):
  os.mkdir(out)


for f in os.listdir(path):
  if '.nc' in f:
    print(f)
    if 'seg' in f:
      resample(os.path.join(path, f), 160, 0)
      fname = f.replace('.nc', '_nx160.nc')
      cmd = 'mv '+os.path.join(path, fname)+' '+os.path.join(path, '160', f)
      os.system(cmd)
    if 'c0' in f:
      resample(os.path.join(path, f), 160, 1)
      fname = f.replace('.nc', '_nx160.nc')
      cmd = 'mv '+os.path.join(path, fname)+' '+os.path.join(path, '160', f)
      os.system(cmd)














