import numpy as np

import os,sys


file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(file_path + '/../helpers/')

from resize_image import resample 


for i in range(1,5):

  syn = 'case%d'%i

  path = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/syndata/'
  indir = os.path.join(path, syn, '256')
  out = os.path.join(path, syn, '160')
  if not os.path.exists(out):
    os.mkdir(out)

  for f in os.listdir(indir):
    if '.nc' in f:
      print(f)
      if 'seg' in f:
        resample(os.path.join(indir, f), 160, 0)
        fname = f.replace('.nc', '_nx160.nc')
        cmd = 'mv '+os.path.join(indir, fname)+' '+os.path.join(out, f)
        print(cmd)
        os.system(cmd)
      if 'c0' in f:
        resample(os.path.join(indir, f), 160, 1)
        fname = f.replace('.nc', '_nx160.nc')
        cmd = 'mv '+os.path.join(indir, fname)+' '+os.path.join(out,  f)
        print(cmd)
        os.system(cmd)
      














