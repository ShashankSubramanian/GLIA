

import numpy as np
import os,subprocess

import sys
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../../'
sys.path.append(code_dir)
from scripts.utils.file_io import readNetCDF, createNetCDF 

#res_path = os.path.join(code_dir, 'syndata','160', 'ed_inv')
res_path = os.path.join(code_dir, 'syndata','160', 'tc_inv')
data_path = os.path.join(code_dir, 'syndata', '160')
if not os.path.exists(res_path):
  os.mkdir(res_path)



ed = readNetCDF(os.path.join(data_path, 'ed_t1.nc'))
nec = readNetCDF(os.path.join(data_path, 'nec_t1.nc'))
en = readNetCDF(os.path.join(data_path, 'en_t1.nc'))

en[en > 0.01] = 1
nec[nec > 0.01] = 1



total = en + nec


total[total > 1] = 1


fname = os.path.join(res_path, 'data_t1.nc')
createNetCDF(fname, total.shape, total)












