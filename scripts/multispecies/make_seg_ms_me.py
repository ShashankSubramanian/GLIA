

import numpy as np
import os,subprocess

import sys
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../../'
sys.path.append(code_dir)
from scripts.utils.file_io import readNetCDF, createNetCDF 

#res_path = os.path.join(code_dir, 'syndata','160', 'ed_inv')

prefix='/scratch1/07544/ghafouri/results'
for l in range(5,9):
   
  syn = 'case%d'%l
  #if l == 3 or l == 4:
  #  syn+= ''
  #print(syn)
  res_path = os.path.join(prefix, 'syndata',syn, 'C1_me')
  data_path = os.path.join(prefix, 'syndata',syn, 'C1_me')
  #res_path = os.path.join(prefix, 'syndata',syn, 'C1')
  #data_path = os.path.join(prefix, 'syndata',syn, 'C1')



  ed = readNetCDF(os.path.join(data_path, 'ed_t1.nc'))
  infl = readNetCDF(os.path.join(data_path, 'i_t1.nc'))
  nec = readNetCDF(os.path.join(data_path, 'nec_t1.nc'))
  en = readNetCDF(os.path.join(data_path, 'en_t1.nc'))
  seg = readNetCDF(os.path.join(data_path, 'seg_t1.nc'))


  total = np.maximum(infl, nec)
  total = np.maximum(total, en)
  
  seg_tmp = seg.copy()
 
  ed[seg_tmp == 7] = 0
  ed[seg_tmp == 8] = 0
 
  seg_tmp[ed >= 0.01] = 2
  #seg_tmp[ed >= 0.01] = 2
  
  tmp1 = (total == nec)
  tmp2 = (seg == 1)
  tmp = tmp2 * tmp1
   
  seg_tmp[tmp == 1] = 1

  tmp1 = (total == en)
  tmp2 = (seg == 1)
  tmp = tmp2 * tmp1
   
  seg_tmp[tmp == 1] = 4

  
  
  c_seg = seg.copy()
  c_seg[c_seg != 1] = 0 

 
  createNetCDF(os.path.join(data_path, 'seg_all_t1.nc'), seg_tmp.shape, seg_tmp)

  createNetCDF(os.path.join(data_path, 'tc.nc'), c_seg.shape, c_seg)
  
          











