

import numpy as np
import os,subprocess

import sys
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../../'
sys.path.append(code_dir)
from scripts.utils.file_io import readNetCDF, createNetCDF 

#res_path = os.path.join(code_dir, 'syndata','160', 'ed_inv')

prefix='/scratch1/07544/ghafouri/results'
for l in range(1,5):
   
  syn = 'case%d'%l
  if l == 3 or l == 4:
    syn+= ''
  print(syn)
  #res_path = os.path.join(prefix, 'syndata',syn, 'C1_me')
  #data_path = os.path.join(prefix, 'syndata',syn, 'C1_me')
  res_path = os.path.join(prefix, 'syndata',syn, 'C1')
  data_path = os.path.join(prefix, 'syndata',syn, 'C1')



  ed = readNetCDF(os.path.join(data_path, 'ed_t1.nc'))
  infl = readNetCDF(os.path.join(data_path, 'i_t1.nc'))
  nec = readNetCDF(os.path.join(data_path, 'nec_t1.nc'))
  en = readNetCDF(os.path.join(data_path, 'en_t1.nc'))
  seg = readNetCDF(os.path.join(data_path, 'seg_t1.nc'))


  seg_all = seg.copy()


  total = np.maximum(ed, nec)
  total = np.maximum(total, en)
  
  seg_tmp = np.zeros(seg_all.shape)
  seg_tmp[total == nec] = 1
  seg_tmp[total == en] = 4
  seg_tmp[seg_all != 1] = 0
  #seg_tmp[total == ed] = 2
  tmp1 = (total == ed)
  tmp2 = (seg_all != 7)
  tmp3 = (seg_all != 8)
  tmp4 = (infl >= 0.01)
  tmp = tmp1 * tmp2 * tmp3 * tmp4
  #seg_tmp[seg_all >= 5] = 0 
  #seg_tmp[seg == 0] = 0
  #seg_all[total == ed] = 0  
  seg_all[seg_all == 1] = 0
  seg_all += seg_tmp
  seg_all[tmp > 0] = 2
  
  createNetCDF(os.path.join(data_path, 'seg_all_t1.nc'), seg_all.shape, seg_all)

  seg_tc_nec = seg_all.copy()
  seg_tc_nec[seg_tc_nec == 4] = 2 
  createNetCDF(os.path.join(data_path, 'seg_all_t1_tc=nec.nc'), seg_all.shape, seg_tc_nec)

  seg_tc_en = seg_all.copy()
  seg_tc_en[seg_tc_en == 1] = 2 
  createNetCDF(os.path.join(data_path, 'seg_all_t1_tc=en.nc'), seg_all.shape, seg_tc_en)
   
  
          











