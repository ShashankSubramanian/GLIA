
import argparse
import numpy as np
import os,subprocess

import sys
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../../'
sys.path.append(code_dir)
from scripts.utils.file_io import readNetCDF, createNetCDF, writeNII 

 
def create_seg_ms(res_dir):
 
  en_path = os.path.join(res_dir, 'en_rec_final.nc')          
  nec_path = os.path.join(res_dir, 'nec_rec_final.nc')          
  i_path = os.path.join(res_dir, 'i_rec_final.nc')  
  config_path = os.path.join(res_dir, 'solver_config.txt')
  seg_path = os.path.join(res_dir, 'seg_rec_final.nc')
  wm_path = os.path.join(res_dir, 'wm_rec_final.nc')
  gm_path = os.path.join(res_dir, 'gm_rec_final.nc')
  
  with open(config_path, 'r') as f:
    lines = f.readlines()
    for l in lines: 
      if 'invasive_thres_data=' in l:
        i_th = float(l.split('invasive_thres_data=')[-1])
  
  en = readNetCDF(en_path)
  nec = readNetCDF(nec_path)
  infl = readNetCDF(i_path)
  seg = readNetCDF(seg_path)
  wm = readNetCDF(wm_path)
  gm = readNetCDF(gm_path)
 
  ed = np.array((infl > i_th)) * (1 - en - nec - infl) 
  total = np.maximum(infl, nec)
  total = np.maximum(total, en)
  
  seg_ms = seg.copy()

  tmp0 = np.array((total == nec))
  tmp1 = np.array((seg == 1))
  tmp = tmp0 * tmp1
  seg_ms[tmp] = 1

  tmp0 = np.array((total == en))
  tmp1 = np.array((seg == 1))
  tmp = tmp0 * tmp1
  seg_ms[tmp] = 4
 
   
  tmp1 = np.array((total == infl))
  tmp2 = np.array((seg != 7))
  tmp3 = np.array((seg != 8))
  tmp4 = np.array((infl > i_th))
  tmp = tmp1 * tmp2 * tmp3 * tmp4 
  seg_ms[tmp] = 2
 
  '''
  tmp0 = np.array((seg != 1))
  tmp1 = np.array((infl > i_th))
  tmp2 = np.array((seg != 7))
  tmp = tmp0 * tmp1 * tmp2
  seg_ms[tmp] = 2
  '''
  tc = (seg == 1)

  createNetCDF(os.path.join(res_dir, 'seg_ms_rec_final.nc'), seg_ms.shape, seg_ms)
  createNetCDF(os.path.join(res_dir, 'tc.nc'), tc.shape, tc)
  
  

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='convert netcdf to nifti', formatter_class= argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-i', '--res_dir', type = str, help = 'path to the results', required = True)
  args = parser.parse_args();

  res_dir = args.res_dir
  
  create_seg_ms(res_dir)
