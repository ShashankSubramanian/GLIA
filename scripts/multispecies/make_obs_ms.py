
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
  vt_path = os.path.join(res_dir, 'vt_rec_final.nc')
  csf_path = os.path.join(res_dir, 'csf_rec_final.nc')
   
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
  vt = readNetCDF(vt_path)
  csf = readNetCDF(csf_path)

  c_vec = en + nec + infl
  
  Oc = HS(c_vec, wm, 128) * HS(c_vec, gm, 128) * HS(c_vec, vt+csf, 128)
  On = HS(nec, en, 128) * HS(nec, infl, 128) * Oc
  Op = HS(en, nec, 128) * HS(en, infl, 128) * Oc
  Ol = HS(infl, i_th, 128) * (1 - Op - On - vt - csf)
  
  tot = Oc + On + Ol
  max_vec = np.maximum(Oc, On)
  max_vec = np.maximum(max_vec, Ol)
  tot[tot >= 1] = 1
  tot[tot <= 0] = 0
  bg = 1 - tot
  seg_ms = seg.copy()
  
  
  seg_ms[((tot > bg) & (max_vec == On))] = 1 
  seg_ms[((tot > bg) & (max_vec == Op))] = 4
  seg_ms[((tot > bg) & (max_vec == Ol))] = 2
 
  createNetCDF(os.path.join(res_dir, 'seg_ms_rec_final.nc'), seg_ms.shape, seg_ms)
  


def HS(vec_0, vec_1, n):
  
   out = (1 / (1 + np.exp(-(n)*(vec_0 - vec_1))))

   return out

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='convert netcdf to nifti', formatter_class= argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-i', '--res_dir', type = str, help = 'path to the results', required = True)
  args = parser.parse_args();

  res_dir = args.res_dir
  
  create_seg_ms(res_dir)
