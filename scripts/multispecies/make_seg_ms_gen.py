
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
  
  bg_path = os.path.join(res_dir, 'bg.nc') 
  csf_path = os.path.join(res_dir, 'csf_rec_final.nc') 
  vt_path = os.path.join(res_dir, 'vt_rec_final.nc')


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
  bg = readNetCDF(bg_path)
  csf = readNetCDF(csf_path)
  vt = readNetCDF(vt_path)
  
  f = bg + csf + vt 

  c = en + nec + infl

  Oc = np.ones(c.shape)
  Oc[c < wm] = 0.0
  Oc[c < gm] = 0.0
  Oc[c < f] = 0.0

  Op = Oc.copy()
  Op[en < nec] = 0.0
  Op[en < infl] = 0.0
  
  On = Oc.copy()
  On[nec < en] = 0.0
  On[nec < infl] = 0.0

  Ol = 1 - Op - On
  Ol[c < f] = 0.0
  Ol[infl < i_th] = 0.0
  
  
  seg_ms = seg.copy()

  seg_ms[Op == 1] = 4
  seg_ms[On == 1] = 1
  seg_ms[Ol == 1] = 2

  seg_ic = seg_ms.copy()
  seg_ic[seg_ic == 4] = 2
  
  createNetCDF(os.path.join(res_dir, 'seg_ms_rec_final.nc'), seg_ms.shape, seg_ms)
  createNetCDF(os.path.join(res_dir, 'seg_ms_rec_final_ic.nc'), seg_ms.shape, seg_ic)
  
  
  

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='convert netcdf to nifti', formatter_class= argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-i', '--res_dir', type = str, help = 'path to the results', required = True)
  args = parser.parse_args();

  res_dir = args.res_dir
  
  create_seg_ms(res_dir)