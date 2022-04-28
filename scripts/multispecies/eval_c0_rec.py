

import numpy as np
import os,subprocess
import nibabel as nib

import sys
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../../'
sys.path.append(code_dir)
from scripts.utils.file_io import readNetCDF, createNetCDF 

#res_path = os.path.join(code_dir, 'syndata','160', 'ed_inv')

for i in range(1,5):
  prefix1 = '/scratch1/07544/ghafouri/results/'
  #prefix = '/scratch1/07544/ghafouri/results/syn_results/C1/til_inv/case%d/inversion/nx256/obs-1.0/'%i
  #prefix = '/scratch1/07544/ghafouri/results/syn_results/C1/til_inv_tc=nec/case%d/inversion/nx256/obs-1.0/'%i
  prefix = '/scratch1/07544/ghafouri/results/syn_results/me_inv/case%d/reg/case%d/case%d_c0Recon_transported.nii.gz'%(i,i,i)
  syn = 'case%d'%i
  #data_path = os.path.join(prefix1, 'syndata',syn, '256', 'c_t0.nii.gz')
  #data_path = os.path.join(prefix1, 'syndata',syn, '', 'c_t0.nii.gz')
  data_path = os.path.join(prefix1, 'syndata', syn, 'C1_me', 'c_t0.nii.gz')
  #data_path = os.path.join(prefix1, 'syndata', syn, 'C1', 'c_t0.nii.gz')
  #res_path = os.path.join(prefix, syn, 'reg', syn, syn+'_c0Recon_transported.nii.gz')
  res_path = os.path.join(prefix)
  #res_path = os.path.join(prefix, 'c0_rec_256256256.nii.gz')

  dat = nib.load(data_path).get_fdata().flatten()
  dat_rec = nib.load(res_path).get_fdata().flatten()


  err = np.linalg.norm((dat - dat_rec))
  rel_err = err / np.linalg.norm(dat)

  coord = np.argwhere(dat_rec > 0)

  print(coord.shape[0])
  print(err)
  print(rel_err)













