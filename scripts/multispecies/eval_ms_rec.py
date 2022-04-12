

import numpy as np
import os,subprocess
import nibabel as nib

import sys
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../../'
sys.path.append(code_dir)
from scripts.utils.file_io import readNetCDF, createNetCDF 

#res_path = os.path.join(code_dir, 'syndata','160', 'ed_inv')

for i in range(1,2):

  syn = 'case%d'%i
  prefix = '/scratch1/07544/ghafouri/results/'
  data_path = os.path.join(prefix, 'syndata', syn, 'C1')
  res_path = os.path.join(prefix, 'syndata', syn, 'C1_fwd_rec')
  #data_path = os.path.join(prefix, 'syndata', syn, 'C1_me')
  #res_path = os.path.join(prefix, 'syndata', syn, 'C1_me_fwd_rec')
  reffile = os.path.join(prefix, 'syndata', syn, '256', 'template.nii.gz')
  

  ed_true = readNetCDF(os.path.join(data_path, 'ed_t1.nc'))
  ed_rec = readNetCDF(os.path.join(res_path, 'ed_rec_final.nc')) 
  cmd_prefix = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/scripts/helpers/convert_to_nifti.py'
  cmd = 'python '+ cmd_prefix + ' -i '+os.path.join(res_path, 'ed_rec_final.nc') +' -r '+reffile
  os.system(cmd)
 
  en_true = readNetCDF(os.path.join(data_path, 'en_t1.nc'))
  en_rec = readNetCDF(os.path.join(res_path, 'en_rec_final.nc'))
  cmd = 'python '+ cmd_prefix + ' -i '+os.path.join(res_path, 'en_rec_final.nc') +' -r '+reffile
  os.system(cmd)

  nec_true = readNetCDF(os.path.join(data_path, 'nec_t1.nc'))
  nec_rec = readNetCDF(os.path.join(res_path, 'nec_rec_final.nc'))
  cmd = 'python '+ cmd_prefix + ' -i '+os.path.join(res_path, 'nec_rec_final.nc') +' -r '+reffile
  os.system(cmd)
  

  vt_true = readNetCDF(os.path.join(data_path, 'seg_t1.nc'))
  vt_rec = readNetCDF(os.path.join(res_path, 'seg_rec_final.nc'))
  seg_rec = readNetCDF(os.path.join(res_path, 'seg_rec_final.nc'))
  cmd = 'python '+ cmd_prefix + ' -i '+os.path.join(res_path, 'seg_rec_final.nc') +' -r '+reffile
  os.system(cmd)
  vt_true[vt_true !=7] = 0
  vt_true[vt_true ==7] = 1
  vt_rec[vt_rec !=7] = 0
  vt_rec[vt_rec ==7] = 1


  


  ed_abs_err = np.linalg.norm((ed_true - ed_rec).flatten())
  ed_rel_err = ed_abs_err / np.linalg.norm((ed_true).flatten())
  
  ed_abs_err_norm = np.linalg.norm((ed_true/np.amax(ed_true) - ed_rec/np.amax(ed_rec)).flatten()) / np.linalg.norm((ed_true/np.amax(ed_true)).flatten()) 


  nec_abs_err = np.linalg.norm((nec_true - nec_rec).flatten())
  nec_rel_err = nec_abs_err / np.linalg.norm((nec_true).flatten())
  nec_abs_err_norm = np.linalg.norm((nec_true/np.amax(nec_true) - nec_rec/np.amax(nec_rec)).flatten()) / np.linalg.norm((nec_true/np.amax(nec_true)).flatten()) 

  en_abs_err = np.linalg.norm((en_true - en_rec).flatten())
  en_rel_err = en_abs_err / np.linalg.norm((en_true).flatten())
  en_abs_err_norm = np.linalg.norm((en_true/np.amax(en_true) - en_rec/np.amax(en_rec)).flatten()) / np.linalg.norm((en_true/np.amax(en_true)).flatten()) 
  
  vt_abs_err = np.linalg.norm((vt_true - vt_rec).flatten())
  vt_rel_err = vt_abs_err / np.linalg.norm((vt_true).flatten())
  vt_abs_err_norm = np.linalg.norm((vt_true/np.amax(vt_true) - vt_rec/np.amax(vt_rec)).flatten()) / np.linalg.norm((vt_true/np.amax(vt_true)).flatten()) 
  

  total = ed_rel_err + nec_rel_err + en_rel_err + vt_rel_err
  print(syn)
  print('ED : %.4f, %.4f'%(ed_rel_err, ed_abs_err_norm)) 
  print('NEC : %.4f, %.4f'%(en_rel_err, en_abs_err_norm)) 
  print('EN : %.4f, %.4f'%(nec_rel_err, nec_abs_err_norm)) 
  print('VT : %.4f, %.4f'%(vt_rel_err, vt_abs_err_norm)) 
  print('TOTAL: %.4f'%(total)) 


  infl = readNetCDF(os.path.join(res_path, 'i_rec_final.nc'))
  seg_all = seg_rec.copy()
  total = np.maximum(ed_rec, nec_rec)
  total = np.maximum(total, en_rec)

  seg_tmp = np.zeros(seg_all.shape)
  seg_tmp[total == nec_rec] = 1
  seg_tmp[total == en_rec] = 4
  seg_tmp[seg_all != 1] = 0
  #seg_tmp[total == ed] = 2
  tmp1 = (total == ed_rec)
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
  
  
  createNetCDF(os.path.join(res_path, 'seg_all_t1.nc'), seg_all.shape, seg_all)
  cmd = 'python '+ cmd_prefix + ' -i '+os.path.join(res_path, 'seg_all_t1.nc') +' -r '+reffile
  os.system(cmd)
  







