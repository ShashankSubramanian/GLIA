import matplotlib as mpl
import pandas as pd
import shutil
import math
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import numpy as np
import os, sys, argparse
from netCDF4 import Dataset
import nibabel as nib
from numpy import linalg as la


def read_netcdf(filename):
  '''
  function to read a netcdf image file and return its contents
  '''
  imgfile = Dataset(filename);
  img = imgfile.variables['data'][:]
  imgfile.close();
  return np.transpose(img)

def load_species(root_dir):
  c = read_netcdf(os.path.join(root_dir, "c_rec_final.nc"))
  wm = read_netcdf(os.path.join(root_dir, "wm_rec_final.nc"))
  gm = read_netcdf(os.path.join(root_dir, "gm_rec_final.nc"))
  vt = read_netcdf(os.path.join(root_dir, "vt_rec_final.nc"))
  csf = read_netcdf(os.path.join(root_dir, "csf_rec_final.nc"))
  bg = read_netcdf(os.path.join(root_dir, "bg.nc"))
  return c, wm, gm, vt, csf, bg

def compute_dice(x, gt):
  smooth = 0 
  intersection = np.sum(x * gt)
  return (2 * intersection + smooth)/ (np.sum(x) + np.sum(gt) + smooth)
  

#--------------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='script to perform some post-inversion analysis',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-p', '--patient_dir', type = str, help = 'path to masseffect patient reconstructions', required = True) 
  r_args.add_argument('-d', '--data_dir', type = str, help = 'path to masseffect patient input data (if different from patient_dir)') 
  r_args.add_argument('-x', '--results_dir', type = str, help = 'results path', required = True) 
  r_args.add_argument('-s', '--survival_file', type = str, help = 'path to survival csv', required = True) 
  args = parser.parse_args();

  me_res_dir = args.patient_dir
  if not args.data_dir:
    data_dir = me_res_dir
  else:
    data_dir = args.data_dir
  temp = os.listdir(me_res_dir)
  for tmp in temp:
    if "csv" in tmp:
      stat_file_name = tmp
      break
  stat_file  = os.path.join(me_res_dir, stat_file_name) 
  stats      = pd.read_csv(stat_file, header=0)
#  stats.reset_index(inplace=True)

  patient_list = stats['PATIENT'].tolist()
  ### add some important columns
  if not 'diff-vol' in stats:
    stats['diff-vol'] = stats['mu-vol-err-nm'] - stats['mu-vol-err']
    stats['diff-l2'] = stats['mu-l2-err-nm'] - stats['mu-l2-err']

  ### append survival data
  if 'survival' not in stats:
    survival = pd.read_csv(args.survival_file, header=0)
    stats['survival'] = np.nan
    stats['age'] = np.nan
    for pat in patient_list:
      data_exists = survival['ID'].str.contains(pat).any()
      if data_exists:
        stats.loc[stats['PATIENT'] == pat, 'survival'] = survival.loc[survival['ID'] == pat]['Survival_days'].values[0]
        stats.loc[stats['PATIENT'] == pat, 'age'] = survival.loc[survival['ID'] == pat]['Age_at_MRI'].values[0]

  if 'mu-err-rsc' not in stats:
    stats['mu-err-rsc'] = 1
    stats['std-err-rsc'] = 1
    stats['mu-dice'] = 0
    stats['std-dice'] = 0
##  patient_list = [patient_list[0]]
    for pat_idx, pat in enumerate(patient_list):
      print("{}: patient = {}".format(pat_idx, pat))
      inv_dir = os.path.join(*[me_res_dir, pat, "tu/160"])
      inv_data_dir = os.path.join(*[data_dir, pat, "tu/160"])
      pat_seg = read_netcdf(os.path.join(inv_data_dir, pat + "_seg_ants_aff2jakob_160.nc"))
      tc = np.logical_or(pat_seg == 1, pat_seg == 4)
#    tc_norm = la.norm(tc[:])
      err_rsc = []
      dice = []
      pat_compute = True
      for at in os.listdir(inv_dir):
        config_file = os.path.join(inv_dir, at) + "/solver_config.txt"
        if not os.path.exists(config_file):
          continue
        at_dir = os.path.join(inv_dir, at)
        c = read_netcdf(os.path.join(at_dir, "c_rec_final.nc"))
        seg = read_netcdf(os.path.join(at_dir, "seg_rec_final.nc"))
        tu = (seg == 1)
        if pat_compute:
          c_data = read_netcdf(os.path.join(at_dir, "data.nc"))
          obs    = read_netcdf(os.path.join(at_dir, "obs.nc"))
          data_norm = la.norm(c_data[:])
          pat_compute = False
        c = obs * c
        tu = obs * tu
        c_rsc = c / np.max(c[:])
        err_rsc.append(la.norm(c_rsc[:] - c_data[:])/data_norm)
        dice.append(compute_dice(tu, tc))
        #c, wm, gm, vt, csf, bg = load_species(at_dir)
        #healthy = wm + gm + vt + csf
        #total = bg + healthy + c
        #prob_c = c / total
      a_err = np.asarray(err_rsc)
      a_dc  = np.asarray(dice)
      stats.loc[stats['PATIENT'] == pat, 'mu-err-rsc'] = np.mean(a_err)
      stats.loc[stats['PATIENT'] == pat, 'std-err-rsc'] = np.std(a_err)
      stats.loc[stats['PATIENT'] == pat, 'mu-dice'] = np.mean(a_dc)
      stats.loc[stats['PATIENT'] == pat, 'std-dice'] = np.std(a_dc)

  if 'mu-vt-dice' not in stats:
    stats['mu-vt-dice'] = 0
    stats['std-vt-dice'] = 0
    for pat_idx, pat in enumerate(patient_list):
      print("{}: patient = {}".format(pat_idx, pat))
      inv_dir = os.path.join(*[me_res_dir, pat, "tu/160"])
      inv_data_dir = os.path.join(*[data_dir, pat, "tu/160"])
      pat_seg = read_netcdf(os.path.join(inv_data_dir, pat + "_seg_ants_aff2jakob_160.nc"))
      vt_pat = (pat_seg == 7) 
      dice = []
      pat_compute = True
      for at in os.listdir(inv_dir):
        config_file = os.path.join(inv_dir, at) + "/solver_config.txt"
        if not os.path.exists(config_file):
          continue
        at_dir = os.path.join(inv_dir, at)
        seg = read_netcdf(os.path.join(at_dir, "seg_rec_final.nc"))
        vt = (seg == 7)
        if pat_compute:
          obs    = read_netcdf(os.path.join(at_dir, "obs.nc"))
          pat_compute = False
        vt = obs * vt
        dice.append(compute_dice(vt, vt_pat))
      a_dc  = np.asarray(dice)
      stats.loc[stats['PATIENT'] == pat, 'mu-vt-dice'] = np.mean(a_dc)
      stats.loc[stats['PATIENT'] == pat, 'std-vt-dice'] = np.std(a_dc)

stats.to_csv(args.results_dir + "/penn_stats.csv", index=False)
