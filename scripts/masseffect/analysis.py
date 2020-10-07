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

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../utils/')
from image_tools import compute_volume

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
  
def create_prob_img(img):
  """
    Creates a probabilistic image of a series of segmentations
    Assumes segmentation labels are in standard brats format
  """
  shp = list(img.shape)
  num_atlases = shp[-1]
  num_props = 6 ## bg, tu, gm, wm, vt, csf
  sz = shp[:3].copy()
  sz.append(num_props)
  prob = np.zeros(tuple(sz))
  idx_dict = {0:0, 1:1, 2:5, 3:6, 4:7, 5:8}
  # for each label, compute all the atlases which predict that label and divide by total number of atlases
  # this gives a probability of each label
  for idx in range(num_props):
    prob[:,:,:,idx] = np.count_nonzero(img == idx_dict[idx], axis=3)
  # resegment using the above probabilites by computing the most likely label
  #prob_seg = np.sum(prob, axis=3)
  prob_seg = np.argmax(prob, axis=3)
  prob_seg_mod = prob_seg.copy()
  for idx in range(num_props):
    prob_seg_mod[prob_seg == idx] = idx_dict[idx]

  return prob_seg_mod

#--------------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='script to perform some post-inversion analysis',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-p', '--patient_dir', type = str, help = 'path to masseffect patient reconstructions', required = True) 
  r_args.add_argument('-pnm', '--patient_dir_nm', type = str, help = 'path to masseffect patient reconstructions using the no-mass-effect model') 
  r_args.add_argument('-i', '--input_dir', type = str, help = 'path to all patients', required = True) 
  r_args.add_argument('-d', '--data_dir', type = str, help = 'path to masseffect patient input data (if different from patient_dir)') 
  r_args.add_argument('-x', '--results_dir', type = str, help = 'results path', required = True) 
  r_args.add_argument('-s', '--survival_file', type = str, help = 'path to survival csv', required = True) 
  args = parser.parse_args();

  me_res_dir = args.patient_dir
  in_dir = args.input_dir
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

  im_sz = 160
  recon = np.zeros((im_sz,im_sz,im_sz,16))
  if 'prob-err-rsc' not in stats:
    stats['prob-err-rsc'] = 1
    stats['prob-dice'] = 0
    stats['prob-vt-dice'] = 0
    for pat_idx, pat in enumerate(patient_list):
      print("{}: patient = {}".format(pat_idx, pat))
      recon[:] = 0
      inv_dir = os.path.join(*[me_res_dir, pat, "tu/160"])
      inv_data_dir = os.path.join(*[data_dir, pat, "tu/160"])
      pat_seg = read_netcdf(os.path.join(inv_data_dir, pat + "_seg_ants_aff2jakob_160.nc"))
      tc = np.logical_or(pat_seg == 1, pat_seg == 4)
      vt_pat = (pat_seg == 7) 
      pat_compute = True
      c = np.zeros((im_sz, im_sz, im_sz))
      at_list = []
      for at in os.listdir(inv_dir):
        config_file = os.path.join(inv_dir, at) + "/solver_config.txt"
        if not os.path.exists(config_file):
          continue
        at_list.append(at)

      num_atlases = len(at_list)
      for idx,at in enumerate(at_list):
        at_dir = os.path.join(inv_dir, at)
        c += read_netcdf(os.path.join(at_dir, "c_rec_final.nc"))
        recon[:,:,:,idx] = read_netcdf(os.path.join(at_dir, "seg_rec_final.nc"))
        if pat_compute:
          c_data = read_netcdf(os.path.join(at_dir, "data.nc"))
          obs    = read_netcdf(os.path.join(at_dir, "obs.nc"))
          data_norm = la.norm(c_data[:])
          pat_compute = False

      c /= num_atlases ## avg prob dist of c
      prob_recon = create_prob_img(recon[:,:,:,0:num_atlases])
      tu = (prob_recon == 1)
      vt = (prob_recon == 7)
      c = obs * c
      tu = obs * tu
      vt = obs * vt
      c_rsc = c / np.max(c[:])
      stats.loc[stats['PATIENT'] == pat, 'prob-err-rsc'] = la.norm(c_rsc[:] - c_data[:])/data_norm
      stats.loc[stats['PATIENT'] == pat, 'prob-dice'] = compute_dice(tu, tc) 
      stats.loc[stats['PATIENT'] == pat, 'prob-vt-dice'] = compute_dice(vt, vt_pat) 
  
  if 'mu-ed-ratio' not in stats:
    if args.patient_dir_nm:
      if me_res_dir == data_dir:
        print("WARNING: needs to be run for mass effect model results")
      nm_dir = args.patient_dir_nm
      stats['mu-ed-ratio'] = 0
      stats['std-ed-ratio'] = 0
      for pat_idx, pat in enumerate(patient_list):
        print("{}: patient = {}".format(pat_idx, pat))
        inv_dir = os.path.join(*[me_res_dir, pat, "tu/160"])
        inv_data_dir = os.path.join(*[data_dir, pat, "tu/160"])
        pat_seg = read_netcdf(os.path.join(inv_data_dir, pat + "_seg_ants_aff2jakob_160.nc"))
        ed = (pat_seg == 2)
        nm_inv_dir = os.path.join(*[nm_dir, pat, "tu/160"])
        ed_ratio = []
        for at in os.listdir(inv_dir):
          config_file = os.path.join(inv_dir, at) + "/solver_config.txt"
          if not os.path.exists(config_file):
            continue
          at_dir = os.path.join(inv_dir, at)
          nm_at_dir = os.path.join(nm_inv_dir, at)
          seg = read_netcdf(os.path.join(at_dir, "seg_rec_final.nc"))
          c = (seg == 1) * ed
          seg = read_netcdf(os.path.join(nm_at_dir, "seg_rec_final.nc"))
          nm_c = (seg == 1) * ed
          ed_ratio.append(la.norm(c[:])/la.norm(nm_c[:]))
        a_ed  = np.asarray(ed_ratio)
        stats.loc[stats['PATIENT'] == pat, 'mu-ed-ratio'] = np.mean(a_ed)
        stats.loc[stats['PATIENT'] == pat, 'std-ed-ratio'] = np.std(a_ed)
    else:
      print("no-mass-effect model results directory not set; exiting...")

  if 'tc-vol' not in stats:
    stats['tc-vol'] = 0
    stats['ed-vol'] = 0
    stats['nec-vol'] = 0
    stats['en-vol'] = 0
    for pat_idx, pat in enumerate(patient_list):
      print("{}: patient = {}".format(pat_idx, pat))
      pat_data_dir = os.path.join(*[in_dir, pat, "aff2jakob"])
      pat_seg = nib.load(os.path.join(pat_data_dir, pat + "_seg_ants_aff2jakob.nii.gz")).get_fdata()
      
      tc = np.logical_or(pat_seg == 1, pat_seg == 4)
      stats.loc[stats['PATIENT'] == pat, 'tc-vol'] = compute_volume(tc)
      stats.loc[stats['PATIENT'] == pat, 'ed-vol'] = compute_volume(pat_seg == 2)
      stats.loc[stats['PATIENT'] == pat, 'nec-vol'] = compute_volume(pat_seg == 1)
      stats.loc[stats['PATIENT'] == pat, 'en-vol'] = compute_volume(pat_seg == 4)

  if 'prob-err-vt-vol' not in stats:
    print("vt vol err\n")
    stats['prob-err-vt-vol'] = 1
    for pat_idx, pat in enumerate(patient_list):
      print("{}: patient = {}".format(pat_idx, pat))
      recon[:] = 0
      inv_dir = os.path.join(*[me_res_dir, pat, "tu/160"])
      inv_data_dir = os.path.join(*[data_dir, pat, "tu/160"])
      pat_seg = read_netcdf(os.path.join(inv_data_dir, pat + "_seg_ants_aff2jakob_160.nc"))
      vt_pat = (pat_seg == 7) 
      pat_compute = True
      at_list = []
      for at in os.listdir(inv_dir):
        config_file = os.path.join(inv_dir, at) + "/solver_config.txt"
        if not os.path.exists(config_file):
          continue
        at_list.append(at)

      num_atlases = len(at_list)
      for idx,at in enumerate(at_list):
        at_dir = os.path.join(inv_dir, at)
        recon[:,:,:,idx] = read_netcdf(os.path.join(at_dir, "seg_rec_final.nc"))
        if pat_compute:
          obs    = read_netcdf(os.path.join(at_dir, "obs.nc"))
          pat_compute = False

      prob_recon = create_prob_img(recon[:,:,:,0:num_atlases])
      vt = (prob_recon == 7)
      vt = obs * vt
      pat_vt_vol = compute_volume(vt_pat)
      stats.loc[stats['PATIENT'] == pat, 'prob-err-vt-vol'] = np.abs(pat_vt_vol - compute_volume(vt))/pat_vt_vol 

stats.to_csv(args.results_dir + "/penn_stats.csv", index=False)
