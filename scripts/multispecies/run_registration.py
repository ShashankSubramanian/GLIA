
import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import nibabel.processing
import scipy as sc
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset
from numpy import linalg as la
import math
import time

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../utils/')
from create_brain_stats import print_atlas_stats
from file_io import writeNII, createNetCDF
from image_tools import resizeImage, resizeNIIImage
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../masseffect/')
import params as par
import random
import shutil
from register import create_patient_labels, create_atlas_labels, register, transport

from run_masseffect import create_sbatch_header

def write_reg(reg, pat, data_dir, atlas_dir, at_list, claire_dir, bash_file, idx, ngpu, r, patient_seg, atlas_seg):
  """ writes registration cmds """
  n_local = 1 
  ### create template(patient) labels
  if not os.path.exists(reg + pat + "_vt.nii.gz"):
    create_patient_labels(os.path.join(data_dir, patient_seg), reg, pat)
  ### create reference(atlas) labels
  with open(bash_file, "a") as f:
    #f.write("\n\nsource /work2/07544/ghafouri/frontera/gits/claire/deps/env_source.sh\n\n")
    f.write("\n\nsource /work/07544/ghafouri/longhorn/gits/claire_glia.sh\n\n")
    f.write("\n\nsource /home/07544/ghafouri/gits/claire/deps/env_source.sh\n\n")
  for i in range(0, n_local):
    at = at_list[ngpu*idx+i]
    if not os.path.exists(reg + at + "/" + at + "_vt.nii.gz"):
      create_atlas_labels(os.path.join(atlas_dir, atlas_seg), reg + at, at)
    ### register
    bash_file = register(claire_dir, reg+at, at, reg, pat, bash_file, i, r = r)

  with open(bash_file, "a") as f:
    f.write("\n\nwait\n\n")

  for i in range(0, n_local):
    ### transport
    at = at_list[ngpu*idx+i]
    bash_file = transport(claire_dir, reg+at, data_dir + "/" + pat + "_c0Recon_aff2jakob.nii.gz", pat + "_c0Recon", bash_file, i, r = r)

  with open(bash_file, "a") as f:
    f.write("\n\nwait\n\n")

  for i in range(0, n_local):
    ### transport
    at = at_list[ngpu*idx+i]
    bash_file = transport(claire_dir, reg+at, data_dir + "/" + "seg_t0.nii.gz", pat + "_labels", bash_file, i, r = r)

  with open(bash_file, "a") as f:
    f.write("\n\nwait\n\n")

  return bash_file



class Args:
  def __init__(self):
    self.patient_dir = ""
    self.atlas_dir = ""
    self.results_dir = ""
    self.code_dir = ""
    self.n_resample = 160
    self.reg = 1
    self.claire_dir = ""
    self.compute_sys = "frontera"
    self.submit = False
    self.syn = False
    self.num_pat_per_job = 1
    self.num_gpus = 1
    self.job_dir = ""
    self.patient_seg = ""
    self.atlas_seg = ""
    self.patient_name = ""
    self.til_dir = "" 
    
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Mass effect inversion',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-p', '--patient_dir', type = str, help = 'path to patients (brats format)', required = True)
  r_args.add_argument('-a', '--atlas_dir', type = str, help = 'path to atlases', required = True)
  r_args.add_argument('-at_name', '--atlas_name', type = str, help = 'path to atlases', required = True)
  r_args.add_argument('-x', '--results_dir', type = str, help = 'path to results', required = True)
  r_args.add_argument('-c', '--code_dir', type = str, help = 'path to tumor solver code', required = True)
  r_args.add_argument('-til', '--til_dir', type = str, help = 'path to til results')
  r_args.add_argument('-n', '--n_resample', type = int, help = 'size for inversion', default = 160)
  r_args.add_argument('-rc', '--claire_dir', type = str, help = 'path to claire bin', default = "")
  r_args.add_argument('-csys', '--compute_sys', type = str, help = 'compute system', default = 'frontera')
  r_args.add_argument('-pat_seg', '--patient_seg', type = str, help = 'compute system', default = 'frontera')
  r_args.add_argument('-at_seg', '--atlas_seg', type = str, help = 'compute system', default = 'frontera')
  r_args.add_argument('-pat', '--patient_name', type = str, help = 'compute system', default = 'frontera')
  r_args.add_argument('-submit', action = 'store_true', help = 'submit jobs (after they have been created)')
  r_args.add_argument('-syn', action = 'store_true', help = 'the data is synthetic, so true concentrations are known')
  args = parser.parse_args();
  
  r = {}
  r['compute_sys'] = 'longhorn' 
  r['log_dir'] = args.results_dir
  patient_dir = args.patient_dir
  atlas_dir = args.atlas_dir
  results_dir = args.results_dir
  code_dir = args.code_dir
  n_res = args.n_resample
  reg = os.path.join(results_dir, 'reg/')
  if not os.path.exists(reg):
    os.mkdir(reg) 
  claire_dir = args.claire_dir
  csys = args.compute_sys
  at_seg = args.atlas_seg
  pat_seg = args.patient_seg
  pat = args.patient_name
  til_dir = args.til_dir
  at_list = [args.atlas_name]


  print(results_dir)
  til = os.path.join(til_dir, 'inversion/nx256/obs-1.0/c0_rec_256256256.nii.gz')
  dst = os.path.join(patient_dir, pat+"_c0Recon_aff2jakob.nii.gz")
  shutil.copy(til, dst)
  
  bash_file = create_sbatch_header(results_dir, 0, csys, r)
  bash_file = write_reg(reg, pat, patient_dir, atlas_dir, at_list, claire_dir, bash_file, 0, 1, r, pat_seg, at_seg)
  
   
  
   
  
  

  
  
  
  




