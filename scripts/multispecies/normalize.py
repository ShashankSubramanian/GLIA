
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
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../helpers/')
from create_brain_stats import print_atlas_stats
from file_io import writeNII, createNetCDF, readNetCDF
from image_tools import resizeImage, resizeNIIImage
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../masseffect/')
import params as par
import random
import shutil





for i in range(1,5):
  syn = 'case'+str(i)

  c0_dir = os.path.join('/scratch1/07544/ghafouri/results/syn_results/me_inv', syn, 'reg', syn, syn+'_c0Recon_transported.nii.gz')
  
  cmd = 'python ../helpers/convert_to_netcdf.py -i '+c0_dir
  os.system(cmd)
  print(cmd) 
  dat = readNetCDF(c0_dir.replace('.nii.gz', '.nc'))
  dat /= np.amax(dat)
  dat[dat < 0.0] = 0.0
  fname = c0_dir.replace('.nii.gz', '_normalized.nc')
  print(np.amax(dat), fname)
  createNetCDF(fname, dat.shape, dat)
  
