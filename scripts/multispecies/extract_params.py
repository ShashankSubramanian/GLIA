import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
from numpy import linalg as LA
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset
from numpy import linalg as la
import math
import re
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../utils/')
from file_io import writeNII, createNetCDF
from image_tools import resizeImage, resizeNIIImage, compute_volume
sys.path.append('../')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='extract/compute tumor stats')
  parser.add_argument ('-results_path', type=str, help = 'path to tumor inversion results');
  args = parser.parse_args();
  list_inv_params = ['k', 'rho', 'ox_hypoxia','death_rate', 'alpha_0', 'ox_consumption', 'ox_source', 'beta_0', 'ox_inv', 'invasive_thres']

  lb = [0.01, 1.0, 0.1, 0.1, 0.01, 1.0, 0.5, 0.01, 0.2, -5]
  ub = [0.8, 30.0, 0.7, 20.0, 4.0, 20.0, 20.0, 2.0, 1.0, -0.5]

  
  log_file=os.path.join(args.results_path, 'log.txt')
  with open(log_file, 'r') as f:
    lines = f.readlines()
  
  for (i, l) in enumerate(lines):
    if 'xbest=array' in l:
      #tot = lines[i].split('([')[-1].strip('\n')+lines[i+1].strip('\n')+lines[i+2].split('])')[0]
      tot = lines[i].split('([')[-1].strip('\n')
      if 'fbest' in lines[i+1]:
        tot += lines[i+1].split('])')[0]
      elif 'fbest' in lines[i+2]:
        tot += lines[i+1].strip('\n')+lines[i+2].split('])')[0]      
      break
  
  tot = tot.split()
  if ',' in tot:
    tot.remove(',')
  print(tot)
  inv_params = np.zeros(len(tot))
  for (i,p) in enumerate(tot):
    #print(p.strip(','))
    val = float(p.strip(','))
    inv_params[i] = lb[i] + (ub[i] - lb[i]) * val
    if list_inv_params[i] == 'invasive_thres':
      inv_params[i] = 10**inv_params[i]
  print(inv_params)
  
  recon_file = os.path.join(args.results_path, 'recon_info.dat')
  with open(recon_file, 'w') as f:
    var = ''
    for i in list_inv_params:
      var += i+' '
    var += '\n'
    for i in inv_params:
      var += str(i)+' '
    f.write(var)
    
      
