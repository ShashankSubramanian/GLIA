
import argparse
import numpy as np
import os,subprocess

import sys
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../../'
sys.path.append(code_dir)
from scripts.utils.file_io import readNetCDF, createNetCDF, writeNII 

 
def compute_dice(res_file, dat_file):
 
  res = readNetCDF(res_file)
  dat = readNetCDF(dat_file)
  
  Op_inv = np.zeros_like(res)
  Op_inv[res == 4] = 1
  
  On_inv = np.zeros_like(res)
  On_inv[res == 1] = 1
   
  Ol_inv = np.zeros_like(res)
  Ol_inv[res == 2] = 1

  p_dat = np.zeros_like(dat)
  p_dat[dat == 4] = 1
  
  n_dat = np.zeros_like(dat)
  n_dat[dat == 1] = 1

  l_dat = np.zeros_like(dat)
  l_dat[dat == 2] = 1
  
  inter_p = Op_inv * p_dat 
  inter_n = On_inv * n_dat
  inter_l = Ol_inv * l_dat
  
  pi_p = 2 * inter_p.sum() / (Op_inv.sum() + p_dat.sum())
  pi_n = 2 * inter_n.sum() / (Op_inv.sum() + n_dat.sum())
  pi_l = 2 * inter_l.sum() / (Ol_inv.sum() + l_dat.sum())
  
  print(inter_p.sum())
  print(p_dat.sum())
  print(Op_inv.sum())

  print("Pi_p : %.4e"%pi_p)
  print("Pi_n : %.4e"%pi_n)
  print("Pi_l : %.4e"%pi_l)



if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='convert netcdf to nifti', formatter_class= argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-i', '--res_file', type = str, help = 'path to the results', required = True)
  r_args.add_argument('-d', '--dat_file', type = str, help = 'path to the results', required = True)
  args = parser.parse_args();

  dat_file = args.dat_file
  res_file = args.res_file
  
  compute_dice(res_file, dat_file)
