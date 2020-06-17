import os,argparse, sys
import numpy as np
import nibabel as nib
import nibabel.processing

import math
from netCDF4 import Dataset
import scipy
import re, shutil, tempfile
import random

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


### ------------------------------------------------------------------------ ###
def update_config(path_res, path_config):
  """ Extracts reconstructed rho, kappa, and gamma values from tumor results and
      updates the config file for the next level.
  """
  eho, k, g =  extract_from_tu_info(os.path.join(path_res, "reconstruction_info.dat"))
  with open(os.path.join(path_config, "solver_config"), "r") as f:
    lines = file.readlines()
  with open(os.path.join(path_config, "solver_config"), "w")  as f:
    for line in lines:
      if "init_rho=" in line:
        print("...updating rho IC to {}".format(rho))
        f.write("init_rho="+str(rho))
      if "init_k=" in line:
        print("...updating k IC to {}".format(k))
        f.write("init_k="+str(k))
      if "init_gamma=" in line:
        print("...updating gamma IC to {}".format(gamma))
        f.write("init_gamma="+str(k))
      else:
        f.write(line)
###
### ------------------------------------------------------------------------ ###
def extract_from_tu_info(path):
  err = False
  rho = -1
  kappa = -1
  gamma = -1
  if os.path.exists(path):
    print("reading reconstruction info dat: {}".format(path))
    with open(path, 'r') as f:
      lines = f.readlines()
      if len(lines) > 1:
        l = lines[1].split(" ")
        rho       = float(l[0])
        kappa     = float(l[1])
        gamma     = float(l[2])
      else:
        err = True
  else:
    err = True

  if err:
    print("Error in accessing/reading reconstruction info dat: {}".format(path))
    print("...using random ICs instead")

  return rho, kappa, gamma

###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
  # parse arguments
  parser = argparse.ArgumentParser(description='update tumor configs')
  parser.add_argument ('-config_path', type=str, help = 'path to config/results');
  parser.add_argument ('-res_path', type=str, help = 'path to config/results');

  args = parser.parse_args();
  print("[] extracting rho and k from ", args.res_path);
  update_config(args.res_path, args.config_path);

