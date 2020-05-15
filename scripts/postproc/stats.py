import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset
from numpy import linalg as la
import re
from postproc_utils import writeNII, createNetCDFFile


row = str()
match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee][+-]?\ *-?\ *[0-9]+)?')
isz = 256
c_avg = np.zeros((isz,isz,isz))
u_avg = np.zeros((isz,isz,isz))
pat_names = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AAP_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1"]
#pat_names = ["Brats18_CBICA_AAP_1", "Brats18_CBICA_ALU_1"]
#pat_names = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AMH_1"]
suff = "-nav-ratio-0"
base_dir  = os.getcwd() + "/../../"
for pat in pat_names:
  c_avg = 0 * c_avg
  u_avg = 0 * u_avg
  num_cases = 8

  print("postop for pat {}".format(pat))
  row = ""
  r_path = base_dir + "results/stat-" + pat + "/"
  if not os.path.exists(r_path):
    os.makedirs(r_path)
  failed_atlas = []

  ### parameters
  gam_list   = []
  rho_list   = []
  kappa_list = []
  disp_list  = []
  err_list   = []
  time_list  = []

### scrub the log file
  statfile = open(r_path + "stats_" + str(isz) + suff + ".txt", 'w+')
  for idx in range(1,num_cases+1):
    if isz == 256:
      atlas = "atlas-" + str(idx) + suff
    else:
      atlas = "atlas-" + str(idx) + "_" + str(isz)
    log_file = base_dir + "results/inv-" + pat + "/" + atlas + "/log"
    start_idx = 0
    with open(log_file) as f:
      lines = f.readlines()
      for line in lines:
        if "estimated" in line:
          break;
        start_idx += 1

      line = lines[start_idx+1]
      l = re.findall("\d*\.?\d+", line)
      gamma = float(l[1])
      line = lines[start_idx+3]
      l = re.findall("\d*\.?\d+", line)
      rho = float(l[2])
      line = lines[start_idx+5]
      l = re.findall("\d*\.?\d+", line)
      kappa = float(l[2])

      ## displacement
      line = lines[start_idx+15]
      l = re.findall(match_number, line)
      max_disp = float(l[2])

      ### relative error
      line = lines[start_idx+20]
      l = re.findall("\d*\.?\d+", line)
      rel_error = float(l[3])

      ### time
      line = lines[start_idx+24]
      l = re.findall("\d*\.?\d+", line)
      t = float(l[1])

      if max_disp > 2 or rel_error > 1:
        failed_atlas.append(idx)
        row += atlas + "(F) \t& "
      else:
        gam_list.append(gamma)
        rho_list.append(rho)
        kappa_list.append(kappa)
        disp_list.append(max_disp)
        err_list.append(rel_error)
        time_list.append(t)
        row += atlas + " \t& "
    
      row += "\\num{" + "{:e}".format(gamma) + "} \t& "
      row += "\\num{" + "{:e}".format(rho) + "} \t& "
      row += "\\num{" + "{:e}".format(kappa) + "} \t& "
      row += "\\num{" + "{:e}".format(max_disp) + "} \t& "
      row += "\\num{" + "{:e}".format(rel_error) + "} \t& "
      row += "\\num{" + "{:e}".format(t) + "} \\\\ \n"

  for idx in range(1,num_cases+1):
    if isz == 256:
      atlas = "atlas-" + str(idx) + suff
    else:
      atlas = "atlas-" + str(idx) + "_" + str(isz)
    if idx in failed_atlas:
      print('skipping failed atlas {}'.format(idx))
      continue

    inv_results = base_dir + "results/inv-" + pat + "/" + atlas + "/"
    file = Dataset(inv_results + "c_rec_final.nc", mode='r', format="NETCDF3_CLASSIC")
    c_avg += np.transpose(file.variables['data'])
    file = Dataset(inv_results + "displacement_rec_final.nc", mode='r', format="NETCDF3_CLASSIC")
    u_avg += np.transpose(file.variables['data'])

  c_avg /= num_cases
  u_avg /= num_cases
  nii = nib.load(base_dir + "brain_data/real_data/" + pat + "/data/" + pat + "_seg_tu_aff2jakob.nii.gz")
  if isz == 256:
    writeNII(c_avg, r_path + "c_avg_" + str(isz) + suff + ".nii.gz", ref_image = nii)
    writeNII(u_avg, r_path + "u_avg_" + str(isz) + suff + ".nii.gz", ref_image = nii)
  else:
    writeNII(c_avg, r_path + "c_avg_" + str(isz) + ".nii.gz")
    writeNII(u_avg, r_path + "u_avg_" + str(isz) + ".nii.gz")

  row += "\n\n ######################################################## \n\n "

  gam_arr    = np.asarray(gam_list)
  rho_arr    = np.asarray(rho_list)
  kappa_arr  = np.asarray(kappa_list)
  err_arr    = np.asarray(err_list)
  disp_arr   = np.asarray(disp_list)
  time_arr   = np.asarray(time_list)

  row += "stats" + " \t& "
  row += "\\num{" + "{:e}".format(np.mean(gam_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(gam_arr)) + "} \t& "
  row += "\\num{" + "{:e}".format(np.mean(rho_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(rho_arr)) + "} \t& "
  row += "\\num{" + "{:e}".format(np.mean(kappa_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(kappa_arr)) + "} \t& "
  row += "\\num{" + "{:e}".format(np.mean(disp_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(disp_arr)) + "} \t& "
  row += "\\num{" + "{:e}".format(np.mean(err_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(err_arr)) + "} \t& "
  row += "\\num{" + "{:e}".format(np.mean(time_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(time_arr)) + "}\n"
  
  statfile.write(row)
  statfile.close()
