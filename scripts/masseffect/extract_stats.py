import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset
from numpy import linalg as la
import re
sys.path.append('../utils/')
from file_io import writeNII, createNetCDF
from image_tools import resizeImage, resizeNIIImage
sys.path.append('../')


###
### ------------------------------------------------------------------------ ###
if __name__=='__main__'
  parser = argparse.ArgumentParser(description='extract/compute tumor stats')
  parser.add_argument ('-n', type=int, help = 'size');
  parser.add_argument ('-res_path', type=str, help = 'path to output results');
  parser.add_argument ('-inv_path', type=str, help = 'path to inversion results');

  args = parser.parse_args();
  res_path = args.res_path
  n = args.n
  inv_path = args.inv_path
  match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee][+-]?\ *-?\ *[0-9]+)?')
  c_avg = np.zeros((n,n,n))
  u_avg = np.zeros((n,n,n))
  suff = ""
  base_dir  = os.getcwd() + "/../../"
  
  c_avg = 0 * c_avg
  u_avg = 0 * u_avg

  print("[] extracting stats")
  row = ""
  row_csv = ""
  failed_atlas = []

  ### parameters
  gam_list   = []
  rho_list   = []
  kappa_list = []
  disp_list  = []
  err_list   = []
  time_list  = []

### scrub the log file
  statfile = open(res_path + "stats" + suff + ".txt", 'w+')
  statfile_csv = open(res_path + "stats" + suff + ".csv", 'w+')
  listfile = res_path + "/atlas-list.txt"
  at_list  = []
  with open(listfile, 'r') as f:
    lines = f.readlines()
  for l in lines:
    at_list.append(l.strip('\n'))

  for atlas in at_list:
    print("reading recon dat file from atlas " + atlas)
    recon_file = inv_path + atlas + "/reconstruction_info.dat"
    if not os.path.exists(recon_file):
      print("recon file for atlas {} not found".format(atlas))
      continue
    err = False 
    with open(recon_file, 'r') as f:
      lines = f.readlines()
      if len(lines) > 1:
        l = lines[1].split(" ")
        rho       = float(l[0])
        kappa     = float(l[1])
        gamma     = float(l[2])
        max_disp  = float(l[3])
        norm_disp = float(l[4])
        rel_error = float(l[5])
      else:
        err = True
        print('recon file is corrupted. skipping...\n')
        continue
       
      ### extract timings from logfile
      log_file = inv_path + atlas + "/log"
      if not os.path.exists(log_file):
        print("logfile does not exist!. breaking..")
        continue

      with open(log_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
          if l.find("Global runtime") is not -1:
            l_s = re.findall("\d*\.?\d+", l)
            t = float(l_s[3])

      if max_disp > 2 or rel_error > 1:
        failed_atlas.append(atlas)
        print('failed atlas {}'.format(atlas))
        row += atlas + "(F) \t& "
        row_csv += atlas + "(F),"
      else:
        gam_list.append(gamma)
        rho_list.append(rho)
        kappa_list.append(kappa)
        disp_list.append(max_disp)
        err_list.append(rel_error)
        time_list.append(t)
        row += atlas + " \t& "
        row_csv += atlas + ","
    
      row += "\\num{" + "{:e}".format(gamma) + "} \t& "
      row += "\\num{" + "{:e}".format(rho) + "} \t& "
      row += "\\num{" + "{:e}".format(kappa) + "} \t& "
      row += "\\num{" + "{:e}".format(max_disp) + "} \t& "
      row += "\\num{" + "{:e}".format(rel_error) + "} \t& "
      row += "\\num{" + "{:e}".format(t) + "} \\\\ \n"

      row_csv += str(gamma) + ","
      row_csv += str(rho) + ","
      row_csv += str(kappa) + ","
      row_csv += str(max_disp) + ","
      row_csv += str(rel_error) + ","
      row_csv += str(t) + "\n"

#  for idx in range(1,num_cases+1):
#    if n == 256:
#      atlas = "atlas-" + str(idx) + suff
#    else:
#      atlas = "atlas-" + str(idx) + "_" + str(n)
#    if idx in failed_atlas:
#      print('skipping failed atlas {}'.format(idx))
#      continue
#
#    inv_results = base_dir + "results/inv-" + pat + "/" + atlas + "/"
#    file = Dataset(inv_results + "c_rec_final.nc", mode='r', format="NETCDF3_CLASSIC")
#    c_avg += np.transpose(file.variables['data'])
#    file = Dataset(inv_results + "displacement_rec_final.nc", mode='r', format="NETCDF3_CLASSIC")
#    u_avg += np.transpose(file.variables['data'])
#
#  c_avg /= num_cases
#  u_avg /= num_cases
#  nii = nib.load(base_dir + "brain_data/real_data/" + pat + "/data/" + pat + "_seg_tu_aff2jakob.nii.gz")
#  if n == 256:
#    writeNII(c_avg, res_path + "c_avg_" + str(n) + suff + ".nii.gz", ref_image = nii)
#    writeNII(u_avg, res_path + "u_avg_" + str(n) + suff + ".nii.gz", ref_image = nii)
#  else:
#    writeNII(c_avg, res_path + "c_avg_" + str(n) + ".nii.gz")
#    writeNII(u_avg, res_path + "u_avg_" + str(n) + ".nii.gz")

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
 
  row_csv += "\n\n ######################################################## \n\n "
  row_csv += str(np.mean(gam_arr)) + "," + str(np.std(gam_arr)) + ","
  row_csv += str(np.mean(rho_arr)) + "," + str(np.std(rho_arr)) + ","
  row_csv += str(np.mean(kappa_arr)) + "," + str(np.std(kappa_arr)) + ","
  row_csv += str(np.mean(disp_arr)) + "," + str(np.std(disp_arr)) + ","
  row_csv += str(np.mean(err_arr)) + "," + str(np.std(err_arr)) + ","
  row_csv += str(np.mean(time_arr)) + "," + str(np.std(time_arr)) + ",\n"

  statfile.write(row)
  statfile.close()
  statfile_csv.write(row_csv)
  statfile_csv.close()
