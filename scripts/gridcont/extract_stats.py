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
from image_tools import resizeImage, resizeNIIImage
sys.path.append('../')


def compute_volume(mat):
  sz      = mat.shape[0]
  h       = (2.0 * math.pi) /  sz
  measure = h * h * h
  vol = np.sum(mat.flatten())
  vol *= measure
  return vol

###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='extract/compute tumor stats')
  parser.add_argument ('-patient_dir', type = str, help = 'path to patients (brats format)') 
  parser.add_argument ('-results_dir', type=str, help = 'path to tumor inversion results');

  args = parser.parse_args(); 
  results_path = args.results_dir
  patient_path = args.patient_dir
  match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee][+-]?\ *-?\ *[0-9]+)?')
  base_dir = os.path.dirname(os.path.realpath(__file__)) + "/../../" 
 
  if os.path.exists(patient_path + "/pat_stats.csv"):
    with open(patient_path + "/pat_stats.csv", "r") as f:
      all_pats = f.readlines()
    patient_list = []
    for l in all_pats:
      patient_list.append(l.split(",")[0])
    if os.path.exists(patient_path + "/failed.txt"): ### if some patients have failed in some preproc routinh; ignore them
      with open(patient_path + "/failed.txt", "r") as f:
        lines = f.readlines()
      for l in lines:
        failed_pat = l.strip("\n")
        print("ignoring failed patient {}".format(failed_pat))
        if failed_pat in patient_list:
          patient_list.remove(failed_pat)
  else:
    print("No stat file; listing directory instead")
    for patient in os.listdir(patient_path):
      suffix = ""
      if not os.path.exists(os.path.join(os.path.join(patient_path, patient), patient + "_t1" + suffix + ".nii.gz")):
        continue
      patient_list.append(patient) 

  other_remove = [] #["Brats18_CBICA_ABO_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1", "Brats18_CBICA_AAP_1"]
  for others in other_remove:
    patient_list.remove(others)

  block_job = False
  suff = ""   ## custom string to append to filenames
  if block_job:
    it = 0
    num_pats = 40
    patient_list = patient_list[it*num_pats:it*num_pats + num_pats]
  
  global_stats = ""
  global_f     = open(results_path + "/tumor_inversion_stats.csv", "w+")
  for pat_name in patient_list:
    inv_path = os.path.join(results_path, pat_name) + "/inversion/nx256/obs-1.0/"
    res_path = os.path.join(results_path, pat_name) + "/inversion/nx256/stats/"
    if not os.path.exists(res_path):
      os.makedirs(res_path)
    print("[] extracting stats for patient {}".format(pat_name))
    row = ""
    row_csv = ""

    ### parameters
    rho_list   = []
    kappa_list = []
    err_list   = []
    time_list  = []

    ### scrub the log file
    statfile = open(res_path + "stats" + suff + ".txt", 'w+')
    statfile_csv = open(res_path + "stats" + suff + ".csv", 'w+')
    recon_file = inv_path + "/reconstruction_info.dat"

    err = False 
    if not os.path.exists(recon_file):
      err = True
      print("recon file not found")
    else:
      with open(recon_file, 'r') as f:
        lines = f.readlines()
        if len(lines) > 1:
          l = lines[1].split(" ")
          rho       = float(l[0])
          kappa     = float(l[1])
          rel_error = float(l[2])
        else:
          err = True
          print('recon file is corrupted. skipping...\n')
     
    if err:
      continue
    ### extract timings from logfile
    log_file = inv_path + "/solver_log.txt"
    if not os.path.exists(log_file):
      print("logfile does not exist!. breaking..")
      continue

    with open(log_file, 'r') as f:
      lines = f.readlines()
      for l in lines:
        if l.find("Global runtime") is not -1:
          l_s = re.findall("\d*\.?\d+", l)
          t = float(l_s[1])
    
    rho_list.append(rho)
    kappa_list.append(kappa)
    err_list.append(rel_error)
    time_list.append(t)
    row += pat_name + " \t& "
    row_csv += pat_name + ","
  
    row += "\\num{" + "{:e}".format(rho) + "} \t& "
    row += "\\num{" + "{:e}".format(kappa) + "} \t& "
    row += "\\num{" + "{:e}".format(rel_error) + "} \t& "
    row += "\\num{" + "{:e}".format(t) + "} \\\\ \n"

    row_csv += str(rho) + ","
    row_csv += str(kappa) + ","
    row_csv += str(rel_error) + ","
    row_csv += str(t) + "\n"

    statfile.write(row)
    statfile.close()
    statfile_csv.write(row_csv)
    statfile_csv.close()
    global_stats += row_csv 
  global_f.write(global_stats)
