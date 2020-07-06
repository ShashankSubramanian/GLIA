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


def computeVolume(mat):
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
  parser.add_argument ('-n', type=int, help = 'size');
  parser.add_argument ('-patient_dir', type = str, help = 'path to patients (brats format)') 
  parser.add_argument ('-results_path', type=str, help = 'path to tumor inversion results');
  parser.add_argument ('-atlas_stat_path', type=str, help = 'path to atlas stats');
  parser.add_argument ('-patient_stat_path', type=str, help = 'path to patient stats');

  args = parser.parse_args();
  n = args.n
  at_stat_path = args.atlas_stat_path
  pat_stat_path = args.patient_stat_path
  results_path = args.results_path + "/"
  match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee][+-]?\ *-?\ *[0-9]+)?')
  c_avg = np.zeros((n,n,n))
  u_avg = np.zeros((n,n,n))
  suff = ""
  base_dir  = os.getcwd() + "/../../"
  use_mat_prop = True
  
  
  
  with open(args.patient_dir + "/brats-pat-stats.csv", "r") as f:
    brats_pats = f.readlines()
  patient_list = []
  for l in brats_pats:
    patient_list.append(l.split(",")[0])
  if os.path.exists(args.patient_dir + "/failed.txt"): ### some patients have failed gridcont; ignore them
    with open(args.patient_dir + "/failed.txt", "r") as f:
      lines = f.readlines()
    for l in lines:
      failed_pat = l.strip("\n")
      print("ignoring failed patient {}".format(failed_pat))
      if failed_pat in patient_list:
        patient_list.remove(failed_pat)

  other_remove = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1", "Brats18_CBICA_AAP_1"]
  for others in other_remove:
    patient_list.remove(others)

  block_job = True
  if block_job:
    it = 0
    num_pats = 10
    ### 0:10
    patient_list = patient_list[it*num_pats:it*num_pats + num_pats]
  
  global_stats = ""
  global_f     = open(results_path + "/tumor_inversion_stats.csv", "w+")
  for pat_name in patient_list:
    c_avg = 0 * c_avg
    u_avg = 0 * u_avg

    res_path = results_path + pat_name + "/stat/"
    inv_path = results_path + pat_name + "/tu/" + str(n) + "/"

    print("[] extracting stats for patient {}".format(pat_name))
    row = ""
    row_csv = ""
    failed_atlas = []

    with open(pat_stat_path, 'r') as f:
      lines = f.readlines()
    for l in lines:
      if l.find(pat_name) is not -1:
        pat_vt = float(l.split(",")[1])
    with open(at_stat_path, 'r') as f:
      atlas_full_list = f.readlines()

    atlas_vt_dict = {}
    for l in atlas_full_list:
      at_nm = l.split(",")[0]
      vt = l.split(",")[1]
      atlas_vt_dict[at_nm] = float(vt)

    ### parameters
    gam_list   = []
    rho_list   = []
    kappa_list = []
    disp_list  = []
    err_list   = []
    time_list  = []
    cond_list  = []
    vt_change_list = []
    vt_err_list    = []

### scrub the log file
    statfile = open(res_path + "stats" + suff + ".txt", 'w+')
    statfile_csv = open(res_path + "stats" + suff + ".csv", 'w+')
    listfile = results_path + pat_name + "/atlas-list.txt"
    at_list  = []
    with open(listfile, 'r') as f:
      lines = f.readlines()
    for l in lines:
      at_list.append(l.strip('\n'))

    patient_compute = False
#    num_atlases = 8 if len(at_list) > 8 else len(at_list)
    for atlas in at_list:
      exist = True
      print("reading recon dat file from atlas " + atlas)
      recon_file = inv_path + atlas + "/reconstruction_info.dat"
      hess_file = inv_path + atlas + "/hessian_opt.txt"

      if use_mat_prop:
        vt_rec_file = inv_path + atlas + "/vt_rec_final.nc"
        at_vt_file = inv_path + atlas + "/vt.nc"
        pat_vt_file = inv_path + atlas + "/p_vt.nc"
        at_vt_recon = 0
        if os.path.exists(vt_rec_file):
          vt_nc = Dataset(vt_rec_file, mode='r', format="NETCDF3_CLASSIC")
          vt = np.transpose(vt_nc.variables['data'])
          at_vt_recon = computeVolume(vt)
        else:
          print("vt rec file {} not found. skipping vt data...".format(vt_rec_file))

        if os.path.exists(at_vt_file):
          vt_nc = Dataset(at_vt_file, mode='r', format="NETCDF3_CLASSIC")
          vt = np.transpose(vt_nc.variables['data'])
          at_orig = computeVolume(vt)
        else:
          print("atlas vt file {} not found. skipping vt data...".format(at_vt_file))

        if not patient_compute:
          if os.path.exists(pat_vt_file):
            vt_nc = Dataset(pat_vt_file, mode='r', format="NETCDF3_CLASSIC")
            vt = np.transpose(vt_nc.variables['data'])
            pat_vt = computeVolume(vt)
            patient_compute = True
          else:
            print("pat vt file {} not found. skipping vt data...".format(pat_vt_file))
      else:
        seg_rec_file = inv_path + atlas + "/seg_rec_final.nc"
        if os.path.exists(seg_rec_file):
          seg_nc = Dataset(seg_rec_file, mode='r', format="NETCDF3_CLASSIC")
          seg = np.transpose(seg_nc.variables['data'])
          vt = (seg == 7)
          at_vt_recon = computeVolume(vt)
        else:
          print("seg rec file {} not found. skipping vt data...".format(seg_rec_file))
        at_orig = atlas_vt_dict[atlas]


      if not os.path.exists(hess_file):
        print("hessian file for atlas {} not found".format(atlas))
        cond = inf
        exist = False
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
              t = float(l_s[1])
        
        raw      = []
        if exist:
          with open(hess_file, "r") as f1:
            for line in f1:
              raw.append(line.split())
          hess = np.array(raw)
          hess = hess.astype(float)
          cond = LA.cond(hess)

        ### compute vt volume change
        ###print("atlas stats: {},{}".format(at_orig, at_vt_recon))
        vt_change = np.abs((at_vt_recon - at_orig)/at_orig)
        vt_err    = np.abs((at_vt_recon - pat_vt)/pat_vt)
        

        if max_disp > 2 or rel_error > 1:
          failed_atlas.append(atlas)
          print('failed atlas {}'.format(atlas))
          row += atlas + "(F) \t& "
          row_csv += atlas + "(F),"
        elif cond > 10000:
          failed_atlas.append(atlas)
          print('bad conditioned atlas {}'.format(atlas))
          row += atlas + "(B) \t& "
          row_csv += atlas + "(B),"
        else:
          gam_list.append(gamma)
          rho_list.append(rho)
          kappa_list.append(kappa)
          disp_list.append(max_disp)
          err_list.append(rel_error)
          time_list.append(t)
          cond_list.append(cond)
          vt_change_list.append(vt_change)
          vt_err_list.append(vt_err)
          row += atlas + " \t& "
          row_csv += atlas + ","
      
        row += "\\num{" + "{:e}".format(gamma) + "} \t& "
        row += "\\num{" + "{:e}".format(rho) + "} \t& "
        row += "\\num{" + "{:e}".format(kappa) + "} \t& "
        row += "\\num{" + "{:e}".format(max_disp) + "} \t& "
        row += "\\num{" + "{:e}".format(rel_error) + "} \t& "
        row += "\\num{" + "{:e}".format(cond) + "} \t& "
        row += "\\num{" + "{:e}".format(vt_change) + "} \t& "
        row += "\\num{" + "{:e}".format(vt_err) + "} \t& "
        row += "\\num{" + "{:e}".format(t) + "} \\\\ \n"

        row_csv += str(gamma) + ","
        row_csv += str(rho) + ","
        row_csv += str(kappa) + ","
        row_csv += str(max_disp) + ","
        row_csv += str(rel_error) + ","
        row_csv += str(cond) + ","
        row_csv += str(vt_change) + ","
        row_csv += str(vt_err) + ","
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
    cond_arr   = np.asarray(cond_list)
    vt_change_arr = np.asarray(vt_change_list)
    vt_err_arr    = np.asarray(vt_err_list)
    
    if len(gam_arr) > 0:
      row += "stats" + " \t& "
      row += "\\num{" + "{:e}".format(np.mean(gam_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(gam_arr)) + "} \t& "
      row += "\\num{" + "{:e}".format(np.mean(rho_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(rho_arr)) + "} \t& "
      row += "\\num{" + "{:e}".format(np.mean(kappa_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(kappa_arr)) + "} \t& "
      row += "\\num{" + "{:e}".format(np.mean(disp_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(disp_arr)) + "} \t& "
      row += "\\num{" + "{:e}".format(np.mean(err_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(err_arr)) + "} \t& "
      row += "\\num{" + "{:e}".format(np.mean(cond_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(cond_arr)) + "} \t& "
      row += "\\num{" + "{:e}".format(np.mean(vt_change_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(vt_change_arr)) + "} \t& "
      row += "\\num{" + "{:e}".format(np.mean(vt_err_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(vt_err_arr)) + "} \t& "
      row += "\\num{" + "{:e}".format(np.mean(time_arr)) + "} $\pm$ \\num{" + "{:e}".format(np.std(time_arr)) + "}\n"
    
      row_csv += "\n\n ######################################################## \n\n "
      row_means = ""
      row_means += pat_name + ","
      row_means += str(np.mean(gam_arr)) + "," + str(np.std(gam_arr)) + ","
      row_means += str(np.mean(rho_arr)) + "," + str(np.std(rho_arr)) + ","
      row_means += str(np.mean(kappa_arr)) + "," + str(np.std(kappa_arr)) + ","
      row_means += str(np.mean(disp_arr)) + "," + str(np.std(disp_arr)) + ","
      row_means += str(np.mean(err_arr)) + "," + str(np.std(err_arr)) + ","
      row_means += str(np.mean(cond_arr)) + "," + str(np.std(cond_arr)) + ","
      row_means += str(np.mean(vt_change_arr)) + "," + str(np.std(vt_change_arr)) + ","
      row_means += str(np.mean(vt_err_arr)) + "," + str(np.std(vt_err_arr)) + ","
      row_means += str(np.mean(time_arr)) + "," + str(np.std(time_arr)) + "\n"
      row_csv += row_means

    statfile.write(row)
    statfile.close()
    statfile_csv.write(row_csv)
    statfile_csv.close()
    global_stats += row_means
  global_f.write(global_stats)
