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

sys.path.append('../utils/')
from file_io import writeNII, createNetCDF
from image_tools import resizeImage, resizeNIIImage
sys.path.append('../')
import params as par
import random
import shutil

def convert_nifti_to_nc(filename, n, reverse=False):
  infilename = filename.replace(".nc", ".nii.gz")
  dimensions = n * np.ones(3)
  data = nib.load(infilename).get_fdata()
  if not reverse:
    createNetCDF(filename, dimensions, np.transpose(data))
  else:
    createNetCDF(filename, dimensions, np.transpose(data[::-1,::-1,:]))

def resize_data(img_path, img_path_out, sz, order = 3):
  sz       -= 1
  img       = nib.load(img_path)
  img_rsz   = resizeNIIImage(img, sz, interp_order = order)
  nib.save(img_rsz, img_path_out)

def create_tusolver_config(n, pat, pat_dir, atlas_dir, res_dir):
  r = {}
  p = {}
  submit_job = False;
  listfile    = res_dir + "/../../atlas-list.txt"
  at_list_f   = open(listfile, 'r')
  at_list     = at_list_f.readlines()
  case_str    = ""  ### for different cases to go in different dir with this suffix
  n_str       = "_" + str(n)

  ### randomize the IC  -- will be overwritten in gridcont at finer levels
  init_rho    = random.uniform(5,10)
  init_k      = random.uniform(0.005,0.05)
  init_gamma  = random.uniform(1E4,1E5)

  for atlas in at_list:
    atlas                   = atlas.strip("\n")
    p['n']                  = n
    p['multilevel']         = 1                 # rescale p activations according to Gaussian width on each level
    p['output_dir'] 		    = os.path.join(res_dir, atlas + case_str + '/');  	# results path
    p['d1_path']            = ""  # from segmentation directly
    p['d0_path']            = pat_dir + pat + '_c0Recon_aff2jakob' + n_str + '.nc'              # path to initial condition for tumor
    p['atlas_labels']       = "[wm=6,gm=5,vt=7,csf=8]"# brats'[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
    p['patient_labels']     = "[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]"# brats'[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
    p['a_seg_path']         = atlas_dir + "/" + atlas + "_seg_aff2jakob_ants" + n_str + ".nc"
    p['p_seg_path'] 	      = pat_dir + pat + '_seg_tu_aff2jakob' + n_str + '.nc'
    p['mri_path'] 			    = atlas_dir + "/" + atlas + "_t1_aff2jakob" + n_str + ".nc"
    p['solver'] 			      = 'mass_effect'
    p['model'] 				      = 4                       	# mass effect model
    p['regularization']     = "L2"                      # L2, L1
    p['obs_lambda']         = 1.0                       # if > 0: creates observation mask OBS = 1[TC] + lambda*1[B/WT] from segmentation file
    p['verbosity'] 			    = 1                  		    # various levels of output density
    p['syn_flag'] 			    = 0                  	      # create synthetic data
    p['init_rho'] 			    = init_rho                  # initial guess rho (reaction in wm)
    p['init_k'] 			      = init_k                    # initial guess kappa (diffusivity in wm)
    p['init_gamma'] 		    = init_gamma                # initial guess (forcing factor for mass effect)
    p['nt_inv'] 			      = 25                    	  # number time steps for inversion
    p['dt_inv'] 			      = 0.04                  	  # time step size for inversion
    p['k_gm_wm']            = 0.2                       # kappa ratio gm/wm (if zero, kappa=0 in gm)
    p['r_gm_wm']            = 1                         # rho ratio gm/wm (if zero, rho=0 in gm)
    p['time_history_off'] 	= 0          				        # 1: do not allocate time history (only works with forward solver or FD inversion)
    p['beta_p'] 			      = 0E-4                  	  # regularization parameter
    p['opttol_grad'] 		    = 1E-5             			    # relative gradient tolerance
    p['newton_maxit'] 		  = 50              			    # number of iterations for optimizer
    p['kappa_lb'] 			    = 0.005                     # lower bound kappa
    p['kappa_ub'] 			    = 0.05                 		  # upper bound kappa
    p['rho_lb'] 			      = 2                     	  # lower bound rho
    p['rho_ub'] 			      = 12                    	  # upper bound rho
    p['gamma_lb']           = 0                         # lower bound gamma
    p['gamma_ub'] 			    = 12E4                		  # upper bound gamma
    p['lbfgs_vectors'] 		  = 5        			            # number of vectors for lbfgs update
    p['lbfgs_scale_type']   = "scalar"                  # initial hessian approximation
    p['lbfgs_scale_hist']   = 5                         # used vecs for initial hessian approx
    p['ls_max_func_evals']  = 20                        # number of max line-search attempts
    p['prediction']         = 0                         # enable prediction

###############=== write config to write_path and submit job
    par.submit(p, r, submit_job);

def create_sbatch_header(results_path, idx, compute_sys='frontera'):
  bash_filename = results_path + "/tuinv-job" + str(idx) + ".sh"
  print("creating job file in ", results_path)

  if compute_sys == 'frontera':
      queue = "rtx"
      num_nodes = str(1)
      num_cores = str(1)
  elif compute_sys == 'maverick2':
      queue = "v100"
      num_nodes = str(1)
      num_cores = str(1)
  elif compute_sys == 'longhorn':
      queue = "v100"
      num_nodes = str(1)
      num_cores = str(1)
  elif compute_sys == 'stampede2':
      queue = "skx-normal"
      num_nodes = str(6)
      num_cores = str(128)
  else:
      queue = "normal"
      num_nodes = str(1)
      num_cores = str(1)

  bash_file = open(bash_filename, 'w')
  bash_file.write("#!/bin/bash\n\n");
  bash_file.write("#SBATCH -J tumor-inv\n");
  bash_file.write("#SBATCH -o " + results_path + "/tu_log_" + str(idx) + ".txt\n")
  bash_file.write("#SBATCH -p " + queue + "\n")
  bash_file.write("#SBATCH -N " + num_nodes + "\n")
  bash_file.write("#SBATCH -n " + num_cores + "\n")
  bash_file.write("#SBATCH -t 12:00:00\n\n")
  bash_file.write("source ~/.bashrc\n")

  bash_file.write("\n\n")
  bash_file.close()

  return bash_filename

def update_config(level, bash_file, scripts_path, invdir, atlist, idx):
  n_local = 4 if len(at_list) >= 4*idx + 4 else len(at_list) % 4
  with open(bash_file, 'a') as f:
    if level is not 64:
      for i in range(0,n_local):
        f.write("config_dir_" + str(i) + "=" + invdir + atlist[4*idx + i] + "\n")
      for i in range(0,n_local):
        f.write('python3 ' + scripts_path + "update_tu_config.py -res_path $results_dir_" + str(i) + " -config_path $config_dir_" + str(i) + "\n") 
    for i in range(0,n_local):
      f.write("results_dir_" + str(i) + "=" + invdir + atlist[4*idx + i] + "\n")

  return bash_file
    
def write_tuinv(invdir, atlist, bash_file, idx):
  f = open(bash_file, 'a') 
  n_local = 4 if len(at_list) >= 4*idx + 4 else len(at_list) % 4
  for i in range(0,n_local):
    f.write("CUDA_VISIBLE_DEVICES=" + str(i) + " ibrun ${bin_path} -config ${results_dir_" + str(i) + "}/solver_config.txt > ${results_dir_" + str(i) + "}/log &\n")
  f.write("\n")
  f.write("wait\n")

  f.close()
  return bash_file

def create_level_specific_data(n, pat, data_dir, create = True):
  n_dir    = res + "/" + str(n) + "/"
  sz       = n
  if not os.path.exists(n_dir):
    os.makedirs(n_dir)

  if create: ### files do not exist
    fname = data_dir + "/" + pat + "_t1_aff2jakob.nii.gz"
    fname_n = n_dir + pat + "_t1_aff2jakob_" + str(n) + ".nii.gz"
    if not os.path.exists(fname_n):
      resize_data(fname, fname_n, sz, order = 1)
    fname_n_nc = fname_n.replace(".nii.gz", ".nc")
    if not os.path.exists(fname_n_nc):
      convert_nifti_to_nc(fname_n_nc, n, reverse=True) ## reverse for files resized through nibabel
    fname = data_dir + "/" + pat + "_seg_tu_aff2jakob.nii.gz"
    fname_n = n_dir + pat + "_seg_tu_aff2jakob_" + str(n) + ".nii.gz"
    if not os.path.exists(fname_n):
      resize_data(fname, fname_n, sz, order = 0)
    fname_n_nc = fname_n.replace(".nii.gz", ".nc")
    if not os.path.exists(fname_n_nc):
      convert_nifti_to_nc(fname_n_nc, n, reverse=True) ## reverse for files resized through nibabel
    fname = data_dir + "/" + pat + "_c0Recon_aff2jakob.nii.gz"
    fname_n = n_dir + pat + "_c0Recon_aff2jakob_" + str(n) + ".nii.gz"
    if not os.path.exists(fname_n):
      resize_data(fname, fname_n, sz, order = 1)
    fname_n_nc = fname_n.replace(".nii.gz", ".nc")
    if not os.path.exists(fname_n_nc):
      convert_nifti_to_nc(fname_n_nc, n, reverse=True) ## reverse for files resized through nibabel
  else:
    fname = data_dir + "/" + pat + "_t1_aff2jakob.nii.gz"
    fname_n = n_dir + pat + "_t1_aff2jakob_" + str(n) + ".nii.gz"
    if not os.path.exists(fname_n):
      shutil.copy(fname, fname_n)
    fname_n_nc = fname_n.replace(".nii.gz", ".nc")
    if not os.path.exists(fname_n_nc):
      convert_nifti_to_nc(fname_n_nc, n)
    fname = data_dir + "/" + pat + "_seg_tu_aff2jakob.nii.gz"
    fname_n = n_dir + pat + "_seg_tu_aff2jakob_" + str(n) + ".nii.gz"
    if not os.path.exists(fname_n):
      shutil.copy(fname, fname_n)
    fname_n_nc = fname_n.replace(".nii.gz", ".nc")
    if not os.path.exists(fname_n_nc):
      convert_nifti_to_nc(fname_n_nc, n)
    fname = data_dir + "/" + pat + "_c0Recon_aff2jakob.nii.gz"
    fname_n = n_dir + pat + "_c0Recon_aff2jakob_" + str(n) + ".nii.gz"
    if not os.path.exists(fname_n):
      shutil.copy(fname, fname_n)
    fname_n_nc = fname_n.replace(".nii.gz", ".nc")
    if not os.path.exists(fname_n_nc):
      convert_nifti_to_nc(fname_n_nc, n)

#--------------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Mass effect inversion',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-p', '--patient_dir', type = str, help = 'path to patients (brats format)', required = True) 
  r_args.add_argument('-a', '--atlas_dir', type = str, help = 'path to atlases', required = True) 
  r_args.add_argument('-x', '--results_dir', type = str, help = 'path to results', required = True) 
  r_args.add_argument('-c', '--code_dir', type = str, help = 'path to tumor solver code', required = True) 
  r_args.add_argument('-csys', '--compute_sys', type = str, help = 'compute system', default = 'frontera') 
  args = parser.parse_args();
  submit_job = True

  patient_list = os.listdir(args.patient_dir)
  if os.path.exists(args.patient_dir + "/failed.txt"): ### some patients have failed gridcont; ignore them
    with open(args.patient_dir + "/failed.txt", "r") as f:
      lines = f.readlines()
    for l in lines:
      failed_pat = l.strip("\n")
      print("ignoring failed patient {}".format(failed_pat))
      patient_list.remove(failed_pat)

  ### SNAFU
  patient_list = ["Brats18_CBICA_AAP_1"]
  #patient_list = ["Brats18_CBICA_AAP_1", "Brats18_CBICA_ABO_1", "Brats18_CBICA_AMH_1"]

  in_dir      = args.patient_dir
  results_dir = args.results_dir
  atlas_dir   = args.atlas_dir
  code_dir    = args.code_dir

  for pat in patient_list:
    data_dir  = os.path.join(os.path.join(in_dir, pat), "aff2jakob")
    res = results_dir + "/" + pat + "/tu/"
    respat = results_dir + "/" + pat
    stat = results_dir + "/" + pat + "/stat/"
    if not os.path.exists(res):
      os.makedirs(res)
    if not os.path.exists(stat):
      os.makedirs(stat)
    listfile = respat + "/atlas-list.txt"

    ### (1) create list of candidate atlases
    at_list = []
    if not os.path.exists(listfile):
      f  = open(listfile, "w+")
      fa = open(atlas_dir + "/adni-atlas-stats.csv", "r")
      fp = open(in_dir + "/brats-pat-stats.csv", "r")
      la = fa.readlines()
      lp = fp.readlines()
      for l in lp:
        if pat in l:
          vt_pat = float(l.split(",")[1])

      for l in la:
        vals = l.split(",")
        if float(vals[1]) >= vt_pat and float(vals[1]) < 1.3*vt_pat: ### choose atlases whose vt vol is greater than pat
          at_list.append(vals[0])
      
      n_samples = 16 if len(at_list) > 16 else len(at_list)
      at_list = random.sample(at_list, n_samples) ### take a random subset
      for item in at_list:
        f.write(item + "\n")

      f.close()
      fa.close()
      fp.close()
    else:
      with open(listfile, "r") as f:
        lines = f.readlines()
      for l in lines:
        at_list.append(l.strip("\n"))
   

    ### create data files
    create_level_specific_data(64, pat, data_dir)
    create_level_specific_data(128, pat, data_dir)
    create_level_specific_data(256, pat, data_dir, create=False) ### just copy these files
    ### (2)  create tumor solver configs
    levels   = [64,128,256]
    for n in levels:
      pat_dir    = res + "/" + str(n) + "/"
      atlas_dir_level  = atlas_dir + "/" + str(n) + "/"
      create_tusolver_config(n, pat, pat_dir, atlas_dir_level, pat_dir)
    
    ### (3)  create job files in tusolver results directories
    numatlas = len(at_list)
    numjobs  = math.ceil(numatlas/4)
    bin_path = code_dir + "build/last/tusolver" 
    scripts_path = code_dir + "scripts/masseffect/"
    for i in range(0,numjobs):
      bash_file = create_sbatch_header(respat, i, compute_sys = "frontera")
      with open(bash_file, 'a') as f:
        f.write("bin_path=" + bin_path + "\n")
      
      ### create tumor_inv stats
      for n in levels: 
        res_level = res + "/" + str(n) + "/"
        bash_file = update_config(n, bash_file, scripts_path, res_level, at_list, i) 
        bash_file = write_tuinv(res_level, at_list, bash_file, i)

#      ### extract stats
#      with open(bash_file, 'a') as f:
#        f.write("python3 " + scripts_path + "/extract_stats.py -n 256 -res_path " + stat + " -inv_path " + res + "/256/ \n") 

      if submit_job:
        subprocess.call(['sbatch', bash_file])

