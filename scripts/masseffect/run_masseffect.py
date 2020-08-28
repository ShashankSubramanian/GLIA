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

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../utils/')
from file_io import writeNII, createNetCDF
from image_tools import resizeImage, resizeNIIImage
from register import create_patient_labels, create_atlas_labels, register, transport
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
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

  for atlas in at_list:
    ### randomize the IC  -- will be overwritten in gridcont at finer levels
    init_rho                = random.uniform(5,10)
    init_k                  = random.uniform(0.005,0.05)
    init_gamma              = random.uniform(1E4,1E5)
    atlas                   = atlas.strip("\n")

    p['n']                  = n
    p['multilevel']         = 1                 # rescale p activations according to Gaussian width on each level
    p['output_dir']         = os.path.join(res_dir, atlas + case_str + '/');    # results path
    p['d1_path']            = ""  # from segmentation directly
    p['d0_path']            = pat_dir + atlas + "/" + pat + '_c0Recon_transported' + n_str + '.nc'              # path to initial condition for tumor
#    p['d0_path']            = pat_dir + pat + '_c0Recon_aff2jakob' + n_str + '.nc'              # path to initial condition for tumor
    p['atlas_labels']       = "[wm=6,gm=5,vt=7,csf=8]"# brats'[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
    p['patient_labels']     = "[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]"# brats'[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
    p['a_seg_path']         = atlas_dir + "/" + atlas + "_seg_aff2jakob_ants" + n_str + ".nc"
    p['p_seg_path']         = pat_dir + pat + '_seg_ants_aff2jakob' + n_str + '.nc'
    p['mri_path']           = atlas_dir + "/" + atlas + "_t1_aff2jakob" + n_str + ".nc"
    p['solver']             = 'mass_effect'
    p['model']              = 4                         # mass effect model
    p['regularization']     = "L2"                      # L2, L1
    p['obs_lambda']         = 1.0                       # if > 0: creates observation mask OBS = 1[TC] + lambda*1[B/WT] from segmentation file
    p['verbosity']          = 1                         # various levels of output density
    p['syn_flag']           = 0                         # create synthetic data
    p['init_rho']           = init_rho                  # initial guess rho (reaction in wm)
    p['init_k']             = init_k                    # initial guess kappa (diffusivity in wm)
    p['init_gamma']         = init_gamma                # initial guess (forcing factor for mass effect)
    p['nt_inv']             = 25                        # number time steps for inversion
    p['dt_inv']             = 0.04                      # time step size for inversion
    p['k_gm_wm']            = 0.2                       # kappa ratio gm/wm (if zero, kappa=0 in gm)
    p['r_gm_wm']            = 1                         # rho ratio gm/wm (if zero, rho=0 in gm)
    p['time_history_off']   = 0                         # 1: do not allocate time history (only works with forward solver or FD inversion)
    p['beta_p']             = 0E-4                      # regularization parameter
    p['opttol_grad']        = 1E-5                      # relative gradient tolerance
    p['newton_maxit']       = 50                        # number of iterations for optimizer
    p['kappa_lb']           = 0.005                     # lower bound kappa
    p['kappa_ub']           = 0.05                      # upper bound kappa
    p['rho_lb']             = 2                         # lower bound rho
    p['rho_ub']             = 12                        # upper bound rho
    p['gamma_lb']           = 0                         # lower bound gamma
    p['gamma_ub']           = 12E4                      # upper bound gamma
    p['lbfgs_vectors']      = 5                         # number of vectors for lbfgs update
    p['lbfgs_scale_type']   = "scalar"                  # initial hessian approximation
    p['lbfgs_scale_hist']   = 5                         # used vecs for initial hessian approx
    p['ls_max_func_evals']  = 20                        # number of max line-search attempts
    p['prediction']         = 1                         # enable prediction
    p['pred_times']         = [1.0, 1.2, 1.5]           # times for prediction
    p['dt_pred']            = 0.02                      # time step size for prediction

###############=== write config to write_path and submit job
    par.submit(p, r, submit_job);

def create_sbatch_header(results_path, idx, compute_sys='frontera'):
  bash_filename = results_path + "/job" + str(idx) + ".sh"
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
  bash_file.write("#SBATCH -o " + results_path + "/log_" + str(idx) + ".txt\n")
  bash_file.write("#SBATCH -p " + queue + "\n")
  bash_file.write("#SBATCH -N " + num_nodes + "\n")
  bash_file.write("#SBATCH -n " + num_cores + "\n")
  bash_file.write("#SBATCH -t 03:00:00\n\n")
  bash_file.write("source ~/.bashrc\n")

  bash_file.write("\n\n")
  bash_file.close()

  return bash_filename

 
def write_tuinv(invdir, atlist, bash_file, idx):
  f = open(bash_file, 'a') 
  n_local = 4 if len(atlist) >= 4*idx + 4 else len(atlist) % 4
  for i in range(0,n_local):
    f.write("results_dir_" + str(i) + "=" + invdir + atlist[4*idx + i] + "\n")
  for i in range(0,n_local):
    f.write("CUDA_VISIBLE_DEVICES=" + str(i) + " ibrun ${bin_path} -config ${results_dir_" + str(i) + "}/solver_config.txt > ${results_dir_" + str(i) + "}/solver_log.txt 2>&1 &\n")
  f.write("\n")
  f.write("wait\n")

  f.close()
  return bash_file

def create_level_specific_data(n, pat, data_dir, res, create = True):
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
    fname = data_dir + "/" + pat + "_seg_ants_aff2jakob.nii.gz"
    fname_n = n_dir + pat + "_seg_ants_aff2jakob_" + str(n) + ".nii.gz"
    if not os.path.exists(fname_n):
      resize_data(fname, fname_n, sz, order = 0)
    fname_n_nc = fname_n.replace(".nii.gz", ".nc")
    if not os.path.exists(fname_n_nc):
      convert_nifti_to_nc(fname_n_nc, n, reverse=True)
    fname = data_dir + "/" + pat + "_c0Recon_aff2jakob.nii.gz"
    fname_n = n_dir + pat + "_c0Recon_aff2jakob_" + str(n) + ".nii.gz"
    if not os.path.exists(fname_n):
      resize_data(fname, fname_n, sz, order = 1)
    fname_n_nc = fname_n.replace(".nii.gz", ".nc")
    if not os.path.exists(fname_n_nc):
      convert_nifti_to_nc(fname_n_nc, n, reverse=True)
  else:
    fname = data_dir + "/" + pat + "_t1_aff2jakob.nii.gz"
    fname_n = n_dir + pat + "_t1_aff2jakob_" + str(n) + ".nii.gz"
    if not os.path.exists(fname_n):
      shutil.copy(fname, fname_n)
    fname_n_nc = fname_n.replace(".nii.gz", ".nc")
    if not os.path.exists(fname_n_nc):
      convert_nifti_to_nc(fname_n_nc, n)
    fname = data_dir + "/" + pat + "_seg_ants_aff2jakob.nii.gz"
    fname_n = n_dir + pat + "_seg_ants_aff2jakob_" + str(n) + ".nii.gz"
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

def write_reg(reg, pat, data_dir, atlas_dir, at_list, claire_dir, bash_file, idx):
  """ writes registration cmds """
  n_local = 4 if len(at_list) >= 4*idx + 4 else len(at_list) % 4
  ### create template(patient) labels
  if not os.path.exists(reg + pat + "_vt.nii.gz"):
    create_patient_labels(data_dir + "/" + pat + "_seg_ants_aff2jakob.nii.gz", reg, pat)
  ### create reference(atlas) labels
  for i in range(0, n_local):
    at = at_list[4*idx+i]
    if not os.path.exists(reg + at + "/" + at + "_vt.nii.gz"):
      create_atlas_labels(atlas_dir + at + "_seg_aff2jakob_ants.nii.gz", reg + at, at) 
    ### register
    bash_file = register(claire_dir, reg+at, at, reg, pat, bash_file, i)
  
  with open(bash_file, "a") as f:
    f.write("\n\nwait\n\n")
  
  for i in range(0, n_local):
    ### transport
    at = at_list[4*idx+i]
    bash_file = transport(claire_dir, reg+at, data_dir + "/" + pat + "_c0Recon_aff2jakob.nii.gz", pat + "_c0Recon", bash_file, i)    
    bash_file = transport(claire_dir, reg+at, data_dir + "/" + pat + "_seg_ants_aff2jakob.nii.gz", pat + "_labels", bash_file, i)    

  with open(bash_file, "a") as f:
    f.write("\n\nwait\n\n")
  
  return bash_file

def convert_and_move(n, bash_file, scripts_path, at_list, reg, pat, tu, idx):
  n_local = 4 if len(at_list) >= 4*idx + 4 else len(at_list) % 4
  with open(bash_file, "a") as f:
    for i in range(0, n_local):
      ### convert transported c0 to netcdf and mv it
      at = at_list[4*idx+i]
      nm = pat + "_c0Recon_transported_" + str(n) + ".nc"
      ###if not os.path.exists(reg + at + "/" + nm):
      f.write("python3 " + scripts_path + "/helpers/convert_to_netcdf.py -i " + reg + at + "/" + pat + "_c0Recon_transported.nii.gz -n " + str(n) + " -resample\n")

    f.write("\n\n")

    for i in range(0, n_local):
      at = at_list[4*idx+i]
      nm = pat + "_c0Recon_transported_" + str(n) + ".nc"
      f.write("cp " + reg + at + "/" + nm + " " + tu + "/" + at + "/" + nm + "\n") 

    f.write("\n\n")
  return bash_file

def find_crossover(arr, low, high, x):
  if x < arr[low]:
    return low
  if x > arr[high]:
    return high
  mid = (low + high) // 2
  if arr[mid] <= x and arr[mid+1] > x:
    return mid
  if arr[mid] < x:
    return find_crossover(arr, mid+1, high, x)
  return find_crossover(arr, low, mid-1, x)

   

def find_k_closest(atlas_dict, k, elem, leave_out=[]):
  """ finds k closest elements to a list with duplicates in leave_out not counted """
  if len(leave_out) > 0:
    atlas_dict_mod = {key:val for key,val in atlas_dict.items() if key not in leave_out}
  else:
    atlas_dict_mod = atlas_dict.copy()

  vt = [v for k,v in atlas_dict_mod.items()]
  names = [k for k,v in atlas_dict_mod.items()]

  ##base case
  if elem < vt[0]:
    return names[0:k]
  if elem > vt[-1]:
    return names[-k:]

  n = len(vt)
  left = find_crossover(vt, 0, n-1, elem)
  right = left + 1

  if vt[left] == elem:
    left -= 1

  count = 0
  at_list = []
  while left >= 0 and right < n and count < k:
    if elem - vt[left] < vt[right] - elem:
      at_list.append(names[left])
      left -= 1
    else:
      at_list.append(names[right])
      right += 1
    count += 1

  ### elements remaining
  while count < k and left >= 0:
    at_list.append(names[left])
    left -= 1
    count += 1

  while count < k  and right < n:
    at_list.append(names[right])
    right += 1
    count += 1

  return at_list

def run(args):
  #patient_list = os.listdir(args.patient_dir)
  with open(args.patient_dir + "/pat_stats.csv", "r") as f:
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

  other_remove = []
  #other_remove = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1", "Brats18_CBICA_AAP_1"]
  for others in other_remove:
    patient_list.remove(others)

  ## keep off
  block_job = False
  if block_job:
    it = 0
    num_pats = 50
    patient_list = patient_list[150:]
#    patient_list = patient_list[it*num_pats:it*num_pats + num_pats]
  else:
    it = 0

  if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

  mylog = open(args.results_dir + "/config_log_" + str(it) + ".log", "a")
  sys.stdout = mylog

  for item in patient_list:
    print(item)
  print(len(patient_list))
  #exit()

  in_dir      = args.patient_dir
  results_dir = args.results_dir
  atlas_dir   = args.atlas_dir
  code_dir    = args.code_dir
  n           = args.n_resample
  reg_flag    = args.reg
  claire_dir  = args.claire_dir

  ### create atlas dict for corner cases and atlas selection
  at_dict = {}
  fa = open(atlas_dir + "/adni-atlas-stats.csv", "r")
  la = fa.readlines()
  for l in la:
    at_dict[l.split(",")[0]] = l.split(",")[1]
  ### sort the dict
  at_dict = {k:float(v) for k,v in sorted(at_dict.items(), key=lambda item : item[1])}
  all_at_vt = [v for k,v in at_dict.items()]
  all_at_names = [k for k,v in at_dict.items()]

  for pat in patient_list:
    data_dir  = os.path.join(os.path.join(in_dir, pat), "aff2jakob")
    res = results_dir + "/" + pat + "/tu/"
    respat = results_dir + "/" + pat
    stat = results_dir + "/" + pat + "/stat/"
    reg = results_dir + "/" + pat + "/reg/"
    if not os.path.exists(res):
      os.makedirs(res)
    if not os.path.exists(stat):
      os.makedirs(stat)
    if not os.path.exists(reg):
      os.makedirs(reg)
    listfile = respat + "/atlas-list.txt"

    ### (1) create list of candidate atlases
    at_list = []
    at_list_err = False
    if not os.path.exists(listfile):
      f  = open(listfile, "w+")
      fp = open(in_dir + "/pat_stats.csv", "r")
      lp = fp.readlines()
      for l in lp:
        if pat in l:
          vt_pat = float(l.split(",")[1])

      for l in la:
        vals = l.split(",")
        if float(vals[1]) >= vt_pat and float(vals[1]) < 1.3*vt_pat: ### choose atlases whose vt vol is greater than pat
          at_list.append(vals[0])

      min_at_needed = 4
      if len(at_list) < min_at_needed:
        at_list_err = True
        print("at_list selection failed for patient {}; finding nearest neighbors instead".format(pat))
        ### cannot find sufficient atlases; relax the restrictions
        if len(at_list) == 0:
          ### pat vt is greater than all the atlases
          ### pat vt is very small (at > 1.3pat for all atlases)
          print("no atlases were found; nearest neighbors selected")
          at_list = find_k_closest(at_dict, k=min_at_needed, elem=vt_pat)
        else:
          num_needed = min_at_needed - len(at_list)
          print("insufficient atlases were found; trying 50% difference for remaining {} atlases".format(num_needed))
          ct = 0
          for l in la:
            vals = l.split(",")
            if float(vals[1]) >= vt_pat and float(vals[1]) < 1.5*vt_pat and vals[0] not in at_list:
              at_list.append(vals[0])
              ct += 1
            if ct >= num_needed:
              break
          
          if ct >= num_needed:
            print("found atlases with relaxed constraint")
          else:
            num_needed -= ct
            print("insufficient atlases were found; nearest neighbors selected for remaining {} atlases".format(num_needed))
            extra = find_k_closest(at_dict, k=num_needed, elem=vt_pat, leave_out = at_list)
            for item in extra:
              at_list.append(item)

      if at_list_err:
        print(at_list)
        ##input("Press enter to continue...")
            

      n_samples = 16 if len(at_list) > 16 else len(at_list)
      at_list = at_list[0:n_samples] ###random.sample(at_list, n_samples) ### take a random subset
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
    if n is not 256:
      create_level_specific_data(n, pat, data_dir, res)
    else:
      create_level_specific_data(n, pat, data_dir, res, create=False) ### just copy these files

    ### (2)  create tumor solver configs
    pat_dir    = res + "/" + str(n) + "/"
    atlas_dir_level  = atlas_dir + "/" + str(n) + "/"
    create_tusolver_config(n, pat, pat_dir, atlas_dir_level, pat_dir)
    
    ### (3)  create job files in tusolver results directories
    numatlas = len(at_list)
    numjobs  = math.ceil(numatlas/4)
    bin_path = code_dir + "build/last/tusolver" 
    scripts_path = code_dir + "scripts/"
    if not args.submit:
      for i in range(0,numjobs):
        bash_file = create_sbatch_header(respat, i, compute_sys = args.compute_sys)
        with open(bash_file, 'a') as f:
          f.write("bin_path=" + bin_path + "\n")
          if reg_flag:
            f.write("claire_bin_path=" + claire_dir + "\n")
        
        ### create tumor_inv stats
        res_level = res + "/" + str(n) + "/"
        if reg_flag: ### perform registration first
          bash_file = write_reg(reg, pat, data_dir, atlas_dir + "/nifti/", at_list, claire_dir, bash_file, i)
        bash_file = convert_and_move(n, bash_file, scripts_path, at_list, reg, pat, res_level, i)
        bash_file = write_tuinv(res_level, at_list, bash_file, i) 
    else:
      for i in range(0,numjobs):
        bash_file = respat + "/job" + str(i) + ".sh"
        subprocess.call(['sbatch', bash_file])


#--------------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Mass effect inversion',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-p', '--patient_dir', type = str, help = 'path to patients (brats format)', required = True) 
  r_args.add_argument('-a', '--atlas_dir', type = str, help = 'path to atlases', required = True) 
  r_args.add_argument('-x', '--results_dir', type = str, help = 'path to results', required = True) 
  r_args.add_argument('-c', '--code_dir', type = str, help = 'path to tumor solver code', required = True) 
  r_args.add_argument('-n', '--n_resample', type = int, help = 'size for inversion', default = 160) 
  r_args.add_argument('-r', '--reg', type = int, help = 'perform registration', default = 0) 
  r_args.add_argument('-rc', '--claire_dir', type = str, help = 'path to claire bin', default = "") 
  r_args.add_argument('-csys', '--compute_sys', type = str, help = 'compute system', default = 'frontera') 
  r_args.add_argument('-submit', action = 'store_true', help = 'submit jobs (after they have been created)') 
  args = parser.parse_args();
  run(args)

