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
from run_masseffect import *
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import params as par
import random
import shutil


def mod_tu_config(indir, resdir, atlas_list):
  for atlas in atlas_list:
    srcconfig = indir + atlas + "/solver_config.txt"
    dstconfig = resdir + atlas + "/solver_config.txt"
    if not os.path.exists(resdir + atlas):
      os.makedirs(resdir + atlas)
    print("copying and modifying solver config for atlas ", atlas)
    shutil.copy(srcconfig, dstconfig)
    output_dir = resdir + atlas + "/"
    with open(dstconfig, "r") as f:
      lines = f.readlines()
    with open(dstconfig, "w") as f:
      for line in lines:
        if "output_dir" in line:
          print("...updating output_dir to ", output_dir)
          f.write("output_dir=" + output_dir + "\n")
        else:
          f.write(line)

def run_nome(args):
  if not args.me_results_dir:
    ## mass effect inversion not done: run this standalone
    run(args)
  else:
    me_dir = args.me_results_dir
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


    for pat in patient_list:
      data_dir  = os.path.join(os.path.join(in_dir, pat), "aff2jakob")
      res = results_dir + "/" + pat + "/tu/"
      respat = results_dir + "/" + pat
      stat = results_dir + "/" + pat + "/stat/"
      reg =  me_dir + "/" + pat + "/reg/"
      if not os.path.exists(res):
        os.makedirs(res)
      if not os.path.exists(stat):
        os.makedirs(stat)
      listfile = respat + "/atlas-list.txt"
      if not os.path.exists(listfile):
        srcfile = me_dir + "/" + pat + "/atlas-list.txt"
        if not os.path.exists(srcfile):
          print("ERROR! {} does not exist! moving to the next patient...".format(srcfile))
          continue
        shutil.copy(srcfile, listfile)

      at_list = []
      with open(listfile, "r") as f:
        lines = f.readlines()
      for l in lines:
        at_list.append(l.strip("\n"))

      ## copy tumor solver config from masseffect and change the output dir
      res_dir    = res + "/" + str(n) + "/"
      if not os.path.exists(res_dir):
        os.makedirs(res_dir)
      mod_tu_config(indir=me_dir + "/" + pat + "/tu/" + str(n) + "/", resdir=res_dir, atlas_list=at_list) 
      ## create job files in tusolver results directories
      numatlas = len(at_list)
      numjobs  = math.ceil(numatlas/4)
      bin_path = code_dir + "build/last/tusolver" 
      scripts_path = code_dir + "scripts/"
      r = {}
      r['compute_sys'] = args.compute_sys
      if not args.submit:
        for i in range(0,numjobs):
          bash_file = create_sbatch_header(respat, i, compute_sys = args.compute_sys)
          with open(bash_file, 'a') as f:
            f.write("bin_path=" + bin_path + "\n")
          ### create tumor_inv stats
          res_level = res + "/" + str(n) + "/"
          ###bash_file = convert_and_move(n, bash_file, scripts_path, at_list, reg, pat, res_level, i)
          bash_file = write_tuinv(res_level, at_list, bash_file, i, args.num_gpus, r) 
      else:
        for i in range(0,numjobs):
          bash_file = respat + "/job" + str(i) + ".sh"
          subprocess.call(['sbatch', bash_file])




#--------------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Runs no-mass effect model using data from mass effect inversion',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-p', '--patient_dir', type = str, help = 'path to patients (brats format)', required = True) 
  r_args.add_argument('-a', '--atlas_dir', type = str, help = 'path to atlases', required = True) 
  r_args.add_argument('-xm', '--me_results_dir', type = str, help = 'path to masseffect inversion results') 
  r_args.add_argument('-x', '--results_dir', type = str, help = 'path to results', required = True) 
  r_args.add_argument('-c', '--code_dir', type = str, help = 'path to tumor solver code', required = True) 
  r_args.add_argument('-n', '--n_resample', type = int, help = 'size for inversion', default = 160) 
  r_args.add_argument('-r', '--reg', type = int, help = 'perform registration', default = 0) 
  r_args.add_argument('-rc', '--claire_dir', type = str, help = 'path to claire bin', default = "") 
  r_args.add_argument('-csys', '--compute_sys', type = str, help = 'compute system', default = 'longhorn') 
  r_args.add_argument('-submit', action = 'store_true', help = 'submit jobs (after they have been created)') 
  args = parser.parse_args();
  args.num_gpus = 4

  in_dir      = args.patient_dir
  results_dir = args.results_dir
  atlas_dir   = args.atlas_dir
  code_dir    = args.code_dir
  n           = args.n_resample
  reg_flag    = args.reg
  claire_dir  = args.claire_dir

  run_nome(args)
