import os, sys, warnings, argparse, subprocess
import numpy as np
import scipy as sc
from scipy.ndimage import gaussian_filter
import math
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../utils/')
from file_io import writeNII, createNetCDF
from image_tools import resizeImage, resizeNIIImage
from register import create_patient_labels, create_atlas_labels, register, transport
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import params as par
import shutil

def write_sbatch_header(job_file, results_path, idx, compute_sys='frontera', num_pat_per_job = 1):
  bash_file = open(job_file, 'w')
  ############### === define parameters
  p = {}
  r = {}
  r['nodes']     = 1
  r['mpi_tasks']  = 1
  r['compute_sys'] = compute_sys
  r['wtime_h']   = 3 * num_pat_per_job
  r['wtime_m']   = 0
  r['log_dir']   = results_path
  r['log_name']  = 'log_' + str(idx)
#  bash_file.write("#!/bin/bash\n\n");
  job_header = par.write_jobscript_header(p, r, use_gpu=True)
  bash_file.write(job_header)
  bash_file.close()


def batch_jobs_and_run(args):
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

  block_job = False
  if block_job:
    it = 0
    num_pats = 50
    patient_list = patient_list[it*num_pats:it*num_pats + num_pats]
  else:
    it = 0

  for item in patient_list:
    print(item)

  print("setup complete, submitting jobs...")

  submit_job  = True
  in_dir      = args.patient_dir
  results_dir = args.results_dir
  job_dir     = args.job_dir
  if not os.path.exists(job_dir):
    os.makedirs(job_dir)

  code_dir    = args.code_dir
  num_pats    = len(patient_list)
  num_pat_per_job = args.num_pat_per_job
  num_jobs        = int(16/args.num_gpus)
  
  for ct in range(0,math.ceil(num_pats/num_pat_per_job)):
    job_bundle_dir = job_dir + "/job-" + str(ct) + "/"
    if not os.path.exists(job_bundle_dir):
      os.makedirs(job_bundle_dir)
    header_write = np.ones((num_jobs,1), dtype=bool)
    for i in range(0,num_pat_per_job):
      idx = ct*num_pat_per_job + i
      if idx < num_pats:
        # patient exists
        pat = patient_list[idx]
        cur_dir = results_dir + "/" + pat + "/"
        # check how many jobs are there
        n_j = 0
        for item in os.listdir(cur_dir):
          if item.find("job") is not -1:
            job_file = job_bundle_dir + "/job-" + str(n_j) + ".sh"
            if header_write[n_j]: 
              write_sbatch_header(job_file, job_bundle_dir, n_j, args.compute_sys, num_pat_per_job)
              header_write[n_j] = False
            with open(cur_dir + item, "r") as f:
              lines = f.readlines()
            with open(job_file, "a") as f:
              f.write("\n ######## PATIENT: " + str(pat) + " ######## \n")
              f.writelines(lines[11:])
              f.write("\n\n")
            n_j += 1
    if submit_job:
      for job_file in os.listdir(job_bundle_dir):
        if job_file.find("job") is not -1:
          subprocess.call(['sbatch', job_bundle_dir + job_file])


#--------------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Mass effect inversion',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-p', '--patient_dir', type = str, help = 'path to patients (brats format)', required = True) 
  r_args.add_argument('-x', '--results_dir', type = str, help = 'path to results', required = True) 
  r_args.add_argument('-j', '--job_dir', type = str, help = 'path to output jobs', required = True) 
  r_args.add_argument('-c', '--code_dir', type = str, help = 'path to tumor solver code', required = True) 
  r_args.add_argument('-csys', '--compute_sys', type = str, help = 'compute system', default = 'frontera') 
  args = parser.parse_args();

  batch_jobs_and_run(args)
