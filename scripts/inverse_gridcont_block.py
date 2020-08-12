""" 
    This script runs sparse TIL reconstruction for a block/list of patients 
    on GPUs/CPUs based on available resources
"""
import os, sys
import params as par
import math
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/utils/')
from gridcont import run_sparsetil_multilevel_multigpu as gridcont_gpu
from gridcont import run_sparsetil_multilevel as gridcont

input = {}
## SETTTINGS ##
# =======================================================================
# == specify any extra modules to be loaded, one string, may contain newline
# bashrc is sourced prior to this command.
#input['extra_modules'] = "module load petsc/3.11"

# == compute system
input['system'] = 'longhorn'
# == define lambda for observation operator
input['obs_lambda'] = 1
# == define segmentation labels
input['segmentation_labels'] = "0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm"
input['submit'] = False
# =======================================================================

path_to_all_patients = '/scratch/05027/shas1693/tmi-results/'
patient_list = []
if os.path.exists(path_to_all_patients + "/pat_stats.csv"):
  with open(path_to_all_patients + "/pat_stats.csv", "r") as f:
    all_pats = f.readlines()
  patient_list = []
  for l in all_pats:
    patient_list.append(l.split(",")[0])
  if os.path.exists(path_to_all_patients + "/failed.txt"): ### if some patients have failed in some preproc routinh; ignore them
    with open(path_to_all_patients + "/failed.txt", "r") as f:
      lines = f.readlines()
    for l in lines:
      failed_pat = l.strip("\n")
      print("ignoring failed patient {}".format(failed_pat))
      if failed_pat in patient_list:
        patient_list.remove(failed_pat)
else:
  for patient in os.listdir(path_to_all_patients):
    suffix = ""
    if not os.path.exists(os.path.join(os.path.join(path_to_all_patients, patient), patient + "_t1" + suffix + ".nii.gz")):
      continue
    patient_list.append(patient) 

### comment; else use a custom list
patient_list = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1", "Brats18_CBICA_AAP_1"]
total_no_patients = len(patient_list)
gpu_flag = True
job_path= "/scratch/05027/shas1693/pglistr_tumor/results/test3/"
if gpu_flag:
  num_gpus_per_node = 4  ### this will run num_gpus_per_node patients parallely on the gpus
  num_jobs = math.ceil(total_no_patients/num_gpus_per_node)
  for job in range(0,num_jobs):
    patient_local_list = patient_list[job*num_gpus_per_node:(job+1)*num_gpus_per_node]
    patient_data_paths = []
    output_base_paths = []
    for pat in patient_local_list:
      # == define path to patient segmentation
      patient_data_paths.append(os.path.join(path_to_all_patients, pat) + "/aff2jakob/" + pat + "_seg_ants_aff2jakob.nii.gz")
      # == define path to output dir
      output_base_paths.append(job_path + pat)
    gridcont_gpu.sparsetil_gridcont_gpu(input, patient_data_paths, output_base_paths, job_path, use_gpu = True);
else:
  for pat in patient_list:
    # == define path to patient segmentation
    input['patient_path'] = os.path.join(path_to_all_patients, pat) + "/" + pat + "_seg_tu.nii.gz"
    # == define path to output dir
    input['output_base_path'] = '/scratch/05027/shas1693/pglistr_tumor/results/gridcont-test/' + pat
    gridcont.sparsetil_gridcont(input, use_gpu = False);
