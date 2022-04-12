import os, sys
import params as par
import math
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../utils/')
from gridcont import run_sparsetil_multilevel_multigpu as gridcont_gpu
from gridcont import run_sparsetil_multilevel as gridcont

#### script to batch patients and run for TIL recon ####
# =======================================================================
def batch_til_and_run(input, pat_params):
  patient_list = input['patient_list']
  path_to_all_patients = input['path_to_all_patients']
  gpu_flag = input['gpu_flag']
  patients_per_job = input['patients_per_job']
  num_gpus_per_node = input['num_gpus_per_node']
  job_path = input['job_path']
  
  if not patient_list:
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
        suffix = "aff2jakob"
        if not os.path.exists(os.path.join(*[path_to_all_patients, patient, suffix, patient + "_t1" + "_" + suffix + ".nii.gz"])):
          continue
        patient_list.append(patient) 

  total_no_patients = len(patient_list)
  print("Running for patients:")
  for pat in patient_list:
    print(pat)
  print("Creating data preprocessing pipeline and job scripts...")

  if gpu_flag:
    num_jobs = math.ceil(total_no_patients/num_gpus_per_node)
    job_idx = 0
    for job in range(0,num_jobs):
      if (job+1)%input['patients_per_job'] == 0:
        job_idx += 1

      if (job+1)*num_gpus_per_node >= total_no_patients:
        ### no more patients; end the job
        input['batch_end'] = True
        if (job+1)%input['patients_per_job'] is not 0:
          job_idx += 1
        patient_local_list = patient_list[job*num_gpus_per_node:]
      else:
        patient_local_list = patient_list[job*num_gpus_per_node:(job+1)*num_gpus_per_node]
      patient_data_paths = []
      output_base_paths = []
      for pat in patient_local_list:
        # == define path to patient segmentation
        patient_data_paths.append(os.path.join(path_to_all_patients, pat, pat_params['patient_data_paths']))
        #patient_data_paths.append(os.path.join(path_to_all_patients, pat) + "/aff2jakob/" + pat + "_seg_ants_aff2jakob.nii.gz")
        # == define path to output dir
        output_base_paths.append(job_path + pat) 
      gridcont_gpu.sparsetil_gridcont_gpu(input, patient_data_paths, output_base_paths, job_path, job_idx, use_gpu = True);
  else:
    job_idx = 0
    for idx,pat in enumerate(patient_list):
      if idx%input['patients_per_job'] == 0:
        job_idx += 1
      if (idx+1) >= total_no_patients:
        input['batch_end'] = True
      # == define path to patient segmentation
      input['patient_path'] = os.path.join(path_to_all_patients, pat) + "/aff2jakob/" + pat + "_seg_ants_aff2jakob.nii.gz"
      # == define path to output dir
      input['output_base_path'] = os.path.join(job_path, pat)
      gridcont.sparsetil_gridcont(input, job_path, job_idx, use_gpu = False);

  print("Finished")
