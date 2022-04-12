""" 
    This script runs sparse TIL reconstruction for a block/list of patients 
    on GPUs based on available resources
    Currently, minor mods are needed for CPUs in this script: TODO
"""
import os, sys
import params as par
import math
import numpy as np
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/utils/')
from gridcont import batch_sparsetil_gen as til
input = {}
pat_params = {}
## SETTTINGS ##
# =======================================================================
# == specify any extra modules to be loaded, one string, may contain newline
# bashrc is sourced prior to this command.
#input['extra_modules'] = "module load petsc/3.11"

### == compute system
input['system']              = 'frontera'
### == define lambda for observation operator (for devs)
input['obs_lambda']          = 1
### == define segmentation labels
input['segmentation_labels'] = "0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm"
### == submit the jobs
input['submit']              = False
### == flag will use gpus
input['gpu_flag']            = True
### == if using gpus; specify how many gpus per compute node to use
input['num_gpus_per_node']   = 1  ### this will run num_gpus_per_node patients parallely on the gpus
### == how many patient blocks in a job (will run num_gpus_per_node*patients_per_job in total in a job)
input['patients_per_job']    = 1
### == path to all patients (assumes a brats directory structure)
#input['path_to_all_patients'] = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/syndata/'
#input['path_to_all_patients'] = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/syndata/'
input['path_to_all_patients'] = '/scratch1/07544/ghafouri/results/syndata/'
### == custom list of patients (can be a single patient); keep empty to simply walk through all patients
input['patient_list']        = ['case1', 'case2', 'case3', 'case4']
### == path to all the jobs and results
input['job_path']            = "/scratch1/07544/ghafouri/results/syn_results/C1_me/til_inv/"
#input['job_path']            = "/scratch1/07544/ghafouri/results/syn_results/C1/til_inv_tc=nec/"
#input['job_path']            = "/scratch1/07544/ghafouri/results/syn_results/til_inv_tc=en/"

pat_params['patient_data_paths'] = 'C1_me/seg_all_t1.nii.gz'
#pat_params['patient_data_paths'] = 'C1/seg_all_t1_tc=nec.nii.gz'
#pat_params['patient_data_paths'] = '256/seg_all_t1_tc=nec.nii.gz'
#pat_params['patient_data_paths'] = '256/seg_all_t1_tc=en.nii.gz'



# =======================================================================
#                        END INPUT
# =======================================================================

### run everything
til.batch_til_and_run(input, pat_params)










