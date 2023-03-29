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
from gridcont import batch_sparsetil as til
input = {}
## SETTTINGS ##
# =======================================================================
# == specify any extra modules to be loaded, one string, may contain newline
# bashrc is sourced prior to this command.
#input['extra_modules'] = "module load petsc/3.11"

### == compute system
input['system']              = 'lonestar'
### == define lambda for observation operator (for devs)
input['obs_lambda']          = 1
### == define segmentation labels
input['segmentation_labels'] = "0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm"
### == submit the jobs
input['submit']              = False
### == flag will use gpus
input['gpu_flag']            = True
### == if using gpus; specify how many gpus per compute node to use
input['num_gpus_per_node']   = 3  ### this will run num_gpus_per_node patients parallely on the gpus
### == how many patient blocks in a job (will run num_gpus_per_node*patients_per_job in total in a job)
input['patients_per_job']    = 1
### == path to all patients (assumes a brats directory structure)
input['path_to_all_patients'] = '/scratch/07544/ghafouri/data/brats2020/'
### == custom list of patients (can be a single patient); keep empty to simply walk through all patients
input['patient_list']        = ['BraTS20_Training_005', 'BraTS20_Training_008', 'BraTS20_Training_013', 'BraTS20_Training_014', 'BraTS20_Training_015', 'BraTS20_Training_043', 'BraTS20_Training_048', 'BraTS20_Training_063', 'BraTS20_Training_086', 'BraTS20_Training_087', 'BraTS20_Training_108', 'BraTS20_Training_146', 'BraTS20_Training_165', 'BraTS20_Training_175', 'BraTS20_Training_178', 'BraTS20_Training_195', 'BraTS20_Training_240', 'BraTS20_Training_241', 'BraTS20_Training_282', 'BraTS20_Training_297', 'BraTS20_Training_304', 'BraTS20_Training_338', 'BraTS20_Training_355']
### == path to all the jobs and results
input['job_path']            = "/scratch/07544/ghafouri/results/til_inv_init/"

cmd = "source ~/.bashrc\n\n"
cmd += "source /work2/07544/ghafouri/ls6/gits/claire_glia.sh\n\n"
input['extra_modules'] = cmd

# =======================================================================
#                        END INPUT
# =======================================================================

### run everything
til.batch_til_and_run(input)
