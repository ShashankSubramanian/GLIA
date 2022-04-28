"""
    This script runs the inverse ensemble mass effect solver
    on GPUs based on available resources
    Currently, minor mods are needed for CPUs in this script: TODO
"""
import os, sys
import params as par
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/utils/')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/helpers/')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/masseffect/')
from masseffect import run_registration as ensemble
from masseffect import run_multiple_patients as batch_ensemble
from helpers import copy_til


args = ensemble.Args()

for i in range(8,9):

  syn = 'case%d'%i

  ### path to patients [requires affine to jakob; directory structure is args.patient_dir/{patient_ID}/aff2jakob/{data}]
  args.patient_dir = '/scratch1/07544/ghafouri/results/syndata/'+syn+'/C1_me/'
  ### path to atlases [adni]
  args.atlas_dir   = "/scratch1/07544/ghafouri/results/syndata/"+syn+"/C1_me/"
  ### path to TIL results; results directory from inverse_til.py script; assumes a directory structure
  args.til_dir     = "/scratch1/07544/ghafouri/results/syn_results/C1_me/til_inv/"+syn+"/"
  ### path to tumor solver code
  args.code_dir    = "/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/"
  ### path to registration solver code binaries
  args.claire_dir  = "/work2/07544/ghafouri/frontera/gits/claire/bin/"
  ### path to results [solver will make this]
  args.results_dir = "/scratch1/07544/ghafouri/results/syn_results/me_inv/"+syn+"/"
  if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)
  ### compute system [TACC systems are: maverick2, frontera, stampede2, longhorn]; others [rebels]
  args.compute_sys = "frontera"



  ### system settings
  ### submit the jobs
  submit               = True
  ### directory to store all the batched jobs; can change if different from results
  args.job_dir         = args.results_dir
  ### number of gpus per node
  args.num_gpus        = 1
  ### number of patients per job; this will serially place patients in the same job; suggest to set
  ### it to 16/num_gpus
  args.num_pat_per_job = 1

  ### ----------------------------------------  INPUT END ------------------------------------------

  ### copy til locations
  #copy_til.copy_inv_til(args)


  cmd = 'python multispecies/run_registration.py '
  cmd += '-p '+args.patient_dir+' '
  cmd += '-a '+args.patient_dir+' '
  cmd += '-at_name '+syn+' '
  cmd += '-x '+args.results_dir+' '
  cmd += '-c '+args.code_dir+' '
  cmd += '-n '+str(256)+' '
  cmd += '-r '+str(1)+' '
  cmd += '-til '+args.til_dir+ ' '
  cmd += '-rc '+args.claire_dir+' '
  cmd += '-csys'+'frontera'+' '
  cmd += '-pat_seg '+'seg_t1.nii.gz'+' '
  cmd += '-at_seg '+'seg_t0.nii.gz'+' '
  cmd += '-pat '+syn+' '
  #cmd += '-submit '+str(0)+' '
  #cmd += '-syn '+str(1)+' '

  print(cmd)


  os.system(cmd)

  '''
  ### sets up the ensemble inversion by preselecting atlases and creating images for registration
  ### and tumor inversion ~ (if on tacc, run on a compute node)
  ensemble.run(args)

  if submit:
    ## batches patients and submits
    batch_ensemble.batch_jobs_and_run(args)
  '''


