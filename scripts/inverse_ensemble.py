"""
    This script runs the inverse ensemble mass effect solver
"""
import os, sys
import params as par
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/utils/')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/masseffect/')
from masseffect import run_masseffect as ensemble
from masseffect import run_multiple_patients as batch_ensemble


args = ensemble.Args()

### path to patients [requires affine to jakob; directory structure is {ID}/aff2jakob/{data}]
args.patient_dir = "/scratch/05027/shas1693/pglistr_tumor/realdata/"
### path to atlases [adni]
args.atlas_dir   = "/scratch/05027/shas1693/adni-nc/"
### path to tumor solver code
args.code_dir    = "/scratch/05027/shas1693/pglistr_tumor/"
### path to registration solver code binaries
args.claire_dir  = "/scratch/05027/shas1693/claire-dev/bingpu/"
### path to results [solver will make this]
args.results_dir = os.path.join(args.code_dir, "results/ensemble_results/")
### compute system [TACC systems are: maverick2, frontera, stampede2, longhorn]; others [rebels]
args.compute_sys = "longhorn"

### system settings
### submit the jobs
submit               = True
### directory to store all the batched jobs; can change if different from results
args.job_dir         = args.results_dir
### number of gpus per node
args.num_gpus        = 2
### number of patients per job; this will serially place patients in the same job; suggest to set
### it to 16/num_gpus
args.num_pat_per_job = 2

### ----------------------------------------  INPUT END ------------------------------------------

### sets up the ensemble inversion by preselecting atlases and creating images for registration
### and tumor inversion ~ (if on tacc, run on a compute node)
ensemble.run(args)

if submit:
  ## batches patients and submits
  batch_ensemble.batch_jobs_and_run(args)



