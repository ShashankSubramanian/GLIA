"""
    This script runs the inverse ensemble mass effect solver
"""
import os, sys
import params as par
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/utils/')
from masseffect import run_masseffect as ensemble


args = ensemble.Args()

### path to patients [requires affine to jakob; directory structure is {ID}/aff2jakob/{data}]
args.patient_dir = "/scratch/05027/shas1693/penn_gbm_survival20/Data/"
### path to atlases [adni]
args.atlas_dir   = "/scratch/05027/shas1693/adni-nc/"
### path to tumor solver code
args.code_dir    = "/scratch/05027/shas1693/pglistr_tumor/"
### path to registration solver code binaries
args.claire_dir  = "/scratch/05027/shas1693/claire-dev/bingpu/"
### path to results [solver will make this]
args.results_dir = os.path.join(args.code_dir, "ensemble_results/")
### compute system [TACC systems are: maverick2, frontera, stampede2, longhorn]; others [rebels]
args.compute_sys = "longhorn"

### sets up the ensemble inversion by preselecting atlases and creating images for registration
### and tumor inversion ~ run on a compute node
ensemble.run(args)
