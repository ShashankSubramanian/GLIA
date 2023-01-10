"""
    This script runs the inverse TIL solver for a single patient
"""
import os, sys
import params as par
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/utils/')
from gridcont import run_sparsetil_multilevel as gridcont



input = {}
## SETTTINGS ##
# =======================================================================
# == specify any extra modules to be loaded, one string, may contain newline
# bashrc is sourced prior to this command.
input['extra_modules'] = "conda activate gen"
input['extra_modules'] += "\nsource /work2/07544/ghafouri/frontera/gits/env_glia.sh"

# == compute system
input['system'] = 'frontera'

## for synthetic patients, use direct data
#input['patient_path'] = "/scratch/05027/shas1693/pglistr_tumor/results/tc-i/seg_final.nii.gz"
#input['data_path'] = "/scratch/05027/shas1693/pglistr_tumor/results/tc-i/c_final.nii.gz"

# == define path to patient segmentation
input['patient_path'] = '/scratch1/07544/ghafouri/results/syndata/case8/seg_ms_rec_final.nii.gz'
input['multispecies'] = True
# == define path to output dir
input['output_base_path'] =  "/scratch1/07544/ghafouri/results/syndata/case8_ic_inv_s3/"

# == define lambda for observation operator
input['obs_lambda'] = 1
# == define sparsity per component
input['sparsity_per_comp'] = 3  ## change if you want (3 also might be fine)
# == define segmentation labels
input['segmentation_labels'] = "0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm"
input['submit'] = True
# =======================================================================

print('running for single patient')
gridcont.sparsetil_gridcont(input, input['output_base_path'], 0, use_gpu = True);
