import os, sys
import params as par
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/utils/')
from gridcont import run_sparsetil_multilevel as gridcont

input = {}
## SETTTINGS ##
# =======================================================================
# == specify any extra modules to be loaded, one string, may contain newline
# bashrc is sourced prior to this command.
#input['extra_modules'] = "module load petsc/3.11"

# == compute system
input['system'] = 'longhorn'

# == define path to patient segmentation
input['patient_path'] = '/scratch/05027/shas1693/tmi-results/Brats18_CBICA_ABO_1/data/Brats18_CBICA_ABO_1_seg_tu.nii.gz'

# == define path to output dir
input['output_base_path'] = '/scratch/05027/shas1693/pglistr_tumor/results/gridcont-test/Brats18_CBICA_ABO_1/'

# == define lambda for observation operator
input['obs_lambda'] = 1
# == define segmentation labels
input['segmentation_labels'] = "0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm"
input['submit'] = False
# =======================================================================

print('running for single patient')
gridcont.sparsetil_gridcont(input, use_gpu = True);
