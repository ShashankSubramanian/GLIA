import os, sys
import params as par
from gridcont import run_sparsetil_multilevel as gridcont

input = {}


## SETTTINGS ##
# -----------------------------------
input['system'] = 'pele'
# define path to patient segmentation
input['patient_path'] = '/scratch/scheufele/test/gridcont-test/data/Brats18_TCIA03_257_1_seg_tu.nii.gz'
# define path to output dir
input['output_base_path'] = "/scratch/scheufele/test/gridcont-test"
# define lambda for observation operator
input['obs_lambda'] = 1
# define segmentation labels
input['segmentation_labels'] = "0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm"
input['submit'] = False
# -----------------------------------

print('running for single patient')
gridcont.sparsetil_gridcont(input);
