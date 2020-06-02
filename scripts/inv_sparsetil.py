import os, sys
import params as par
from .gridcont import sparsetil_gridcont

input = {}
###############
input['patient_path'] = ""     # define path to patient segmentation
input['output_base_path'] = "" # define path to output dir
input['obs_lambda'] = 1        # define lambda for observation operator
input['segmentation_labels'] = "0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm" # define segmentation labels
input['submit'] = False
###############

print('running for single patient')
sparsetil_gridcont(input);
