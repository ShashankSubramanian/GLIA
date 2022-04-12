import os, sys
import params as par
from gridcont import run_sparsetil_multilevel as gridcont

input = {}


# gpu/single precision
extra_modules = "\nmodule load cuda"
extra_modules += "\nmodule load cudnn"
extra_modules += "\nmodule load nccl"
extra_modules += "\nmodule load petsc/3.11-rtx"
extra_modules += "\nexport ACCFFT_DIR=/work/04678/scheufks/frontera/libs/accfft/build_gpu/"
extra_modules += "\nexport ACCFFT_LIB=${ACCFFT_DIR}/lib/"
extra_modules += "\nexport ACCFFT_INC=${ACCFFT_DIR}/include/"
extra_modules += "\nexport CUDA_DIR=${TACC_CUDA_DIR}/"

# double precision
#extra_modules =  "module load petsc/3.11"
#extra_modules =  "module load petsc/3.11-single"


## SETTTINGS ##
# =======================================================================
# == specify any extra modules to be loaded, one string, may contain newline
input['extra_modules'] = extra_modules
# == compute system
input['system'] = 'frontera'
input['queue'] = 'rtx-dev'

# == define path to patient segmentation
#input['patient_path'] = '/scratch1/04678/scheufks/test/case_035_S_4114/data/time_point_0_seg.nii.gz'
input['patient_path'] = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/scripts/../syndata/160/seg_t1.nc'
input['data_path'] = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/scripts/../syndata/160/ed_inv/data_t1.nc'

# == define path to output dir
input['output_base_path'] = ''

# == define lambda for observation operator
input['obs_lambda'] = 1
# == define segmentation labels
input['segmentation_labels'] = "0=bg,1=vt,2=gm,3=wm"
# observation threshold
input['obs_threshold_1'] = 0.0
input['obs_threshold_rel'] = 1
input['thresh_component_weight'] = 1E-2

input['rho_init'] = 4

input['k_gm_wm'] = 1./5
input['r_gm_wm'] = 0

input['out_dir_suffix'] = 'obs-'+str(input['obs_threshold_1']) + '_no-obs-on-c'

input['submit'] = False
# =======================================================================

print('running for single patient')
gridcont.sparsetil_gridcont(input);
