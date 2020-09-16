### This script creates a slurm job script for the inverse
### tumor solver

from TumorParams import *
import subprocess

scripts_path = os.path.dirname(os.path.realpath(__file__))

tumor_dir = scripts_path + '/../'
params = {}
<<<<<<< HEAD:scripts/submit.py


base_dir = '/scratch1/04678/scheufks/alzh/syn_test/_NEW_synthetic_adv/'
case_dir = os.path.join(base_dir, 'tc/t_series/')
d_dir = os.path.join(base_dir, 'data/')

params['code_path'] = tumor_dir
params['results_path'] = case_dir; #tumor_dir + '/results/dev-tc4-l0-lb0-50vecs/'
params['compute_sys'] = 'frontera'


# ### Real data
# params['data_path'] = tumor_dir + '/results/tc2_128/data.nc'
# # # ## Atlas
#params['gm_path'] = tumor_dir + "/brain_data/jakob/256/jakob_gm.nc" 
#params['wm_path'] = tumor_dir + "/brain_data/jakob/256/jakob_wm.nc" 
#params['glm_path'] = tumor_dir + "/brain_data/jakob/256/jakob_csf.nc" 
#params['csf_path'] = tumor_dir + "/brain_data/jakob/256/jakob_vt.nc"
params['gm_path'] = d_dir + "0368Y01_seg_gm.nc"
params['wm_path'] = d_dir + "0368Y01_seg_wm.nc"
params['glm_path'] = ""
params['csf_path'] = d_dir + "0368Y01_seg_csf.nc"

vx1 = 'reg/velocity-field-x1.nc'
vx2 = 'reg/velocity-field-x2.nc'
vx3 = 'reg/velocity-field-x3.nc'

params['velocity_x1'] = os.path.join(base_dir, vx1);
params['velocity_x2'] = os.path.join(base_dir, vx2);
params['velocity_x3'] = os.path.join(base_dir, vx3);
 


params['data_path'] = ""
#params['init_tumor_path'] = case_dir + "c0Recon.nc"
=======
params['code_path'] = tumor_dir
params['results_path'] = tumor_dir + '/results/check/'
params['compute_sys'] = 'frontera'
#### tumor data
#params['data_path'] = tumor_dir + '/results/t16-case6/c_final.nc'
#### atlas
#params['gm_path'] = tumor_dir + "/brain_data/t16/256/t16_gm.nc" 
#params['wm_path'] = tumor_dir + "/brain_data/t16/256/t16_wm.nc" 
#params['glm_path'] = tumor_dir + "/brain_data/t16/256/t16_csf.nc" 
#params['csf_path'] = tumor_dir + "/brain_data/t16/256/t16_vt.nc"
#
##params['gm_path'] = tumor_dir + "/results/reg-t16-case6-mask/atlas-6/atlas-6_gm.nc"
##params['wm_path'] = tumor_dir + "/results/reg-t16-case6-mask/atlas-6/atlas-6_wm.nc"
##params['glm_path'] = tumor_dir + "/results/reg-t16-case6-mask/atlas-6/atlas-6_csf.nc"
##params['csf_path'] = tumor_dir + "/results/reg-t16-case6-mask/atlas-6/atlas-6_vt.nc"
#
#### input p and phi-cm (needed for mass-effect inversion)
##params['gaussian_cm_path'] = tumor_dir + "/results/t16-case6/phi-mesh-forward.txt"
##params['pvec_path'] = tumor_dir + "/results/t16-case6/p-rec-forward.txt"
#params['gaussian_cm_path'] = tumor_dir + "/results/rd-inv-t16-case6/tumor_inversion/nx256/obs-1.0/phi-mesh-scaled.txt"
#params['pvec_path'] = tumor_dir + "/results/rd-inv-t16-case6/tumor_inversion/nx256/obs-1.0/p-rec-scaled.txt"
##params['gaussian_cm_path'] = tumor_dir + "/results/reg-t16-case6-mask/atlas-6/phi-mesh-scaled-transported.txt"
##params['pvec_path'] = tumor_dir + "/results/reg-t16-case6-mask/atlas-6/p-rec-scaled-transported.txt"
##params['init_tumor_path'] = tumor_dir + "/results/reg-t16-case6-mask/atlas-6/c0Recon_transported.nc"
#### data to patient
#params['p_gm_path'] = tumor_dir + "/results/t16-case6/gm_final.nc"
#params['p_wm_path'] = tumor_dir + "/results/t16-case6/wm_final.nc"
#params['p_csf_path'] = tumor_dir + "/results/t16-case6/vt_final.nc"
#params['p_glm_path'] = tumor_dir + "/results/t16-case6/csf_final.nc"
#### data to MRI of atlas ~ if needed
#params['mri_path'] = tumor_dir + "/brain_data/t16/t1.nc"
>>>>>>> dev_alzh-ali:scripts/old/submit.py

if params['compute_sys'] == 'rebels':
    queue = 'rebels'
    N = 1
    n = 20
elif params['compute_sys'] == 'stampede2':
    queue = 'skx-normal'
    N = 3
    n = 64
elif params['compute_sys'] == 'frontera':
<<<<<<< HEAD:scripts/submit.py
    queue = 'rtx'
=======
    queue = 'v100'
>>>>>>> dev_alzh-ali:scripts/old/submit.py
    N = 1
    n = 1
elif params['compute_sys'] == 'maverick2':
    queue = 'gtx'
    N = 1
    n = 1
elif params['compute_sys'] == 'hazelhen':
    N = 3
    n = 64
    params['mpi_pernode'] = 64
else:
    queue = 'normal'
    N = 1
    n = 1

run_str, err = getTumorRunCmd (params)  ### Use default parameters (if not, define dict with usable values)

if not err:  # No error in tumor input parameters
    print('No errors, submitting jobfile\n')
    fname = scripts_path + '/job.sh'
    submit_file = open(fname, 'w+')
    if params['compute_sys'] == 'hazelhen':
        submit_file.write("#!/bin/bash\n" + \
        "#PBS -N ITP\n" + \
        "#PBS -l nodes="+str(N)+":ppn=24 \n" + \
        "#PBS -l walltime=01:00:00 \n" + \
        "#PBS -m e\n" + \
        "#PBS -M kscheufele@austin.utexas.edu\n\n" + \
        "source /zhome/academic/HLRS/ipv/ipvscheu/env_intel.sh\n" + \
        "export OMP_NUM_THREADS=1\n")
    else:
        submit_file.write ("#!/bin/bash\n" + \
        "#SBATCH -J ITP\n" + \
        "#SBATCH -o " + params['results_path'] + "/log\n" + \
        "#SBATCH -p " + queue + "\n" + \
        "#SBATCH -N " + str(N) + "\n" + \
        "#SBATCH -n " + str(n) + "\n" + \
        "#SBATCH -t 06:00:00\n" + \
        "source ~/.bashrc\n" + \
        "export OMP_NUM_THREADS=1\n")
        submit_file.write("\nmodule load cuda")
        submit_file.write("\nmodule load cudnn")
        submit_file.write("\nmodule load nccl")
        submit_file.write("\nmodule load petsc/3.11-rtx")
        submit_file.write("\nexport ACCFFT_DIR=/work/04678/scheufks/frontera/libs/accfft/build_gpu/")
        submit_file.write("\nexport ACCFFT_LIB=${ACCFFT_DIR}/lib/")
        submit_file.write("\nexport ACCFFT_INC=${ACCFFT_DIR}/include/")
        submit_file.write("\nexport CUDA_DIR=${TACC_CUDA_DIR}/\n")


    submit_file.write(run_str)
    submit_file.close()
    ### submit jobfile
    if params['compute_sys'] == 'hazelhen':
        subprocess.call(['qsub', fname])
    else:
        subprocess.call(['sbatch', fname])
else:
    print('Errors, no job submitted\n')
