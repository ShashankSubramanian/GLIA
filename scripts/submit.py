### This script creates a slurm job script for the inverse
### tumor solver

from TumorParams import *
import subprocess

scripts_path = os.path.dirname(os.path.realpath(__file__))

tumor_dir = scripts_path + '/../'
params = {}


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

if params['compute_sys'] == 'rebels':
    queue = 'rebels'
    N = 1
    n = 20
elif params['compute_sys'] == 'stampede2':
    queue = 'skx-normal'
    N = 3
    n = 64
elif params['compute_sys'] == 'frontera':
    queue = 'rtx'
    N = 1
    n = 1
elif params['compute_sys'] == 'maverick2':
    queue = 'p100'
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
        "#SBATCH -t 01:00:00\n" + \
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
