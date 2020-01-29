### This script creates a slurm job script for the inverse
### tumor solver

from TumorParams import *
import subprocess

scripts_path = os.path.dirname(os.path.realpath(__file__))

tumor_dir = scripts_path + '/../'
params = {}
params['code_path'] = tumor_dir
params['results_path'] = tumor_dir + '/results/inv-128/'
params['compute_sys'] = 'rebels'
#### tumor data
params['data_path'] = tumor_dir + '/results/fwd-128/c_final.nc'
### atlas
params['gm_path'] = tumor_dir + "/brain_data/jakob/128/jakob_gm.nc" 
params['wm_path'] = tumor_dir + "/brain_data/jakob/128/jakob_wm.nc" 
params['glm_path'] = tumor_dir + "/brain_data/jakob/128/jakob_csf.nc" 
params['csf_path'] = tumor_dir + "/brain_data/jakob/128/jakob_vt.nc"
### input p and phi-cm (needed for mass-effect inversion)
params['gaussian_cm_path'] = tumor_dir + "/results/fwd-128/phi-mesh-forward.txt"
params['pvec_path'] = tumor_dir + "/results/fwd-128/p-rec-forward.txt"
### data to healthy patient
params['p_gm_path'] = tumor_dir + "/results/fwd-128/gm_final.nc"
params['p_wm_path'] = tumor_dir + "/results/fwd-128/wm_final.nc"
params['p_csf_path'] = tumor_dir + "/results/fwd-128/vt_final.nc"
params['p_glm_path'] = tumor_dir + "/results/fwd-128/csf_final.nc"


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
        "#SBATCH -t 100:00:00\n" + \
        "source ~/.bashrc\n" + \
        "export OMP_NUM_THREADS=1\n")
    submit_file.write(run_str)
    submit_file.close()
    ### submit jobfile
    if params['compute_sys'] == 'hazelhen':
        subprocess.call(['qsub', fname])
    else:
        subprocess.call(['sbatch', fname])
else:
    print('Errors, no job submitted\n')
