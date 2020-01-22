### This script creates a slurm job script for the inverse
### tumor solver

from TumorParams import *
import subprocess

scripts_path = os.path.dirname(os.path.realpath(__file__))

tumor_dir = scripts_path + '/../'
params = {}
params['code_path'] = tumor_dir
params['results_path'] = tumor_dir + '/results/check-gpu/'
params['compute_sys'] = 'frontera'



# ### Real data
params['data_path'] = tumor_dir + '/results/masseffect-f12/c_t[50].nc'
# # # ## Atlas
params['gm_path'] = tumor_dir + "/brain_data/jakob/256/jakob_gm.nc" 
params['wm_path'] = tumor_dir + "/brain_data/jakob/256/jakob_wm.nc" 
params['glm_path'] = tumor_dir + "/brain_data/jakob/256/jakob_csf.nc" 
params['csf_path'] = tumor_dir + "/brain_data/jakob/256/jakob_vt.nc"

### input p and phi-cm (needed for mass-effect inversion)
params['gaussian_cm_path'] = tumor_dir + "/results/masseffect-f12/phi-mesh-forward.txt"
params['pvec_path'] = tumor_dir + "/results/masseffect-f12/p-rec-forward.txt"


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
    submit_file.write(run_str)
    submit_file.close()
    ### submit jobfile
    if params['compute_sys'] == 'hazelhen':
        subprocess.call(['qsub', fname])
    else:
        subprocess.call(['sbatch', fname])
else:
    print('Errors, no job submitted\n')
