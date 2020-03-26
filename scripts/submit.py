### This script creates a slurm job script for the inverse
### tumor solver

from TumorParams import *
import subprocess

scripts_path = os.path.dirname(os.path.realpath(__file__))

tumor_dir = scripts_path + '/../'
params = {}
params['code_path'] = tumor_dir
params['results_path'] = tumor_dir + '/results/l0-tc2/'
params['compute_sys'] = 'frontera'
#### tumor data
#params['data_path'] = tumor_dir + '/results/Image12-case6/c_final.nc'
#### atlas
#params['gm_path'] = tumor_dir + "/brain_data/Image12/256/Image12_gm.nc" 
#params['wm_path'] = tumor_dir + "/brain_data/Image12/256/Image12_wm.nc" 
#params['glm_path'] = tumor_dir + "/brain_data/Image12/256/Image12_csf.nc" 
#params['csf_path'] = tumor_dir + "/brain_data/Image12/256/Image12_vt.nc"
#
##params['gm_path'] = tumor_dir + "/results/reg-Image12-case6-mask/atlas-6/atlas-6_gm.nc"
##params['wm_path'] = tumor_dir + "/results/reg-Image12-case6-mask/atlas-6/atlas-6_wm.nc"
##params['glm_path'] = tumor_dir + "/results/reg-Image12-case6-mask/atlas-6/atlas-6_csf.nc"
##params['csf_path'] = tumor_dir + "/results/reg-Image12-case6-mask/atlas-6/atlas-6_vt.nc"
#
#### input p and phi-cm (needed for mass-effect inversion)
##params['gaussian_cm_path'] = tumor_dir + "/results/Image12-case6/phi-mesh-forward.txt"
##params['pvec_path'] = tumor_dir + "/results/Image12-case6/p-rec-forward.txt"
#params['gaussian_cm_path'] = tumor_dir + "/results/rd-inv-Image12-case6/tumor_inversion/nx256/obs-1.0/phi-mesh-scaled.txt"
#params['pvec_path'] = tumor_dir + "/results/rd-inv-Image12-case6/tumor_inversion/nx256/obs-1.0/p-rec-scaled.txt"
##params['gaussian_cm_path'] = tumor_dir + "/results/reg-Image12-case6-mask/atlas-6/phi-mesh-scaled-transported.txt"
##params['pvec_path'] = tumor_dir + "/results/reg-Image12-case6-mask/atlas-6/p-rec-scaled-transported.txt"
##params['init_tumor_path'] = tumor_dir + "/results/reg-Image12-case6-mask/atlas-6/c0Recon_transported.nc"
#### data to patient
#params['p_gm_path'] = tumor_dir + "/results/Image12-case6/gm_final.nc"
#params['p_wm_path'] = tumor_dir + "/results/Image12-case6/wm_final.nc"
#params['p_csf_path'] = tumor_dir + "/results/Image12-case6/vt_final.nc"
#params['p_glm_path'] = tumor_dir + "/results/Image12-case6/csf_final.nc"
#### data to MRI of atlas ~ if needed
#params['mri_path'] = tumor_dir + "/brain_data/Image12/t1.nc"

if params['compute_sys'] == 'rebels':
    queue = 'rebels'
    N = 1
    n = 20
elif params['compute_sys'] == 'stampede2':
    queue = 'skx-normal'
    N = 3
    n = 64
elif params['compute_sys'] == 'frontera':
    queue = 'v100'
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
        "#SBATCH -t 24:00:00\n" + \
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
