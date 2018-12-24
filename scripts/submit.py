### This script creates a slurm job script for the inverse
### tumor solver

from TumorParams import *
import subprocess

scripts_path = os.path.dirname(os.path.realpath(__file__))

tumor_dir = scripts_path + '/../'
params = {}
params['code_path'] = tumor_dir
params['results_path'] = tumor_dir + '/results/cosamp_rho9.5_grad/'
# params['data_path'] = tumor_dir + '/results/check/data.nc'

### Real data
# params['data_path'] = tumor_dir + '/../SIBIA/base/brain_data/symlinks/glistr_data/AAMH/patient-AAMH_128_tu.nc'
# ## Atlas
# params['gm_path'] = tumor_dir + '/../SIBIA/base/brain_data/symlinks/glistr_data/AAMH/atlas-jakob_128_gm.nc'
# params['wm_path'] = tumor_dir + '/../SIBIA/base/brain_data/symlinks/glistr_data/AAMH/atlas-jakob_128_wm.nc'
# params['csf_path'] = tumor_dir + '/../SIBIA/base/brain_data/symlinks/glistr_data/AAMH/atlas-jakob_128_csf.nc'
	



run_str, err = getTumorRunCmd (params)  ### Use default parameters (if not, define dict with usable values)

if not err:  # No error in tumor input parameters
	print('No errors, submitting jobfile\n')
	fname = scripts_path + '/job.sh'
	submit_file = open(fname, 'w+')
	submit_file.write ("#!/bin/bash\n" + \
	"#SBATCH -J ITP\n" + \
	"#SBATCH -o " + params['results_path'] + "/log\n" + \
	"#SBATCH -p rebels\n" + \
	"#SBATCH -N 1\n" + \
	"#SBATCH -n 20\n" + \
	"#SBATCH -t 48:00:00\n" + \
	"source ~/.bashrc\n" + \
	"export OMP_NUM_THREADS=1\n")
	submit_file.write(run_str)

	submit_file.close()

	### submit jobfile
	subprocess.call(['sbatch', fname])
else:
	print('Errors, no job submitted\n')
