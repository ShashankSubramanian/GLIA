### This script creates a slurm job script for the inverse
### tumor solver

from TumorParams import *
import subprocess

scripts_path = os.path.dirname(os.path.realpath(__file__))

tumor_dir = scripts_path + '/../'
params = {}
params['code_path'] = tumor_dir
params['results_path'] = tumor_dir + '/results/check_128/'
params['compute_sys'] = 'rebels'


# ### Real data
# params['data_path'] = tumor_dir + '/results/tc2_128/data.nc'
# params['data_path'] = '/workspace/shashank/label_maps/tcia_09_141/data.nc'
# params['data_path'] = '/workspace/shashank/tumor_tools/axo/input/patient_seg_tc.nc'
# # # ## Atlas
# params['gm_path'] = '/workspace/shashank/tumor_tools/axo/input/patient_seg_gm.nc'
# params['wm_path'] = '/workspace/shashank/tumor_tools/axo/input/patient_seg_wm_wt.nc'
# params['csf_path'] = '/workspace/shashank/tumor_tools/axo/input/patient_seg_csf.nc'
# params['gm_path'] = '/workspace/shashank/label_maps/tcia_09_141/gm.nc'
# params['wm_path'] = '/workspace/shashank/label_maps/tcia_09_141/wm.nc'
# params['csf_path'] = '/workspace/shashank/label_maps/tcia_09_141/csf.nc'
	



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
	"#SBATCH -t 100:00:00\n" + \
	"source ~/.bashrc\n" + \
	"export OMP_NUM_THREADS=1\n")
	submit_file.write(run_str)

	submit_file.close()

	### submit jobfile
	subprocess.call(['sbatch', fname])
else:
	print('Errors, no job submitted\n')
