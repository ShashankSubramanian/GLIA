### Run parameters for inverse tumor solve
### This script defines a method which sets all tumor input parameters
### and returns the run command for the inverse tumor solve
### run.py runs the code with default parameters defined here if 
### not overwritten
import os

def getTumorRunCmd(params):  
	### params is a dictionary which can be populated with user values for compulsory tumor parameters
	### params can contain the following keys:
	### N, data_path, results_path, gm_path, wm_path, csf_path
	### Other parameters are default and changed only if necessary

	############################################################################################################################
	############################################################################################################################
	############################################################################################################################
	############################################################################################################################

	### tumor_dir is the path to pglistr
	if not 'code_path' in params:
		print('Path to pglistr not provided: Setting as current working directory\n')
		params['code_path'] = os.getcwd()
	tumor_dir = params['code_path']

	### TUMOR PARAMETERS SET BEGIN

	### No of discretization points (Assumed uniform)
	N = 128
	### Path to all output results (Directories are created automatically)
	results_path = tumor_dir + '/results/'
	if not os.path.exists(results_path):
		os.makedirs(results_path)
	### Input data
	### Path to data 
	data_path = tumor_dir + '/brain_data/' + str(N) +'/cpl/c1p.nc'
	### Atlas
	### Path to gm
	gm_path = tumor_dir + '/brain_data/' + str(N) +'/gray_matter.nc'
	### Path to csf
	csf_path = tumor_dir + '/brain_data/' + str(N) +'/csf.nc'
	### Path to wm
	wm_path = tumor_dir + '/brain_data/' + str(N) +'/white_matter.nc'

	### Other user parameters which typically stay as default: Change if needed
	### Flag to create synthetic data
	create_synthetic = 1
	### Inversion tumor parameters  -- Tumor is inverted with these parameters: Use k_inv=0 if diffusivity is being inverted
	rho_inv = 10
	k_inv = 0.01
	nt_inv = 15
	dt_inv = 0.02

	### tumor regularization type -- L1, L1c, L2, L2b  : L1c is cosamp
	reg_type = "L1c"
	### Model type: 1: RD, 2: RD + pos, 3: RD + full objective
	model = 1
	### Synthetic data parameters  -- Tumor is grown with these parameters
	rho_data = 10
	k_data = 0.01
	nt_data = 15
	dt_data = 0.02

	### Interpolation flag   -- Flag to solve an interpolation problem (find parameterization of the data) only
	interp_flag = 0
	### Diffusivity inversion flag  -- Flag to invert for diffusivity/diffusion coefficient
	diffusivity_flag = 1
	### Radial basis flag: 1 - data driven, 0 - grid-based (bounding box)  (Use data-driven for all tests)
	basis_type = 1
	### Lambda continuation flag -- Flag for parameter continuation in L1 optimization (Keep turned on)
	lam_cont = 1
	### Tumor L2 regularization
	beta = 1e-4
	### No of radial basis functions (Only used if basis_type is grid-based)
	np = 27
	### Factor (integer only) which controls the variance of the basis function for synthetic data (\sigma  =  fac * 2 * pi / 256)
	fac = 2
	### Spacing factor between radial basis functions (Keep as 2 to have a well-conditioned matrix for the radial basis functions)
	space = 2
	### Gaussian volume fraction -- Fraction of Gaussian that has to be tumorous to switch on the basis function at any grid point
	gvf = 0.99
	### Threshold of data tumor concentration above which Gaussians are switched on
	data_thres = 0.1
	### Target sparsity we expect for our initial tumor condition
	target_spars = 0.99
	### Factor (integer only) which controls the variance of the basis function for tumor inversion (\sigma  =  fac * 2 * pi / 256)
	dd_fac = 2
	### Solver type: QN - Quasi newton, GN - Gauss newton
	solvertype = "QN"
	### Newton max iterations
	max_iter = 50
	### GIST max iterations (for L1 solver)
	max_gist_iter = 50
	### Krylov max iterations
	max_krylov_iter = 30

	### TUMOR PARAMETERS SET END

	############################################################################################################################
	############################################################################################################################
	############################################################################################################################
	############################################################################################################################


	### Error checking and run script string generation 
	error_flag = 0
	if 'N' in params:
		N = params['N']
	else:
		print ('Default N = {} used').format(N)

	## Error checking done by petsc/outside

	if 'results_path' in params:
		results_path = params['results_path']
		if not os.path.exists(results_path):
			print ('Results path does not exist, making the required folders and sub-folders...\n')
			os.makedirs(results_path)
	
	if 'data_path' in params:
		data_path = params['data_path']
		print('Tumor data path = {}').format(data_path)
	else:
		if not os.path.exists(data_path):
			if not create_synthetic:
				print('Default data path does not exist and no input path provided!\n')
				error_flag = 1
		else:
			print ('Default datapath = {} used').format(data_path)
			
	if 'gm_path' in params:
		gm_path = params['gm_path']
		print('Gray matter path = {}').format(gm_path)
	else:
		if not os.path.exists(gm_path):
			print('Default atlas gray matter path does not exist and no input path provided!\n')
			error_flag = 1
		else:
			print ('Default atlas gray matter path = {} used').format(gm_path)

	if 'wm_path' in params:
		wm_path = params['wm_path']
		print('White matter path = {}').format(wm_path)
	else:
		if not os.path.exists(wm_path):
			print('Default atlas white matter path does not exist and no input path provided!\n')
			error_flag = 1
		else:
			print ('Default atlas white matter path = {} used').format(wm_path)


	if 'csf_path' in params:
		csf_path = params['csf_path']
		print('CSF path = {}').format(csf_path)
	else:
		if not os.path.exists(csf_path):
			print('Default atlas csf path does not exist and no input path provided!\n')
			error_flag = 1
		else:
			print ('Default atlas csf path = {} used').format(csf_path)

	run_str = "mpirun " + tumor_dir + "/build/last/inverse -nx " + str(N) + " -ny " + str(N) + " -nz " + str(N) + " -beta " + str(beta) + \
	" -rho_inversion " + str(rho_inv) + " -k_inversion " + str(k_inv) + " -nt_inversion " + str(nt_inv) + " -dt_inversion " + str(dt_inv) + \
	" -rho_data " + str(rho_data) + " -k_data " + str(k_data) + " -nt_data " + str(nt_data) + " -dt_data " + str(dt_data) + \
	" -regularization " + reg_type + " -interpolation " + str(interp_flag)+ " -diffusivity_inversion " + str(diffusivity_flag) + \
	" -basis_type " + str(basis_type) + " -number_gaussians " + str(np) + " -sigma_factor " + str(fac) + " -sigma_spacing " + str(space) + \
	" -gaussian_volume_fraction " + str(gvf) +  \
	" -lambda_continuation " + str(lam_cont) +  \
	" -target_sparsity " + str(target_spars) +  \
	" -threshold_data_driven " + str(data_thres) +  \
	" -sigma_data_driven " + str(dd_fac) + \
	" -output_dir " + results_path + \
	" -newton_solver " + solvertype + \
	" -newton_maxit " + str(max_iter) + \
	" -gist_maxit " + str(max_gist_iter) + \
	" -krylov_maxit " + str(max_krylov_iter) + \
	" -syn_flag " + str(create_synthetic) + \
	" -data_path " + data_path + \
	" -gm_path " + gm_path + \
	" -wm_path " + wm_path + \
	" -csf_path " + csf_path + \
	" -model " + str(model) + \
	" -tao_lmm_vectors 50 -tao_lmm_scale_type broyden -tao_lmm_scalar_history 5 -tao_lmm_rescale_type scalar -tao_lmm_rescale_history 5"

	return run_str, error_flag



##########################################################################################################################################################################################

### Parsers --- old version, too messy
### Command line parameters which superscripts will have to compulsarily provide
# parser = argparse.ArgumentParser (description = 'Glioblastoma inversion parameters')
# parser.add_argument ('-N', type = int, default = 256, metavar = 'N', help = 'Dimensions of uniform grid')
# parser.add_argument ('-data_path', type = str, default = os.getcwd() + '/brain_data/256/cpl/c1p.nc', metavar = 'data_path', help = 'Path to tumor data probability')
# parser.add_argument ('-atlas_gm_path', type = str, default = os.getcwd() + '/brain_data/256/gray_matter.nc', metavar = 'gm_path', help = 'Path to atlas gray matter probability')
# parser.add_argument ('-atlas_wm_path', type = str, default = os.getcwd() + '/brain_data/256/white_matter.nc', metavar = 'wm_path', help = 'Path to atlas white matter probability')
# parser.add_argument ('-atlas_csf_path', type = str, default = os.getcwd() + '/brain_data/256/csf.nc', metavar = 'csf_path', help = 'Path to atlas csf probability')
# parser.add_argument ('-atlas_glm_path', type = str, default = os.getcwd() + '/brain_data/256/glial_matter.nc', metavar = 'glm_path', help = 'Path to atlas glial matter probability')
# parser.add_argument ('-results_path', type = str, default = os.getcwd() + '/results/', metavar = 'results', help = 'Path to output files')

# args = parser.parse_arg2s()