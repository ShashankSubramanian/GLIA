import os, sys
import params as par
import subprocess
import argparse
from shutil import copyfile

import nibabel as nib
import numpy as np
import glob

import .file_io as fio
import .image_tools as imgtools
import .utils as utils

cases_per_jobfile_counter = 0;
JOBfile = "";


def sparsetil_gridcont(input):
    """ Creates tumor solver input files, config files, and submits grid continuation
        jobs for sparse TIL tumor inversion.
    """
    # ########### SETTINGS ########### #
    # -------------------------------- #
    patients_per_job   = 1;            # specify if multiple cases should be combined in single job script
    submit             = True;
    # -------------------------------- #
    nodes              = 2;
    procs              = 96;
    wtime_h            = [x * patients_per_job for x in [0,2,12]];
    wtime_m            = [x * patients_per_job for x in [30,0,0]];
    # -------------------------------- #
    split_segmentation = False         # pass segmentation directly as input, or split up in tissue labels
    levels             = [64,128,256]  # coarsening levels
    invert_diffusivity = [1,1,1];      # enable kappa inversion per level
    inject_coarse_sol  = True;
    pre_reacdiff_solve = True;         # performs a reaction/diffusion solve before sparse TIL solve
    sparsity_per_comp  = 5;            # allowed sparsity per tumor component
    predict            = [0,0,0]       # enable predict flag on level
    # -------------------------------- #
    rho_init           = 8;            # initial guess rho on coarsest level
    k_init             = 0;            # initial guess kappa on coarsest level
    beta_p             = 0;            # L2 regularization weight
    opttol             = 1E-4;         # gradient tolerance for the optimizer
    ls_max_func_evals  = [10, 10, 10]; # max number of allowed ls attempts per level
    lbound_kappa       = [1E-4, 1E-4, 1E-4];
    ubound_kappa       = [1., 1., 1.]; # bounds per level
    # -------------------------------- #
    gaussian_mode      = "PHI";        # alternatives: {"D", "PHI", "C0", "C0_RANKED"}
    data_thresh        = [1E-1, 1E-4, 1E-4] if (gaussian_selection_mode in ["PHI","C0"]) else [1E-1, 1E-1, 1E-1];
    sigma_fac          = [1,1,1]       # on every level, sigma = fac * hx
    gvf                = [0.0,0.9,0.9] # Gaussian volume fraction for Phi selection; ignored for C0_RANKED
    # -------------------------------- #
    # ################################ #

    ###############
    r = {}
    p = {}

    # handle submit if multiple cases per job file
    global cases_per_jobfile_counter;
    cases_per_jobfile_counter = cases_per_jobfile_counter + 1 if cases_per_jobfile_counter < patients_per_job else 1;
    batch_end = cases_per_jobfile_counter == patients_per_job
    submit    = submit and batch_end;
    new_job   = cases_per_jobfile_counter == 1
    global JOBfile;
    JOBfile = "" if new_job else JOBfile + "\n\n###############################################################\n###############################################################\n###############################################################\n\n\n";

    out_dir = input['out_dir']
    patient_dir = input['patient_path']
    patient_labels = input['segmentation_labels']
    if 'obs_lambda' in input:
        obs_lambda = input['obs_lambda']
    else:
        obs_lambda = 1.0

    input_folder = os.path.join(output_path, 'input');
    if not os.path.exists(input_folder):
        os.mkdir(input_folder);


    # == load and resample input segmentation
    if ".nii.gz" in patient_dir or ".nc" in patient_dir:
        ext = "." + patient_dir.split(".")[-1]
        fname = patient_dir
    else:
        segmentation_files = glob.glob(os.path.join(patient_dir, '*seg*'))
        if len(segmentation_files) > 1:
            print(" Warning: Multiple segmentation files found in patient directory: \n {} \n .. please specify filename (first one is selected for now).".format(segmentation_files))
        ext = "." + segmentation_files[0].split(".")[-1]
        fname = os.path.join(patient_path, segmentation_files[0])
    filename = fname.split('/')[-1].split('.')[0]

    header = affine256 = None
    order = 0 # nearest neighbor interpolation
    if ext in ['.nii.gz', '.nii']:
        p_seg = nib.load(fname)
        p_seg_d = p_seg.get_fdata()
        header = p_seg.header
        affine256 = p_seg.affine
        # 256 --> 128
        affine64 = np.copy(affine256)
        row,col = np.diag_indices(affine64.shape[0])
        affine64[row,col] = np.array([-4,-4,4,1]);
        p_seg128 = nib.processing.resample_from_to(p_seg, (np.multiply(0.5, p_seg.shape).astype(int), affine128), order=order)
        nib.save(p_seg128, os.path.join(input_folder, filename + '_nx128' + ext));
        p_seg_d128 = p_seg128.get_fdata()
        # 256 --> 64
        affine128 = np.copy(affine256)
        row,col = np.diag_indices(affine128.shape[0])
        affine128[row,col] = np.array([-2,-2,2,1]);
        p_seg64 = nib.processing.resample_from_to(p_seg, (np.multiply(0.25, p_seg.shape).astype(int), affine64), order=order)
        nib.save(p_seg64, os.path.join(input_folder, filename + '_nx64' + ext));
        p_seg_d64 = p_seg64.get_fdata()
    else:
        p_seg_d = readNetCDF(fname)
        p_seg_d64  = imgtools.resizeImage(p_seg_d, tuple([64,64,64]), interp_order=order)
        p_seg_d128 = imgtools.resizeImage(p_seg_d, tuple([128,128,128]), interp_order=order)


    if split_segmentation:
        # TODO:
        # 1. split segmentation into wm, gm, csf, tc, ed on each level; write label maps to file
        # 2. create observation mask for corresponding lambda on each level
        # 3. create symlinks to label maps and observation mask (pointing from level/init/ folder to input/)
        # 4. set all p['..'] paths to label maps, and obs mask

        raise NotImplementedError;
        pass;
    else:
        # tumor code reads in segmentation on each level, and splits it into label maps.
        # tumor code implicitly replaces tc area with wm, and combines enh + nec into tc
        # tumor code constructs observation mask as OBS = 1[TC] + lambda * 1[B\WT] based on lambda
        p['']






    # a_seg_path=testdata/tcia03_257_seg_nx64.nc
    # atlas_labels=[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]


submit_job = False;

###############
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../'

############### === define parameters
p['output_dir'] 		= os.path.join(code_dir, 'results/inverse/');   		# results path
p['d1_path'] 			= code_dir + '/brain_data/256/cpl/c1p.nc'			# tumor data path
p['a_wm_path'] 			= code_dir + '/brain_data/256/white_matter.nc'	# healthy patient path
p['a_gm_path'] 			= code_dir + '/brain_data/256/gray_matter.nc'
p['a_csf_path'] 		= ""
p['a_vt_path'] 			= code_dir + '/brain_data/256/csf.nc'
p['p_wm_path'] 			= ""													# patient brain data path
p['p_gm_path'] 			= ""
p['p_csf_path'] 		= ""
p['p_vt_path'] 			= ""
p['solver'] 			= 'sparse_til'              # modes: sparse_til; nonsparse_til, reaction_diffusion, mass_effect, multi_species, forward, test
p['model'] 				= 1                       	# 1: reaction-diffuion; 2: alzh, 3: full objective, 4: mass-effect, 5: multi-species
p['verbosity'] 			= 1                  		# various levels of output density
p['syn_flag'] 			= 1                  	    # create synthetic data
p['user_cms'] 			= [(112,136,144,1)]   		# arbitrary number of TILs (x,y,z,scale) with activation scale
p['rho_data'] 			= 12                  		# tumor parameters for synthetic data
p['k_data'] 			= 0.05
p['nt_data'] 			= 25
p['dt_data']            = 0.04
p['init_rho'] 			= 15                   		# initial guess rho (reaction in wm)
p['init_k'] 			= 0                    		# initial guess kappa (diffusivity in wm)
p['init_gamma'] 		= 12E4              		# initial guess (forcing factor for mass effect)
p['nt_inv'] 			= 25                    	# number time steps for inversion
p['dt_inv'] 			= 0.04                  	# time step size for inversion
p['time_history_off'] 	= 0          				# 1: do not allocate time history (only works with forward solver or FD inversion)
p['sparsity_level'] 	= 10             			# target sparsity of recovered IC in sparse_til solver
p['beta_p'] 			= 0E-4                  	# regularization parameter
p['opttol_grad'] 		= 1E-5             			# relative gradient tolerance
p['newton_maxit'] 		= 50              			# number of iterations for optimizer
p['kappa_lb'] 			= 0                   		# lower bound kappa
p['kappa_ub'] 			= 0.1                 		# upper bound kappa
p['rho_lb'] 			= 0                     	# lower bound rho
p['rho_ub'] 			= 15                    	# upper bound rho
p['gamma_ub'] 			= 15                  		# upper bound gamma
p['lbfgs_vectors'] 		= 50				        # number of vectors for lbfgs update

############### === define run configuration
r['code_path'] 			= code_dir;
r['compute_sys'] 		= 'rebels'         			# TACC systems are: maverick2, frontera, stampede2, longhorn

###############=== write config to write_path and submit job
par.submit(p, r, submit_job);

### -tao_blmvm_mat_lmvm_num_vecs 50 -tao_blmvm_mat_lmvm_scale_type diagonal -tumor_tao_ls_max_funcs 10


if __name__=='__main__':
    # repository base directory
    basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
    # parse arguments
    parser = argparse.ArgumentParser(description='Process input images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r_args = parser.add_argument_group('required arguments')
    r_args.add_argument ('-ppath',   '--patient_path',   type = str, help = 'path to patient image directory containing patient segmentation and T1 MRI)', required=True)
    r_args.add_argument ('-plabels', '--patient_labels', type=str,   help = 'comma separated patient image segmentation labels. for ex.\n  0=bg,1=nec,2=ed,4=enh,5=wm,6=gm,7=vt,8=csf\n for BRATS type segmentation.');
    r_args.add_argument ('-x',       '--out_dir',        type = str, default = os.path.join(basedir, 'results/'), help = 'path to results directory');
    parser.add_argument ('-lambda'   '--obs_lambda',     type = float, default = 1,   help = 'parameter to control observation operator OBS = TC + lambda (1-WT)');
    parser.add_argument ('multi',    '--multiple_patients', action='store_true', help = 'process multiple patients, -patient_path should be the base directory containing patient folders which contain patient image(s).');
    args = parser.parse_args();

    input = {}
    input['patient_path'] = args.patient_path
    input['out_dir'] = args.out_dir
    input['segmentation_labels'] = args.patient_labels

    if args.obs_lambda:
        input['obs_lambda'] = args.obs_lambda

    if args.multiple_patients:
        print('running for multiple patients')
        # get the number of patients in the given path
        patient_list = next(os.walk(args.patient_path))[1];
        base_out_dir = args.out_dir;
        base_patient_path = args.patient_path
        for patient in patient_list:
            print('processing ', patient)
            input['patient_path'] = os.path.join(os.path.join(base_patient_path, patient), patient + '_seg_tu.nii.gz');
            input['out_dir'] = os.path.join(base_out_dir, patient);
            gridcont(input);
    else:
        print('running for single patient')
        gridcont(input);
