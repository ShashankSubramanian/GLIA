import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir), os.path.pardir))

#from scripts import params as par
import params as par
import subprocess
import argparse
from shutil import copyfile

import nibabel as nib
import numpy as np
import glob

from utils import file_io as fio
from utils import image_tools as imgtools
# from ..utils import utils_gridcont as utils

cases_per_jobfile_counter = 0;
JOBfile = "";


### ________________________________________________________________________ ___
### //////////////////////////////////////////////////////////////////////// ###
def sparsetil_gridcont(input, job_path, job_idx, use_gpu = False):
    """ Creates tumor solver input files, config files, and submits grid continuation
        jobs for sparse TIL tumor inversion.
    """
    # ########### SETTINGS ########### #
    # -------------------------------- #
    if 'patients_per_job' in input:
      patients_per_job = input['patients_per_job']
    else:
      patients_per_job = 1;            # specify if multiple cases should be combined in single job script
    submit             = input['submit'] if 'submit' in input else False
    dtype              = '.nc'
    # -------------------------------- #
    system             = input['system'] if 'system' in input else 'frontera' # TACC systems are: stampede2, frontera, maverick2, pele, longhorn
    if use_gpu:
      nodes            = 1
      procs            = [1, 1, 1]
      wtime_h          = [x * patients_per_job for x in [0,1,3]];
      wtime_m          = [x * patients_per_job for x in [30,0,0]];
    else:
      nodes            = 2;
      procs            = [24, 48, 96];
      wtime_h          = [x * patients_per_job for x in [0,2,12]];
      wtime_m          = [x * patients_per_job for x in [30,0,0]];
    # -------------------------------- #
    split_segmentation = False         # pass segmentation directly as input, or split up in tissue labels
    levels             = [64,128,256]  # coarsening levels
    invert_diffusivity = [1,1,1];      # enable kappa inversion per level
    inject_coarse_sol  = True;
    pre_reacdiff_solve = True;         # performs a reaction/diffusion solve before sparse TIL solve
    sparsity_per_comp  = 5;            # allowed sparsity per tumor component
    predict            = [0,0,1]       # enable predict flag on level
    # -------------------------------- #
    rho_init           = 8;            # initial guess rho on coarsest level
    k_init             = 0;            # initial guess kappa on coarsest level
    r_gm_wm            = 0             # ratio of reaction coeffient in gm to wm
    k_gm_wm            = 0             # ratio of diffusion coeffient in gm to wm
    nt                 = 40            # number of time steps
    beta_p             = 0;            # L2 regularization weight
    opttol             = 1E-4;         # gradient tolerance for the optimizer
    newton_maxit       = [50, 50, 30]  # max number of newton its per level
    gist_maxit         = [4, 2, 2]     # number of cosamp iterations per level
    ls_max_func_evals  = [10, 10, 10]; # max number of allowed ls attempts per level
    ls_type            = ['mt', 'armijo', 'armijo'] # ls type per level
    kappa_lb           = [1E-4, 1E-4, 1E-4];
    kappa_ub           = [.5, .5, .5]; # kappa bounds per level
    rho_lb             = [1, 1, 1];    #
    rho_ub             = [18, 18, 18]; # rho bounds per level
    k_gm_wm            = 0.2             # kappa ratio wm to gm
    rho_gm_wm          = 0             # rho ratio wm to gm
    # -------------------------------- #
    gaussian_mode      = "PHI";        # alternatives: {"D", "PHI", "C0", "C0_RANKED"}
    data_thresh        = [1E-2, 1E-4, 1E-4] if (gaussian_mode in ["PHI","C0"]) else [1E-1, 1E-1, 1E-1];
    sigma_fac          = [1,1,1]       # on every level, sigma = fac * hx
    gvf                = [0.0,0.9,0.9] # Gaussian volume fraction for Phi selection; ignored for C0_RANKED
    # -------------------------------- #
    # ################################ #

    if 'rho_init' in input:
        rho_init = input['rho_init']
    if 'k_init' in input:
        k_init = input['k_init']
    if 'k_gm_wm' in input:
        k_gm_wm = input['k_gm_wm']
    if 'r_gm_wm' in input:
        r_gm_wm = input['r_gm_wm']
    if 'sparsity_per_comp' in input:
        sparsity_per_comp = input['sparsity_per_comp']

    r = {}
    scripts_path = os.path.dirname(os.path.realpath(__file__))
    utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'utils')
    ############### === define run configuration
    r['code_path']     = scripts_path + '/../../';
    r['compute_sys']   = system
    if 'extra_modules' in input:
      r['extra_modules'] = input['extra_modules']
    if 'queue' in input:
        r['queue'] = input['queue']

    cmd_command = ''
    symlink_cmd = ''
    pythoncmd = "python "

    # handle submit if multiple cases per job file
    global cases_per_jobfile_counter;
    cases_per_jobfile_counter = cases_per_jobfile_counter + 1 if cases_per_jobfile_counter < patients_per_job else 1;
    batch_end = cases_per_jobfile_counter == patients_per_job
    if 'batch_end' in input:
      if input['batch_end'] == True:
        batch_end = True
    submit    = submit and batch_end;
    new_job   = cases_per_jobfile_counter == 1
    global JOBfile;
    JOBfile = "" if new_job else JOBfile + "\n\n###############################################################\n###############################################################\n###############################################################\n\n\n";

    # labels
    labels = {}
    for x in input['segmentation_labels'].split(','):
        labels[int(x.split('=')[0])] = x.split('=')[1];
    labels_rev = {v:k for k,v in labels.items()};
    # obs
    if 'obs_lambda' in input:
        obs_lambda = input['obs_lambda']
    else:
        obs_lambda = 1.0
    obs_dir = "obs-{0:1.1f}".format(obs_lambda) + "/"
    # paths
    output_base_path = input['output_base_path']
    patient_data_path = input['patient_path']
    input_path = os.path.join(output_base_path, 'input');
    output_path_tumor = os.path.join(output_base_path, 'inversion');
    if 'out_dir_suffix' in input:
        output_path_tumor = output_path_tumor + '_' + input['out_dir_suffix']
    if not os.path.exists(input_path):
        os.makedirs(input_path);


    # == load and resample input segmentation
    if ".nii.gz" in patient_data_path or ".nc" in patient_data_path:
        ext = ".n" + patient_data_path.split(".n")[-1]
        fname = patient_data_path
    else:
        segmentation_files = glob.glob(os.path.join(patient_data_path, '*seg*'))
        if len(segmentation_files) > 1:
            print(" Warning: Multiple segmentation files found in patient directory: \n {} \n .. please specify filename (first one is selected for now).".format(segmentation_files))
        ext = ".n" + segmentation_files[0].split(".n")[-1]
        fname = os.path.join(patient_path, segmentation_files[0])
#    filename = fname.split('/')[-1].split('.')[0]
    filename = fname.split('/')[-1].replace(ext, "")
    cmd_command ="\n#===================================== PATIENT = " + filename + " ====================================================#"

    cp_cmd = ''
    reference_img = dataimg = None
    healthy_seg = False
    try:
       reference_img = nib.load(fname)
       if 'data_path' in input and input['data_path']:
           filename_data = os.path.basename(input['data_path']).split('.n')[0]
           dataimg = nib.load(input['data_path'])
           healthy_seg = True
    except Exception as e:
      print(e)
    if dtype == '.nc':
       ext = '.nc'
       suffix = '_nx256'
       tmp = reference_img.get_fdata()
       tmp_regular = imgtools.resizeImage(tmp, tuple([256, 256, 256]), interp_order=0)
       fio.createNetCDF(os.path.join(input_path, filename + suffix + ext), tmp_regular.shape, np.swapaxes(tmp_regular, 0, 2))
       if healthy_seg:
           data = dataimg.get_fdata()
           data_regular = imgtools.resizeImage(data, tuple([256, 256, 256]), interp_order=1)
           fio.createNetCDF(os.path.join(input_path, filename_data + suffix + ext), tmp_regular.shape, np.swapaxes(data_regular, 0, 2))

    else:
      suffix = "_nx{}x{}x{}".format(reference_img.shape[0], reference_img.shape[1], reference_img.shape[2])
      cp_cmd = "\n" + "cp -r " + fname + " " + os.path.join(input_path, filename + suffix + ext)
      if healthy_seg:
          cp_cmd = "\n" + "cp -r " + input['data_path'] + " " + os.path.join(input_path, filename_data + suffix + ext)

    # loop over levels and create config files
    for level, ii in zip(levels, range(len(levels))):
        p = {}

        resample_cmd  = "\n# resample data"
        resample_cmd += "\n" + pythoncmd + os.path.join(utils_path, 'utils_gridcont.py') + " -resample_input -input_path " + input_path +  " -fname " + filename + suffix + ext + " -ndim " + str(level)
        if healthy_seg:
            resample_cmd += "\n" + pythoncmd + utils_path + '/utils_gridcont.py -resample -input_path ' + input_path + ' -fname ' + filename_data + suffix  + ext + ' -ndim ' + str(level)
        # create dirs
        output_path_level = os.path.join(output_path_tumor, 'nx' + str(level) + "/");
        result_path_level = os.path.join(output_path_level, obs_dir)
        result_path_level_coarse = os.path.join(output_path_tumor, 'nx' + str(int(level/2)) + "/" + obs_dir);
        input_path_level = os.path.join(output_path_level, 'init');
        if not os.path.exists(result_path_level):
            os.makedirs(result_path_level, exist_ok=True);
        if not os.path.exists(input_path_level):
            os.makedirs(input_path_level);
        # create symlinks

        symlink_cmd = "\n\n### Symlinks ###\n#----------"
        symlink_cmd += "\n" + "cd " + str(input_path_level)
        # set paths for tumor config
        if split_segmentation:
            """ TODO:
                1. split segmentation into wm, gm, csf, tc, ed on each level; write label maps to file
                2. create observation mask for corresponding lambda on each level
                3. create symlinks to label maps and observation mask (pointing from level/init/ folder to input/)
                    - symlink_cmd += "ln -sf ../../../input/" + filename + "_nx"+str(level) + ext + " patient_seg" + ext
                4. set all p['..'] paths to label maps, and obs mask
            """
            # split segmentation via script: python3 utils ...
            # create obs mask via script: python3 utils ...

            # p['d1_path'] =
            # p['a_wm_path'] =
            # p['a_gm_path'] =
            # p['a_vt_path'] =
            # p['a_csf_path'] =
            # p['obs_mask_path'] = os.path.join(inp_dir, 'obs_mask_lbd-${LAMBDA_OBS}.nc') if (args.vary_obs_lambda) else os.path.join(inp_dir, 'obs_mask.nc');

            raise NotImplementedError("Splitting segmentation into label maps is not yet implemented.");
            pass;
        else:
            """ tumor code reads in segmentation on each level, and splits it into label maps.
                tumor code implicitly replaces tc area with wm, and combines enh + nec into tc
                tumor code constructs observation mask as OBS = 1[TC] + lambda * 1[B\WT] based on lambda
            """

            # symlink
            symlink_cmd += "\nln -sf ../../../input/" + filename + "_nx"+str(level) + ext + " patient_seg" + ext + " "
            if healthy_seg:
                symlink_cmd += "\nln -sf ../../../input/" + filename_data + "_nx"+str(level) + ext + " data" + ext + " "
            #  set labels: [wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]
            labelstr = ""
            if "wm" in labels_rev and "gm" in labels_rev:
                labelstr += "[wm="+str(labels_rev['wm'])+",gm="+str(labels_rev['gm'])
            if "vt" in labels_rev and "csf" in labels_rev:
                labelstr += ",vt="+str(labels_rev['vt'])+",csf="+str(labels_rev['csf'])
            elif "vt" in labels_rev:
                labelstr += ",vt="+str(labels_rev['vt'])
            elif "csf" in labels_rev:
                labelstr += ",vt="+str(labels_rev['csf'])
            if "ed" in labels_rev:
                labelstr += ",ed="+str(labels_rev['ed'])
            if "nec" in labels_rev:
                labelstr += ",nec="+str(labels_rev['nec'])
            if "en" in labels_rev:
                labelstr += ",en="+str(labels_rev['en'])
            if not labelstr:
                raise ValueError("Cannot interpret '{}' as valid label string. Please double check labels.".format(labelstr))
            labelstr += "]"
            p['atlas_labels'] = labelstr
            p['a_seg_path'] = os.path.join(input_path_level, 'patient_seg' + ext)
            if healthy_seg:
                #p['d1_path'] = os.path.join(input_path_level, 'data' + ext)
                p['d1_path'] = os.path.join(input_path_level, 'target_data' + ext)
            else:
                p['obs_lambda'] = obs_lambda
                p['d1_path'] = ""
            p['a_wm_path'] = p['a_gm_path'] = p['a_vt_path'] = ""

        symlink_cmd +=  "\nln -sf " + "../../../input/data_comps_nx" + str(level) + ext + " data_comps" + ext + " "
        symlink_cmd +=  "\nln -sf " + "../../../input/target_data_nx" + str(level) + ext + " target_data" + ext + " "
        if level == 64 or gaussian_mode == "D":
            symlink_cmd +=  "\nln -sf " + "../../../input/target_data_nx"+str(level)   + ext + " support_data" + ext + " "
        else:
            level_prev = int(level/2);
            resample_cmd += "\n" + pythoncmd + utils_path + '/utils_gridcont.py -resample -input_path ' + result_path_level_coarse + ' -fname ' + 'c0_rec'+ext + ' -ndim ' + str(level)
            symlink_cmd += "\n" + "ln -sf " + "../../nx"+ str(level_prev)+ "/" + obs_dir +"/c0_rec_nx"+str(level)+ ext + " support_data" + ext
            resample_cmd += "\n" + pythoncmd + utils_path + '/utils_gridcont.py -resample -input_path ' + result_path_level_coarse + ' -fname ' + 'phiSupportFinal' + ext + ' -ndim ' + str(level)
            symlink_cmd += "\n" + "ln -sf " + "../../nx"+ str(level_prev)+ "/" + obs_dir +"/phiSupportFinal_nx"+str(level)+ ext + " support_data_phi" + ext;
        symlink_cmd +=  "\n" + "cd " + str(result_path_level) + "\n#----------"

        # compute connected components of target data
        rdir = "obs"
        cmd_concomp = "\n\n# compute connected component of TC data"
        cmd_concomp += "\n" + pythoncmd + utils_path + '/utils_gridcont.py -concomp_data  -input_path ' +  result_path_level + ' -output_path ' +  input_path 
        if healthy_seg:
            cmd_concomp += ' -obs_th ' + str(input['obs_threshold_1']) + ' '
        else:
            cmd_concomp += ' -labels ' + input['segmentation_labels'] + ' '
        cmd_concomp += "  -sigma " + str(sigma_fac[ii]) + " ";
        
        if 'multispecies' in input and input['multispecies']:
          cmd_concomp += " -multispecies 1 ";
        
        cmd_concomp +=  " -select_gaussians  \n" if (gaussian_mode == 'C0_RANKED' and level > 64) else " \n";
        # extract reconstructed rho and k
        if level > 64:
            cmd_extractrhok =  "\n\n# extract reconstructed rho, k from logfile, and modify config";
            cmd_extractrhok += "\n" + pythoncmd + utils_path + '/utils_gridcont.py -update_config -input_path ' + result_path_level_coarse + ' -output_path ' + result_path_level
            # cmd_extractrhok += "\n" + "source " + os.path.join(result_path_level_coarse, 'env_rhok.sh') + "\n"


        # postprocess results
        #   - resize all images back to input resolution and save as nifti
        if level == 256:
            cmd_postproc = "\n# resize back to original resolution and convert to nifty" 
            cmd_postproc += "\n" + pythoncmd + utils_path + '/utils_gridcont.py -convert -input_path ' + output_path_tumor + ' -reference_image ' + fname + ' -multilevel '

        # TODO
        # cmd_postproc += " -rdir " + rdir + " ";
        # cmd_postproc += " -convert_images -gridcont ";
        # cmd_postproc += " -compute_tumor_stats ";
        # if args.run_registration:
        #     cmd_postproc += " -postprocess_registration ";
        # cmd_postproc += " -analyze_concomps ";
        # cmd_postproc += " -generate_slices " + "\n";

        cmd_command += "\n\n\n" + "# ### LEVEL {} ###\n# ==================\n".format(level)
        cmd_command += cp_cmd
        cmd_command += resample_cmd
        cmd_command += symlink_cmd
        cmd_command += cmd_concomp
        if level > 64:
            cmd_command += cmd_extractrhok

        ############### === define parameters
        r['nodes']     = nodes
        r['mpi_tasks']  = procs[-1]
        r['wtime_h']   = wtime_h[-1]
        r['wtime_m']   = wtime_m[-1]
        r['log_dir']   = output_path_tumor
        if not use_gpu:
          r['ibrun_man'] = " -n " + str(procs[ii]) + " -o 0 "
        
        p['output_dir']         = result_path_level
        p['n']                  = level
        p['solver']             = 'sparse_til'
        p['multilevel']         = 1
        p['invert_reac']        = 1
        p['invert_diff']        = invert_diffusivity[ii]
        p['inject_solution']    = 1 if (inject_coarse_sol  and level > 64) else 0;
        p['pre_reacdiff_solve'] = 1 if (pre_reacdiff_solve and level > 64) else 0;
        p['model']              = 1
        p['verbosity']          = 1
        p['syn_flag']           = 0
        p['init_rho']           = rho_init if (level == 64) else 'TBD'
        p['init_k']             = k_init if (level == 64) else 'TBD'
        p['nt_inv']             = nt
        p['dt_inv']             = 1./nt
        p['time_history_off'] 	= 0
        p['sparsity_level'] 	= sparsity_per_comp
        p['gist_maxit']         = gist_maxit[ii]
        p['beta_p']             = beta_p
        p['opttol_grad']        = opttol
        p['newton_maxit']       = newton_maxit[ii]
        p['kappa_lb']           = kappa_lb[ii]
        p['kappa_ub']           = kappa_ub[ii]
        p['rho_lb']             = rho_lb[ii]
        p['rho_ub']             = rho_ub[ii]
        p['k_gm_wm']            = k_gm_wm
        p['r_gm_wm']            = r_gm_wm
        p['newton_solver']      = "QN"
        p['line_search']        = ls_type[ii]
        p['lbfgs_vectors'] 		= 50
        p['store_adjoint']    = 1           # 1: store adjoint time history
        p['gaussian_volume_fraction'] = gvf[ii]
        p['threshold_data_driven']    = data_thresh[ii]
        p['sigma_factor']             = sigma_fac[ii]
        p['prediction']               = predict[ii]
        if predict[ii]:
            p['pred_times'] = [1.0, 1.2, 1.5]
            p['dt_pred'] = 0.01
        p['smoothing_factor'] = 1
        if healthy_seg:
            p['smoothing_factor_data'] = 0
        else:
            p['smoothing_factor_data'] = 1
        if 'obs_threshold_1' in input:
            p['obs_threshold_1'] = input['obs_threshold_1']
        if 'obs_threshold_rel' in input:
            p['obs_threshold_rel'] = 0 #input['obs_threshold_rel']
        if 'thresh_component_weight' in input:
            p['thresh_component_weight'] = input['thresh_component_weight']

        if gaussian_mode in ["C0", "D"] or level == 64:
            p['support_data_path'] = os.path.join(input_path_level, 'support_data' + ext); # on coarsest level always d(1), i.e., TC as support_data
            p['data_comp_path'] = os.path.join(input_path_level, 'data_comps' + ext);
        elif gaussian_mode == "PHI":
            p['support_data_path'] = os.path.join(input_path_level, 'support_data_phi' + ext);
            p['data_comp_path'] = os.path.join(input_path_level, 'data_comps' + ext);
        else:
            p['support_data_path'] = os.path.join(input_path_level, 'phi-support-c0.txt');
            p['data_comp_path'] = "";
        p['data_comp_data_path'] = os.path.join(result_path_level, 'dcomp.dat');

        if inject_coarse_sol and level > 64:
            p['gaussian_cm_path'] = os.path.join(result_path_level_coarse, 'phi-mesh-scaled.txt');
            p['pvec_path'] = os.path.join(result_path_level_coarse, 'p-rec-scaled.txt');



        run_str = par.write_config(p, r, use_gpu)
        cmd_command += "\n" + "# run tumor solver\n" + run_str + " > " + os.path.join(result_path_level, "solver_log.txt") + " 2>&1"
        if level == 256:
          cmd_command += "\n" + cmd_postproc
        cmd_command +="\n#=================="
        JOBfile += cmd_command

    ### write config to write_path and submit job
    if batch_end:
        fname_job = os.path.join(job_path, 'job_' + str(job_idx) + '.sh')
        job_file = open(fname_job, 'w+')
        # get header, config and run str
        job_header = par.write_jobscript_header(p, r, use_gpu)
        # write
        job_file.write(job_header)
        job_file.write(JOBfile)
        job_file.close()
        if submit:
          process = subprocess.check_output(['sbatch', fname_job]).strip();
          print(process)
          # subprocess.call(['sbatch', fname_job])



### ________________________________________________________________________ ___
### //////////////////////////////////////////////////////////////////////// ###
if __name__=='__main__':
    # repository base directory
    basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
    # parse arguments
    parser = argparse.ArgumentParser(description='Process input images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument ('-ppath',   '--patient_path', type = str, help = 'path to patient image directory containing patient segmentation and T1 MRI)', required=True)
    parser.add_argument ('-plabels', '--patient_labels', type=str, help = 'comma separated patient image segmentation labels. for ex.\n  0=bg,1=nec,2=ed,4=enh,5=wm,6=gm,7=vt,8=csf\n for BRATS type segmentation.');
    parser.add_argument ('-x',       '--output_base_path', type = str, default = os.path.join(basedir, 'results/'), help = 'path to results directory');
    parser.add_argument ('-lambda',  '--obs_lambda', type = float, default = 1, help = 'parameter to control observation operator OBS = TC + lambda (1-WT)');
    parser.add_argument ('-multi',   '--multiple_patients', action='store_true', help = 'process multiple patients, -patient_path should be the base directory containing patient folders which contain patient image(s).');
    args = parser.parse_args();

    input = {}
    input['patient_path'] = args.patient_path
    input['output_base_path'] = args.output_base_path
    input['obs_lambda'] = args.obs_lambda
    if args.patient_labels is None:
        args.patient_labels = "0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm"
    input['segmentation_labels'] = args.patient_labels


    if args.multiple_patients:
        print('running for multiple patients')
        # get the number of patients in the given path
        patient_list = next(os.walk(args.patient_path))[1];
        base_output_base_path = args.output_base_path;
        base_patient_path = args.patient_path
        for patient in patient_list:
            print('processing ', patient)
            input['patient_path'] = os.path.join(os.path.join(base_patient_path, patient), patient + '_seg_tu.nii.gz');
            input['output_base_path'] = os.path.join(base_output_base_path, patient);
            sparsetil_gridcont(input);
    else:
        print('running for single patient')
        sparsetil_gridcont(input);
