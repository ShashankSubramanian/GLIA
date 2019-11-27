import os, sys, warnings, argparse, subprocess
import TumorParams
from shutil import copyfile
import preprocess as prep
import nibabel as nib
import numpy as np
import imageTools as imgtools
import nibabel as nib
import file_io as fio
import claire


P_COUNTER = 0;
ONEJOB    = "";

###
### ------------------------------------------------------------------------ ###
def createJobsubFile(cmd, opt, level):
    """
    @short - create job submission file on tacc systems
    """
    # construct bash file name
    bash_filename = "job-submission-l" + str(level) +".sh";

    if not os.path.exists(opt['output_dir']):
        os.makedirs(opt['output_dir'])

    bash_filename = os.path.join(opt['output_dir'], bash_filename);

    # create bash file
    print("creating", bash_filename)
    bash_file = open(bash_filename,'w');

    # header
    bash_file.write("#!/bin/bash\n\n");

    if opt['compute_sys'] == 'hazelhen':

        bash_file.write("#PBS -N tumor-inv-grid-cont\n");
        bash_file.write("#PBS -l nodes=" + str(opt['num_nodes']) + ":ppn=24" + "\n");
        bash_file.write("#PBS -l walltime=" + str(opt['wtime_h']) + ":" + str(opt['wtime_m']) + ":00"  "\n");
        bash_file.write("#PBS -m e\n");
        bash_file.write("#PBS -j oe\n");
        bash_file.write("#PBS -M kscheufele@austin.utexas.edu\n");
        bash_file.write("\n\n");
        bash_file.write("source /zhome/academic/HLRS/ipv/ipvscheu/env_intel.sh\n");

    elif opt['compute_sys'] == 'cbica':
        bash_file.write("#$ -S /bin/bash\n")
        bash_file.write("#$ -cwd\n")
        bash_file.write("#$ -pe openmpi " + str(opt['mpi_pernode']*opt['num_nodes']) + "\n");

    else:
        bash_file.write("#SBATCH -J tumor-inv-grid-cont\n");
        bash_file.write("#SBATCH -n " + str(opt['mpi_pernode']) + "\n");
        if opt['compute_sys'] == 'lonestar':
            bash_file.write("#SBATCH -p normal\n");
            opt['num_nodes'] = 2;
            bash_file.write("#SBATCH -N " + str(opt['num_nodes']) + "\n");
        elif opt['compute_sys'] == 'stampede2':
            bash_file.write('#SBATCH -p skx-normal\n');
            #opt['num_nodes'] = 3;
            bash_file.write("#SBATCH -N " + str(opt['num_nodes']) + "\n");
        elif opt['compute_sys'] == 'frontera':
            bash_file.write('#SBATCH -p normal\n');
            #opt['num_nodes'] = 3;
            bash_file.write("#SBATCH -N " + str(opt['num_nodes']) + "\n");
        elif opt['compute_sys'] == 'local':
            bash_file.write('#SBATCH -p rebels\n');
            opt['num_nodes'] = 1;
            bash_file.write("#SBATCH -N " + str(opt['num_nodes']) + "\n");
        elif opt['compute_sys'] == 'maverick2':
            bash_file.write('#SBATCH -p p100\n');
            opt['num_nodes'] = 1;
            bash_file.write("#SBATCH -N " + str(opt['num_nodes']) + "\n");
        else:
            bash_file.write("#SBATCH -p normal\n");
            opt['num_nodes'] = 1;
            bash_file.write("#SBATCH -N " + str(opt['num_nodes']) + "\n");

    if opt['compute_sys'] == 'cbica':
        bash_file.write("#$ -o " + os.path.join(opt['output_dir'], "grid-cont-l"+str(level)+".out") + "\n");

    else:
        bash_file.write("#SBATCH -o " + os.path.join(opt['output_dir'], "grid-cont-l"+str(level)+".out ") + "\n");
        bash_file.write("#SBATCH -t " + str(opt['wtime_h']) + ":" + str(opt['wtime_m']) + ":00\n");
        #bash_file.write("#SBATCH --mail-user=nutexas.edu\n");
        #bash_file.write("#SBATCH --mail-type=fail\n");
        if opt['compute_sys'] == 'frontera':
            bash_file.write("#SBATCH -A FTA-Biros\n");
        else:
            bash_file.write("#SBATCH -A PADAS\n");


    bash_file.write("\n\n");
    bash_file.write("source ~/.bashrc\n");
    # bash_file.write("#### define paths\n");
    # bash_file.write("DATA_DIR=" + opt['input_dir'] + "\n");
    # bash_file.write("OUTPUT_DIR=" + opt['output_dir'] + "\n");
    # bash_file.write("cd " + opt['output_dir']  + "\n");
    bash_file.write("export OMP_NUM_THREADS=1\n");
    bash_file.write("umask 002\n");
    bash_file.write("\n");
    bash_file.write("\n");
    bash_file.write(cmd);
    bash_file.write("\n");

    # write out done
    bash_file.close();
    return bash_filename;
    # submit job
    # subprocess.call(['sbatch',bash_filename]);

###
### ------------------------------------------------------------------------ ###
def registration(args, basedir):
    '''
    Function to run registration
    '''

    # create output folder
    base_output_dir = args.results_directory
    if not os.path.exists(base_output_dir):
        os.mkdir(base_output_dir);
    # create input folder
    data_dir = os.path.join(base_output_dir, 'input');
    if not os.path.exists(data_dir):
        os.mkdir(data_dir);

    tumor_dir = os.path.join(base_output_dir, 'tumor_inversion/nx256/obs-1.0');
    if not os.path.exists(tumor_dir):
        os.mkdir(tumor_dir)

    reg_dir = os.path.join(base_output_dir, 'registration');
    if not os.path.exists(reg_dir):
        os.mkdir(reg_dir)

    patient_labels = ",".join([x.split('=')[0] for x in args.patient_segmentation_labels.split(',')])
    reg_param = {}
    reg_param['reg_code_dir'] = args.reg_code_dir
    reg_param['compute_sys'] = args.compute_cluster
    claire.set_parameters(reg_param, base_output_dir, tumor_dir)
    reg_cmd = "#### define paths\n"
    reg_cmd += "CLAIRE_BDIR=" + reg_param["reg_code_dir"] + "/bin\n"
    reg_cmd += "DATA_DIR=" + reg_param['data_dir'] + "\n"
    reg_cmd += "OUTPUT_DIR=" + reg_param['output_dir'] + "\n"
    reg_cmd += "TUMOR_DIR=" + reg_param['tumor_output_dir'] + "\n"

    # registration command
    reg_cmd += claire.createCmdLineReg(reg_param)
    # transport labels command
    reg_cmd += claire.createCmdLineTransport(reg_param, task='tlabelmap', labels=patient_labels, input_filename="$DATA_DIR/patient_seg.nii.gz", output_filename="$OUTPUT_DIR/patient_seg_in_Aspace.nii.gz", )
    reg_cmd += "\n#convert netcdf to nifti\npython3 " + basedir + "/grid-cont/utils.py -convert_netcdf_to_nii --name_old $TUMOR_DIR/c0Recon.nc --name_new $TUMOR_DIR/c0Recon.nii.gz --reference_image $DATA_DIR/patient_wm.nii.gz\n\n"
    reg_cmd += "#convert netcdf to nifti\npython3 " + basedir + "/grid-cont/utils.py -convert_netcdf_to_nii --name_old $TUMOR_DIR/cRecon.nc --name_new $TUMOR_DIR/cRecon.nii.gz --reference_image $DATA_DIR/patient_wm.nii.gz\n\n\n"

    # transport c0 command
    reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$TUMOR_DIR/c0Recon.nii.gz", output_filename="$TUMOR_DIR/c0Recon_in_Aspace.nii.gz")
    # transport c1 command
    reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$TUMOR_DIR/cRecon.nii.gz", output_filename="$TUMOR_DIR/cRecon_in_Aspace.nii.gz")
    # transport ventricles command
    reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$DATA_DIR/patient_vt.nii.gz", output_filename="$OUTPUT_DIR/patient_vt_in_Aspace.nii.gz")
    # transport CSF command
    reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$DATA_DIR/patient_csf_no_vt.nii.gz", output_filename="$OUTPUT_DIR/patient_csf_no_vt_in_Aspace.nii.gz")
    # transport gray matter command
    reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$DATA_DIR/patient_gm.nii.gz", output_filename="$OUTPUT_DIR/patient_gm_in_Aspace.nii.gz")
    # transport white matter command
    reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$DATA_DIR/patient_wm.nii.gz", output_filename="$OUTPUT_DIR/patient_wm_in_Aspace.nii.gz")
    # transport tumor command
    reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$DATA_DIR/patient_tu.nii.gz", output_filename="$OUTPUT_DIR/patient_tu_in_Aspace.nii.gz")
    # transport edema+white matter command
    reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$DATA_DIR/patient_ed_wm.nii.gz", output_filename="$OUTPUT_DIR/patient_ed_wm_in_Aspace.nii.gz")

    return reg_cmd
    #local_cmd = preproc_cmd + "\n\n\n" + reg_cmd + "\n\n\n" + postproc_cmd
    #
    #bash_filename = claire.create_cmd_file(local_cmd,reg_param)
    #
    ## create Job file
    #if num_patients is not None:
    #    if num_patients == 0:
    #        global_cmd = ""
    #    global_cmd += "\n" +  bash_filename
    #    num_patients += 1
    #
    #    if num_patients == patients_per_job:
    #        claire.createJobsubFile(global_cmd,reg_param,submit=True)
    #        global_cmd = ""
    #        num_patients = 0
    #else:
    #    global_cmd = bash_filename
    #    claire.createJobsubFile(global_cmd,reg_param,submit=True)
    #    global_cmd = ""
    #
    #return global_cmd,num_patients


###
### ------------------------------------------------------------------------ ###
def gridcont(basedir, args):

    # ########### SETTINGS ############
    patients_per_job   = 1;
    levels             = [64,128,256]
    if args.compute_cluster == "stampede2":
      nodes      = [1,1,2]
      procs      = [24,48,96]
    if args.compute_cluster == "frontera":
      nodes      = [1,1,2]
      procs      = [24,48,96]
    if args.compute_cluster == "hazelhen":
      nodes            = [1,2,4]
      procs            = [24,48,96]
    if args.compute_cluster == "maverick2":
      nodes            = [1,1,1]
      procs            = [1,1,1]
    if args.compute_cluster == "cbica":
      nodes            = [1,1,1]
      procs            = [20,20,20]
    wtime_h            = [x * patients_per_job for x in [0,2,12]];
    wtime_m            = [x * patients_per_job for x in [30,0,0]];
    sigma_fac          = [1,1,1]                    # on every level, sigma = fac * hx
    predict            = [0,0,0]
    gvf                = [0.0,0.9,0.9]              # ignored for C0_RANKED
    rho_default        = 8;
    k_default          = 0;
    beta_p             = 1;
    opttol             = 1E-4;
    p_prev             = "";
    submit             = True;
    separatejobs       = False;
    inject_coarse_sol  = True;
    pre_reacdiff_solve = True;
    pid_prev           = 0;
    obs_masks          = []
    lbound_kappa       = [1E-4, 1E-4, 1E-4]; # [1E-2, 1E-3, 1E-4];
    ubound_kappa       = [1.0, 1.0, 1.0];
    gaussian_selection_mode = "PHI"; # alternatives: {"D", "PHI", "C0", "C0_RANKED"}
    data_thresh  = [1E-1, 1E-4, 1E-4] if (gaussian_selection_mode in ["PHI","C0"]) else [1E-1, 1E-1, 1E-1];
    sparsity_lvl_per_component = 5;
    ls_max_func_evals  = [10, 10, 10];
    invert_diffusivity = [1,1,1];
    # #################################

    global P_COUNTER;
    P_COUNTER = P_COUNTER + 1 if P_COUNTER < patients_per_job else 1;
    batch_end = P_COUNTER == patients_per_job
    submit    = submit and batch_end;
    new_job   = P_COUNTER == 1

    global ONEJOB;
    if new_job:
        ONEJOB = ""
    else:
        ONEJOB += "\n\n###############################################################\n###############################################################\n###############################################################\n\n\n";

    MULTJOB = ""

    os.environ['DIR_SUFFIX'] = "{0:1.1f}".format(args.obs_lambda);
    obs_dir = "obs-{0:1.1f}".format(args.obs_lambda) + "/";
    if args.cm_data:
        obs_dir     = "cm-data-obs-{0:1.1f}".format(args.obs_lambda) + "/";
        obs_dir_std = "obs-{0:1.1f}".format(args.obs_lambda) + "/";

    # create output folder
    output_path = args.results_directory
    if not os.path.exists(output_path):
        print("results folder doesn't exist, creating one!\n");
        os.mkdir(output_path);
    # python command
    pythoncmd = "python ";
    if args.compute_cluster in ["stampede2", "frontera", "local"]:
        pythoncmd = "python3 ";
    # create input folder
    input_folder = os.path.join(output_path, 'input');
    if not os.path.exists(input_folder):
        os.mkdir(input_folder);

    #   ------------------------------------------------------------------------
    #   - read atlas segmented image and create probability maps
    #   - read segmented patient or patient probability maps and resize them
    cmd_preproc = pythoncmd + basedir + '/grid-cont/preprocess.py -atlas_image_path ' + args.atlas_image_path  + ' -patient_image_path ' + args.patient_image_path + ' -output_path ' + input_folder + ' -N ' + str(args.resolution) + ' -patient_labels ' + args.patient_segmentation_labels + ' -atlas_labels ' + args.atlas_segmentation_labels;
    if args.use_atlas_segmentation:
        cmd_preproc += ' --use_atlas_segmentation'
    if args.use_patient_segmentation:
        cmd_preproc += ' --use_patient_segmentation'

    # compute observation operartor mask (lambda)
    cmd_obs =  "\n\n# observation mask\n" + pythoncmd + basedir + '/grid-cont/utils.py -compute_observation_mask --obs_lambda ' + str(args.obs_lambda) + ' -output_path ' + input_folder
    cmd_obs_resample64  = "\n" + pythoncmd + basedir + '/grid-cont/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'obs_mask.nc'          + ' --name_new ' + 'obs_mask_nx64.nc'          + ' --N_old 256 --N_new 64 &';
    cmd_obs_resample128 = "\n" + pythoncmd + basedir + '/grid-cont/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'obs_mask.nc'          + ' --name_new ' + 'obs_mask_nx128.nc'         + ' --N_old 256 --N_new 128 &';
    cmd_obs_rename    = "mv " + os.path.join(input_folder, 'obs_mask.nc')          + " " + os.path.join(input_folder, 'obs_mask_nx' + str(256) + '.nc') + " \n"
    # compute observation operator mask (lambda \in {1, 0.8, 0.6, 0.4, 0.2, 0.1})
    if args.vary_obs_lambda:
        cmd_obs =  "\n\n# observation mask\n";
        cmd_obs_resample64 = ""
        cmd_obs_resample128 = ""
        for lambda_o in [1]: #[1, 0.8, 0.6, 0.4, 0.2, 0.1]:
            name = "obs_mask" + '_lbd-'+str(lambda_o);
            obs_masks.append(name);
            cmd_obs             += "\n" + pythoncmd + basedir + '/grid-cont/utils.py -compute_observation_mask --obs_lambda ' + str(lambda_o) + ' -output_path ' + input_folder + ' --suffix ' + '_lbd-'+str(lambda_o)
            cmd_obs_resample64  += "\n" + pythoncmd + basedir + '/grid-cont/utils.py -resample -output_path ' + input_folder + ' --name_old ' + name + '.nc'          + ' --name_new ' + name + '_nx64.nc'          + ' --N_old 256 --N_new 64 &';
            cmd_obs_resample128 += "\n" + pythoncmd + basedir + '/grid-cont/utils.py -resample -output_path ' + input_folder + ' --name_old ' + name + '.nc'          + ' --name_new ' + name + '_nx128.nc'         + ' --N_old 256 --N_new 128 &';
            cmd_obs_rename      += "mv " + os.path.join(input_folder, name + '.nc')          + " " + os.path.join(input_folder, name + '_nx' + str(256) + '.nc') + " \n"


    cmd_preproc += cmd_obs;

    #   - resample 256 --> 64
    cmd_preproc +=  "\n\n# resample\n";
    cmd_preproc +=           pythoncmd + basedir + '/grid-cont/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_csf.nc'   + ' --name_new ' + 'patient_nx64_seg_csf.nc'   + ' --N_old 256 --N_new 64 &';
    cmd_preproc +=    "\n" + pythoncmd + basedir + '/grid-cont/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_gm.nc'    + ' --name_new ' + 'patient_nx64_seg_gm.nc'    + ' --N_old 256 --N_new 64 &';
    cmd_preproc +=    "\n" + pythoncmd + basedir + '/grid-cont/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_wm_wt.nc' + ' --name_new ' + 'patient_nx64_seg_wm_wt.nc' + ' --N_old 256 --N_new 64 &';
    cmd_preproc +=    "\n" + pythoncmd + basedir + '/grid-cont/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_tc.nc'    + ' --name_new ' + 'patient_nx64_seg_tc.nc'    + ' --N_old 256 --N_new 64 &';
    cmd_preproc += cmd_obs_resample64;
    #   - resample 256 --> 128
    cmd_preproc +=  "\n\n" + pythoncmd + basedir + '/grid-cont/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_csf.nc'   + ' --name_new ' + 'patient_nx128_seg_csf.nc'   + ' --N_old 256 --N_new 128 &';
    cmd_preproc +=    "\n" + pythoncmd + basedir + '/grid-cont/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_gm.nc'    + ' --name_new ' + 'patient_nx128_seg_gm.nc'    + ' --N_old 256 --N_new 128 &';
    cmd_preproc +=    "\n" + pythoncmd + basedir + '/grid-cont/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_wm_wt.nc' + ' --name_new ' + 'patient_nx128_seg_wm_wt.nc' + ' --N_old 256 --N_new 128 &';
    cmd_preproc +=    "\n" + pythoncmd + basedir + '/grid-cont/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_tc.nc'    + ' --name_new ' + 'patient_nx128_seg_tc.nc'    + ' --N_old 256 --N_new 128 &';
    cmd_preproc += cmd_obs_resample128;
    #   - rename 256
    cmd_rename  = "mv " + os.path.join(input_folder, 'patient_seg_csf.nc')   + " " + os.path.join(input_folder, 'patient_nx'  + str(256) + '_seg_csf.nc') + " \n"
    cmd_rename += "mv " + os.path.join(input_folder, 'patient_seg_gm.nc')    + " " + os.path.join(input_folder, 'patient_nx'  + str(256) + '_seg_gm.nc') + " \n"
    cmd_rename += "mv " + os.path.join(input_folder, 'patient_seg_wm_wt.nc') + " " + os.path.join(input_folder, 'patient_nx'  + str(256) + '_seg_wm_wt.nc') + " \n"
    cmd_rename += "mv " + os.path.join(input_folder, 'patient_seg_tc.nc')    + " " + os.path.join(input_folder, 'patient_nx'  + str(256) + '_seg_tc.nc') + " \n"
    cmd_rename += cmd_obs_rename;

    #   ------------------------------------------------------------------------
    #   - tumor inversion, levels -- N=64^3, N=128^3, N=256^3 --
    t_params = {};
    tumor_out_path = os.path.join(output_path, 'tumor_inversion/');
    t_params['compute_sys']  = args.compute_cluster;
    if args.tumor_code_dir is not None:
        t_params['code_path'] = args.tumor_code_dir
    else:
        t_params['code_path']    = os.path.join(basedir, '3rdparty/pglistr_tumor');

    cmd_lvl = "\n\n# generate maps, resample\n" + cmd_preproc + "\nwait\n\n# rename\n" + cmd_rename + "\n\n";
    # cmd     += cmd_lvl;
    ONEJOB  += cmd_lvl;
    MULTJOB += cmd_lvl;

    # loop over levels
    opt = {}
    for level, ii  in zip(levels, range(len(levels))):
        cmd = ""

        res_dir = os.path.join(tumor_out_path, 'nx' + str(level) + "/");
        res_dir_out = os.path.join(res_dir, obs_dir)
        res_dir_prev = os.path.join(tumor_out_path, 'nx' + str(int(level/2)) + "/" + obs_dir);
        if not os.path.exists(res_dir):
            os.makedirs(res_dir, exist_ok=True);
        inp_dir = os.path.join(res_dir, 'init');
        if not os.path.exists(inp_dir):
            os.mkdir(inp_dir);

        cmd += "\n\n### ----------------------------------------- ###\n"
        cmd += "###   LEVEL nx=" + str(level) + ", sigma=2pi/" + str(int(level/sigma_fac[ii])) + "   ###\n"
        cmd += "\n" + "cd " + str(res_dir_out)  + "\n"
        # symlinks
        cmd_symlink  =  "PWDO=${PWD} \n"
        cmd_symlink +=  "cd " + str(inp_dir) + "\n"
        cmd_symlink +=  "ln -sf " + "../../../input/patient_nx" + str(level)  + "_seg_csf.nc"   " patient_seg_csf.nc \n";
        cmd_symlink +=  "ln -sf " + "../../../input/patient_nx" + str(level)  + "_seg_gm.nc"    " patient_seg_gm.nc \n";
        cmd_symlink +=  "ln -sf " + "../../../input/patient_nx" + str(level)  + "_seg_wm_wt.nc" " patient_seg_wm_wt.nc \n";
        cmd_symlink +=  "ln -sf " + "../../../input/patient_nx" + str(level)  + "_seg_tc.nc"    " patient_seg_tc.nc \n";
        if args.vary_obs_lambda:
            for name in obs_masks:
                cmd_symlink +=  "ln -sf " + "../../../input/" + name + "_nx" + str(level) + ".nc"    " " + name + ".nc \n";
        else:
            cmd_symlink +=  "ln -sf " + "../../../input/obs_mask_nx" + str(level) + ".nc"           " obs_mask.nc \n";

        if not args.cm_data:
            if level == 64 or gaussian_selection_mode == "D":
                cmd_symlink +=  "ln -sf " + "../../../input/patient_nx" + str(level) + "_seg_tc.nc"     " support_data.nc \n";
            else:
                level_prev = int(level/2);
                cmd_symlink += pythoncmd + basedir + '/grid-cont/utils.py -resample -output_path ' + os.path.join(tumor_out_path, 'nx' + str(level_prev) + "/" + obs_dir) + ' --name_old ' + 'c0Recon.nc'           + ' --name_new ' + 'c0Recon_nx'+ str(level)         + '.nc' + ' --N_old ' + str(level_prev) + ' --N_new ' + str(level) + ' \n';
                cmd_symlink +=  "ln -sf " + "../../nx"+ str(level_prev)+ "/" + obs_dir +"/c0Recon_nx"+str(level)+".nc"   " support_data.nc \n";
                cmd_symlink += pythoncmd + basedir + '/grid-cont/utils.py -resample -output_path ' + os.path.join(tumor_out_path, 'nx' + str(level_prev) + "/" + obs_dir) + ' --name_old ' + 'phiSupportFinal.nc'   + ' --name_new ' + 'phiSupportFinal_nx'+ str(level) + '.nc' + ' --N_old ' + str(level_prev) + ' --N_new ' + str(level) + ' \n';
                cmd_symlink +=  "ln -sf " + "../../nx"+ str(level_prev)+ "/" + obs_dir +"/phiSupportFinal_nx"+str(level)+".nc"   " support_data_phi.nc \n";
        cmd_symlink  +=  "cd ${PWDO} \n"

        # compute connecte components of target data
        rdir = "obs" if not args.cm_data else "cm-data";
        cmd_concomp  = pythoncmd + basedir + '/grid-cont/utils.py -concomp_data  -input_path ' +  os.path.join(tumor_out_path, 'nx' + str(level)) + ' -output_path ' +  input_folder + ' --obs_lambda ' + str(args.obs_lambda);
        cmd_concomp += " -rdir " + rdir  + "  --sigma " + str(sigma_fac[ii]) + " ";
        cmd_concomp +=  " -select_gaussians  \n" if (gaussian_selection_mode == 'C0_RANKED' and level > 64) else " \n";
        cmd_concomp +=  "PWDO=${PWD} \n"
        cmd_concomp +=  "cd " + str(inp_dir) + "\n"
        cmd_concomp +=  "ln -sf " + "../../../input/data_comps_nx" + str(level)  + ".nc"    " data_comps.nc \n";
        cmd_concomp +=  "cd ${PWDO} \n"

        #   ------------------------------------------------------------------------
        #   - extract reconstructed rho and k
        if level > 64:
            cmd_extractrhok =  "\n\n# extract reconstructed rho, k\n";
            cmd_extractrhok +=  pythoncmd + basedir + '/grid-cont/utils.py -extract -output_path ' + res_dir_prev + ' --rho_fac 1' + " \n"
            cmd_extractrhok += "source " + os.path.join(res_dir_prev, 'env_rhok.sh') + "\n"

        # tumor settings for current level
        t_params['num_nodes']             = nodes[ii];
        t_params['mpi_pernode']           = procs[ii];
        t_params['wtime_h']               = wtime_h[ii];
        t_params['wtime_m']               = wtime_m[ii];
        t_params['ibrun_man']             = (level <= 128);
        t_params['results_path']          = res_dir_out;
        t_params['N']                     = level;
        t_params['grad_tol']              = opttol;
        t_params['sparsity_lvl']          = sparsity_lvl_per_component;
        t_params['multilevel']            = 1;
        t_params['model']                 = 1;
        t_params['solve_rho_k']           = 1 if args.cm_data else 0;
        t_params['inject_solution']       = 1 if (inject_coarse_sol  and level > 64) else 0;
        t_params['pre_reacdiff_solve']    = 1 if (pre_reacdiff_solve and level > 64) else 0;
        t_params['create_synthetic']      = 0;
        t_params['ls_max_func_evals']     = ls_max_func_evals[ii];
        t_params['diffusivity_inversion'] = invert_diffusivity[ii];
        t_params['data_thres']            = data_thresh[ii];
        t_params['rho_inv']               = rho_default if (level == 64) else '${RHO_INIT}';
        t_params['k_inv']                 = k_default   if (level == 64) else '${K_INIT}';
        t_params['gist_maxit']            = 4           if (level == 64) else 2;
        t_params['linesearchtype']        = 'mt'        if (level == 64) else 'armijo';
        t_params['newton_maxit']          = 30 if (level == 256) else 50;
        t_params['gvf']                   = gvf[ii];
        t_params['beta']                  = beta_p;
        t_params['lower_bound_kappa']     = lbound_kappa[ii];
        t_params['upper_bound_kappa']     = ubound_kappa[ii];
        t_params['dd_fac']                = sigma_fac[ii];
        t_params['predict_flag']          = predict[ii];
        t_params['csf_path']              = os.path.join(inp_dir, 'patient_seg_csf.nc');
        t_params['gm_path']               = os.path.join(inp_dir, 'patient_seg_gm.nc');
        t_params['wm_path']               = os.path.join(inp_dir, 'patient_seg_wm_wt.nc');
        t_params['data_path']             = os.path.join(inp_dir, 'patient_seg_tc.nc');
        t_params['forward_flag']          = 0
        t_params['smooth_f']              = 1.5
        t_params['model']                 = 1
        if gaussian_selection_mode in ["C0", "D"] or level == 64:
            t_params['support_data_path']  = os.path.join(inp_dir, 'support_data.nc'); # on coarsest level always d(1), i.e., TC as support_data
            t_params['data_comp_path']     = os.path.join(inp_dir, 'data_comps.nc');
        elif gaussian_selection_mode == "PHI":
            t_params['support_data_path']  = os.path.join(inp_dir, 'support_data_phi.nc');
            t_params['data_comp_path']     = os.path.join(inp_dir, 'data_comps.nc');
        else:
            t_params['support_data_path']  = os.path.join(inp_dir, 'phi-support-c0.txt');
            t_params['data_comp_path']     = "";
        t_params['data_comp_dat_path']     = os.path.join(res_dir_out, 'dcomp.dat');
        t_params['obs_mask_path']          = os.path.join(inp_dir, 'obs_mask_lbd-${LAMBDA_OBS}.nc') if (args.vary_obs_lambda) else os.path.join(inp_dir, 'obs_mask.nc');

        if args.cm_data:
            t_params['support_data_path']  = "";
            t_params['data_comp_path']     = "";
            t_params['data_comp_dat_path'] = "";
            t_params['gaussian_cm_path']   = os.path.join(res_dir_out, 'phi-cm-data.txt');
            t_params['pvec_path']          = os.path.join(res_dir_out, 'p-cm-data.txt');
        if inject_coarse_sol and level > 64:
            t_params['gaussian_cm_path']   = os.path.join(res_dir_prev, 'phi-mesh-scaled.txt');
            t_params['pvec_path']          = os.path.join(res_dir_prev, 'p-rec-scaled.txt');


        #   ------------------------------------------------------------------------
        #    - get command line for tumor inversion
        cmdline_tumor, err = TumorParams.getTumorRunCmd(t_params)
        cmdline_tumor += " &> " + t_params["results_path"] + "tumor_solver_log_nx"+str(level)+".txt";
        if err:
            warnings.warn("Error in tumor parameters\n");
            quit();
        cmd += "\n\n# symlinks\n" + cmd_symlink;
        cmd += "\n\n# connected components\n" + cmd_concomp;
        if level > 64:
            cmd += cmd_extractrhok;
        cmd += "\n### tumor inversion ### \n";
        cmd += "export LAMBDA_OBS=1\n";
        # cmd += "export DIR_SUFFIX='obs-1.0'\n";
        # cmd += "mkdir -p ${DIR_SUFFIX}\n"
        cmd += cmdline_tumor + "\n\n";


        #   ------------------------------------------------------------------------
        #   - resize all images back to input resolution and save as nifti
        cmd_postproc  = pythoncmd + basedir + '/grid-cont/postprocess.py -input_path ' + args.results_directory + ' -reference_image_path ' + args.patient_image_path + " -patient_labels " +  args.patient_segmentation_labels
        cmd_postproc += " -rdir " + rdir + " ";
        cmd_postproc += " -convert_images -gridcont ";
        cmd_postproc += " -compute_tumor_stats ";
        if args.run_registration:
            cmd_postproc += " -postprocess_registration ";
        cmd_postproc += " -analyze_concomps ";
        cmd_postproc += " -generate_slices " + "\n";

        # cmd_postproc += " -compute_dice_healthy " + "\n";

        #### TODO  REGISTRATION HERE ####
        if level == 256:
            if args.run_registration:
                cmdline_reg = registration(args, basedir)
                cmd += cmdline_reg + "\n\n";
            cmd += "\n\n# postproc, compute dice\n" + cmd_postproc + "\n\n";

        ONEJOB  += cmd;
        MULTJOB += cmd;

        opt['compute_sys']  = args.compute_cluster;
        opt['output_dir']  = res_dir_out;
        opt['input_dir']   = inp_dir;
        opt['num_nodes']   = nodes[ii];
        opt['mpi_pernode'] = procs[ii];
        opt['wtime_h']     = wtime_h[ii];
        opt['wtime_m']     = wtime_m[ii];

        if separatejobs:
            job_file = createJobsubFile(MULTJOB, opt, level);
            if submit:
                if level == 64:
                    if args.compute_cluster in ['hazelhen','cbica']:
                      process = subprocess.check_output(['qsub',job_file]).strip();
                    else:
                      process = subprocess.check_output(['sbatch',job_file]).strip();
                    print(process)
                else:
                    if args.compute_cluster == 'hazelhen':
                      process = subprocess.check_output(['qsub', '-W depend=afterok:'+str(pid_prev), job_file]).strip();
                    elif args.compute_cluster == 'cbica':
                        process = subprocess.check_output(['qsub', '-hold_jid '+str(pid_prev), job_file]).strip();
                    else:
                      process = subprocess.check_output(['sbatch', '--dependency=afterok:'+str(pid_prev), job_file]).strip();
                    print(process)
                print("\n");
                if args.compute_cluster == 'hazelhen':
                  print(str(process).split(".")[0])
                  pid_prev = int(str(process,'utf-8').split(".")[0])
                else:
                  print("\n pid:", str(process, 'utf-8').split("Submitted batch job ")[-1])
                  pid_prev = int(str(process, 'utf-8').split("Submitted batch job ")[-1])

    if not separatejobs and batch_end:
        job_file = createJobsubFile(ONEJOB, opt, 256);
        if submit:
            if args.compute_cluster in ['hazelhen','cbica']:
              process = subprocess.check_output(['qsub',job_file]).strip();
            else:
              process = subprocess.check_output(['sbatch',job_file]).strip();
            print(process)



###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    # repository base directory
    basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
    # parse arguments
    parser = argparse.ArgumentParser(description='Process input images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r_args = parser.add_argument_group('required arguments')
    r_args.add_argument ('-patient_path',   '--patient_image_path',          type = str, help = 'path to patient image directory containing the T1,T2,FLAIR,T1CE and segmentation images\n (format- PatientName_{t1,t2,t1ce,flair,segmented}.ext)', required=True)
    r_args.add_argument ('-patient_labels', '--patient_segmentation_labels', type=str,   help = 'comma separated patient segmented image labels. for ex.\n  0=bg,1=nec,2=ed,4=enh,5=wm,6=gm,7=vt,8=csf\n for BRATS type segmentation. DISCLAIMER vt and every extra label mentioned will be merged with csf');
    r_args.add_argument ('-atlas_path',     '--atlas_image_path',            type = str, help = 'path to a segmented atlas image (affinely registered to given patient)', required=True)
    r_args.add_argument ('-atlas_labels',   '--atlas_segmentation_labels',   type = str, help = 'comma separated atlas segmented image labels. for ex.\n 0=bg,1=vt,2=csf,3=gm,4=wm\n DISCLAIMER vt will be merged with csf')
    r_args.add_argument ('-cluster',        '--compute_cluster',             type = str, help = 'compute cluster name for creation of job script (ex. stampede2, frontera, hazelhen, cbica etc)', required=True);
    parser.add_argument ('-x',              '--results_directory',           type = str, default = os.path.join(basedir, 'results/'), help = 'path to results directory');
    parser.add_argument ('-np',             '--num_mpi_tasks',               type = int, default = 20,  help = 'number of MPI tasks per node, always run on a single node');
    parser.add_argument ('-nodes',          '--num_nodes',                   type = int, default = 3,   help = 'number of nodes');
    parser.add_argument ('-nx',             '--resolution',                  type = int, default = 256, help = 'spatial resolution');
    parser.add_argument ('-wtime_h',        '--wtime_h',                     type = int, default = 10,  help = 'wall clock time [hours]');
    parser.add_argument ('-wtime_m',        '--wtime_m',                     type = int, default = 0,   help = 'wall clock time [minutes]');
    parser.add_argument (                   '--use_patient_segmentation',    action='store_true', help = 'indicate whether the input patient image is a segmentation. Probability maps are then generated from given segmented image');
    parser.add_argument (                   '--use_atlas_segmentation',      action='store_true', help = 'indicate whether the input atlas image is a segmentation. Probability maps are then generated from given segmented image');
    parser.add_argument (                   '--vary_obs_lambda',             action='store_true', help = 'indicate wether or not to perform a series of experiment with different obervation operators OBS(lambda)');
    parser.add_argument (                   '--obs_lambda',                  type = float, default = 1,   help = 'parameter to control observation operator OBS = TC + lambda (1-WT)');
    parser.add_argument (                   '--multiple_patients',           action='store_true', help = 'process multiple patients, -patient_path should be the base directory containing patient folders which contain patient image(s).');
    parser.add_argument (                   '--run_registration',           action='store_true', help = 'run registration');
    parser.add_argument (                   '--cm_data',                    action='store_true', help = 'if true, L1 phase is skipped and CM of data is used, performing one L2 solve followed by rho and kappa inversion.');
    parser.add_argument (                   '--tumor_code_dir',                    type = str, help = 'path to tumor solver code directory')
    parser.add_argument (                   '--reg_code_dir',                      type = str, help = 'path to registration solver code directory')
    args = parser.parse_args();

    if args.patient_image_path is None:
        parser.error("--patient_image_path needs to be set.");
    if args.atlas_image_path is None:
        parser.error("--atlas_image_path needs to be set.");
    if args.compute_cluster is None:
        parser.error("--compute_cluster needs to be set.");
    if args.patient_segmentation_labels is None:
        parser.error("--patient_segmentation_labels needs to be set.");
    if args.atlas_segmentation_labels is None:
        parser.error("--atlas_segmentation_labels needs to be set.");

    if args.multiple_patients:
        print('running for multiple patients')
        # get the number of patients in the given path
        patient_list = next(os.walk(args.patient_image_path))[1];
        base_results_dir = args.results_directory;
        base_patient_image_path = args.patient_image_path
        for patient in patient_list:
            print('processing ', patient)
            args.patient_image_path = os.path.join(os.path.join(base_patient_image_path, patient), patient + '_seg_tu.nii.gz');
            args.results_directory = os.path.join(base_results_dir, patient);
            gridcont(basedir, args);
    else:
        print('running for single patient')
        gridcont(basedir, args);
