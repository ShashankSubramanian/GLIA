import os, sys, warnings, argparse, subprocess
import claire
import TumorParams
from shutil import copyfile
import preprocess as prep
import nibabel as nib
import numpy as np
import imageTools as imgtools
import nibabel as nib
import file_io as fio


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

        elif opt['compute_sys'] == 'local':
            bash_file.write('#SBATCH -p rebels\n');
            opt['num_nodes'] = 1;
            bash_file.write("#SBATCH -N " + str(opt['num_nodes']) + "\n");

        else:
            bash_file.write("#SBATCH -p normal\n");
            opt['num_nodes'] = 1;
            bash_file.write("#SBATCH -N " + str(opt['num_nodes']) + "\n");

        bash_file.write("#SBATCH -t " + str(opt['wtime_h']) + ":" + str(opt['wtime_m']) + ":00\n");
        bash_file.write("#SBATCH --mail-user=kscheufele@austin.utexas.edu\n");
        bash_file.write("#SBATCH --mail-type=fail\n");
        bash_file.write("#SBATCH -o " + os.path.join(opt['output_dir'], "grid-cont-l"+str(level)+".out ") + "\n");
        bash_file.write("\n\n");
        bash_file.write("source ~/.bashrc\n");
    bash_file.write("#### define paths\n");
    bash_file.write("DATA_DIR=" + opt['input_dir'] + "\n");
    bash_file.write("OUTPUT_DIR=" + opt['output_dir'] + "\n");
    bash_file.write("cd " + opt['output_dir']  + "\n");
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
def gridcont(basedir, args):

    cmd = ""
    # ########### SETTINGS ############
    levels  = [64,128,256]
    nodes   = [1,2,4]
    procs   = [24,48,96]
    wtime_h = [0,2,10]
    wtime_m = [30,0,0]
    dd_fac  = [1,1,1]                    # on every level, sigma = fac * hx
    predict = [0,0,1]
    gvf     = [0.0,0.9,0.9]              # ignored for C0_RANKED
    rho_default = 8;
    k_default   = 0;
    betap_prev  = 1E-4;
    p_prev      = "";
    submit      = True;
    pid_prev    = 0;
    obs_masks   = []
    gaussian_selection_mode = "PHI"; # alternatives: {"PHI", "C0", "C0_RANKED"}
    data_thresh = [1E-1, 1E-4, 1E-4] if (gaussian_selection_mode == "PHI") else [1E-1, 1E-4, 1E-4];
    sparsity_lvl_per_component = 5;
    ls_max_func_evals  = [10, 10, 10];
    invert_diffusivity = [1,1,1];
    # #################################

    os.environ['DIR_SUFFIX'] = "{0:1.1f}".format(args.obs_lambda);
    obs_dir = "obs-{0:1.1f}".format(args.obs_lambda) + "/";

    # create output folder
    output_path = args.results_directory
    if not os.path.exists(output_path):
        print("results folder doesn't exist, creating one!\n");
        os.mkdir(output_path);
    # python command
    pythoncmd = "python ";
    if args.compute_cluster == "stampede2" or args.compute_cluster == "local":
        pythoncmd = "python3 ";
    # create input folder
    input_folder = os.path.join(output_path, 'input');
    if not os.path.exists(input_folder):
        os.mkdir(input_folder);

    #   ------------------------------------------------------------------------
    #   - read atlas segmented image and create probability maps
    #   - read segmented patient or patient probability maps and resize them
    cmd_preproc = pythoncmd + basedir + '/scripts/preprocess.py -atlas_image_path ' + args.atlas_image_path  + ' -patient_image_path ' + args.patient_image_path + ' -output_path ' + input_folder + ' -N ' + str(args.resolution) + ' -patient_labels ' + args.patient_segmentation_labels + ' -atlas_labels ' + args.atlas_segmentation_labels;
    if args.use_atlas_segmentation:
        cmd_preproc += ' --use_atlas_segmentation'
    if args.use_patient_segmentation:
        cmd_preproc += ' --use_patient_segmentation'

    # compute observation operartor mask (lambda)
    cmd_obs =  "\n\n# observation mask\n" + pythoncmd + basedir + '/scripts/utils.py -compute_observation_mask --obs_lambda ' + str(args.obs_lambda) + ' -output_path ' + input_folder
    cmd_obs_resample64  = "\n" + pythoncmd + basedir + '/scripts/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'obs_mask.nc'          + ' --name_new ' + 'obs_mask_nx64.nc'          + ' --N_old 256 --N_new 64 &';
    cmd_obs_resample128 = "\n" + pythoncmd + basedir + '/scripts/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'obs_mask.nc'          + ' --name_new ' + 'obs_mask_nx128.nc'         + ' --N_old 256 --N_new 128 &';
    cmd_obs_rename    = "mv " + os.path.join(input_folder, 'obs_mask.nc')          + " " + os.path.join(input_folder, 'obs_mask_nx' + str(256) + '.nc') + " \n"
    # compute observation operator mask (lambda \in {1, 0.8, 0.6, 0.4, 0.2, 0.1})
    if args.vary_obs_lambda:
        cmd_obs =  "\n\n# observation mask\n";
        cmd_obs_resample64 = ""
        cmd_obs_resample128 = ""
        for lambda_o in [1, 0.8, 0.6, 0.4, 0.2, 0.1]:
            name = "obs_mask" + '_lbd-'+str(lambda_o);
            obs_masks.append(name);
            cmd_obs             += "\n" + pythoncmd + basedir + '/scripts/utils.py -compute_observation_mask --obs_lambda ' + str(lambda_o) + ' -output_path ' + input_folder + ' --suffix ' + '_lbd-'+str(lambda_o)
            cmd_obs_resample64  += "\n" + pythoncmd + basedir + '/scripts/utils.py -resample -output_path ' + input_folder + ' --name_old ' + name + '.nc'          + ' --name_new ' + name + '_nx64.nc'          + ' --N_old 256 --N_new 64 &';
            cmd_obs_resample128 += "\n" + pythoncmd + basedir + '/scripts/utils.py -resample -output_path ' + input_folder + ' --name_old ' + name + '.nc'          + ' --name_new ' + name + '_nx128.nc'         + ' --N_old 256 --N_new 128 &';
            cmd_obs_rename      += "mv " + os.path.join(input_folder, name + '.nc')          + " " + os.path.join(input_folder, name + '_nx' + str(256) + '.nc') + " \n"


    cmd_preproc += cmd_obs;

    #   - resample 256 --> 64
    cmd_preproc +=  "\n\n# resample\n";
    cmd_preproc +=           pythoncmd + basedir + '/scripts/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_csf.nc'   + ' --name_new ' + 'patient_nx64_seg_csf.nc'   + ' --N_old 256 --N_new 64 &';
    cmd_preproc +=    "\n" + pythoncmd + basedir + '/scripts/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_gm.nc'    + ' --name_new ' + 'patient_nx64_seg_gm.nc'    + ' --N_old 256 --N_new 64 &';
    cmd_preproc +=    "\n" + pythoncmd + basedir + '/scripts/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_wm_wt.nc' + ' --name_new ' + 'patient_nx64_seg_wm_wt.nc' + ' --N_old 256 --N_new 64 &';
    cmd_preproc +=    "\n" + pythoncmd + basedir + '/scripts/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_tc.nc'    + ' --name_new ' + 'patient_nx64_seg_tc.nc'    + ' --N_old 256 --N_new 64 &';
    cmd_preproc += cmd_obs_resample64;
    #   - resample 256 --> 128
    cmd_preproc +=  "\n\n" + pythoncmd + basedir + '/scripts/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_csf.nc'   + ' --name_new ' + 'patient_nx128_seg_csf.nc'   + ' --N_old 256 --N_new 128 &';
    cmd_preproc +=    "\n" + pythoncmd + basedir + '/scripts/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_gm.nc'    + ' --name_new ' + 'patient_nx128_seg_gm.nc'    + ' --N_old 256 --N_new 128 &';
    cmd_preproc +=    "\n" + pythoncmd + basedir + '/scripts/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_wm_wt.nc' + ' --name_new ' + 'patient_nx128_seg_wm_wt.nc' + ' --N_old 256 --N_new 128 &';
    cmd_preproc +=    "\n" + pythoncmd + basedir + '/scripts/utils.py -resample -output_path ' + input_folder + ' --name_old ' + 'patient_seg_tc.nc'    + ' --name_new ' + 'patient_nx128_seg_tc.nc'    + ' --N_old 256 --N_new 128 &';
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
    t_params['code_path']    = os.path.join(basedir, '3rdparty/pglistr_tumor');

    cmd_lvl = "\n\n# generate maps, resample\n" + cmd_preproc + "\nwait\n\n# rename\n" + cmd_rename + "\n\n";
    cmd     += cmd_lvl;

    # loop over levels
    for level, sigma_fac, n, p, h, m, pred, gvf_, d_thresh, ls_max_func, diff_inv in zip(levels, dd_fac, nodes, procs, wtime_h, wtime_m, predict, gvf, data_thresh, ls_max_func_evals, invert_diffusivity):

        res_dir = os.path.join(tumor_out_path, 'nx' + str(level) + "/");
        res_dir_out = os.path.join(res_dir, obs_dir)
        res_dir_prev = os.path.join(tumor_out_path, 'nx' + str(int(level/2)) + "/" + obs_dir);
        if not os.path.exists(res_dir):
            os.makedirs(res_dir, exist_ok=True);
        inp_dir = os.path.join(res_dir, 'init');
        if not os.path.exists(inp_dir):
            os.mkdir(inp_dir);

        cmd += "\n\n### ----------------------------------------- ###\n"
        cmd += "###   LEVEL nx=" + str(level) + ", sigma=2pi/" + str(int(level/sigma_fac)) + "   ###\n"

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

        if level == 64:
            cmd_symlink +=  "ln -sf " + "../../../input/patient_nx" + str(level) + "_seg_tc.nc"     " support_data.nc \n";
        else:
            level_prev = int(level/2);
            cmd_symlink += pythoncmd + basedir + '/scripts/utils.py -resample -output_path ' + os.path.join(tumor_out_path, 'nx' + str(level_prev) + "/" + obs_dir) + ' --name_old ' + 'c0Recon.nc'           + ' --name_new ' + 'c0Recon_nx'+ str(level)         + '.nc' + ' --N_old ' + str(level_prev) + ' --N_new ' + str(level) + ' \n';
            cmd_symlink +=  "ln -sf " + "../../nx"+ str(level_prev)+ "/" + obs_dir +"/c0Recon_nx"+str(level)+".nc"   " support_data.nc \n";
            cmd_symlink += pythoncmd + basedir + '/scripts/utils.py -resample -output_path ' + os.path.join(tumor_out_path, 'nx' + str(level_prev) + "/" + obs_dir) + ' --name_old ' + 'phiSupportFinal.nc'   + ' --name_new ' + 'phiSupportFinal_nx'+ str(level) + '.nc' + ' --N_old ' + str(level_prev) + ' --N_new ' + str(level) + ' \n';
            cmd_symlink +=  "ln -sf " + "../../nx"+ str(level_prev)+ "/" + obs_dir +"/phiSupportFinal_nx"+str(level)+".nc"   " support_data_phi.nc \n";
        cmd_symlink  +=  "cd ${PWDO} \n"

        # compute connecte components of target data
        cmd_concomp  = pythoncmd + basedir + '/scripts/utils.py -concomp_data  -input_path ' +  os.path.join(tumor_out_path, 'nx' + str(level)) + ' -output_path ' +  input_folder + ' --obs_lambda ' + str(args.obs_lambda);
        cmd_concomp += " -select_gaussians " + " --sigma " + str(sigma_fac) + " \n" if (gaussian_selection_mode == 'C0_RANKED' and level > 64) else " \n";
        cmd_concomp +=  "PWDO=${PWD} \n"
        cmd_concomp +=  "cd " + str(inp_dir) + "\n"
        cmd_concomp +=  "ln -sf " + "../../../input/data_comps_nx" + str(level)  + ".nc"    " data_comps.nc \n";
        cmd_concomp +=  "cd ${PWDO} \n"

        #   ------------------------------------------------------------------------
        #   - extract reconstructed rho and k
        if level > 64:
            cmd = ""
            cmd_extractrhok =  "\n\n# extract reconstructed rho, k\n";
            cmd_extractrhok +=  pythoncmd + basedir + '/scripts/utils.py -extract -output_path ' + res_dir_prev + ' --rho_fac 1' + " \n"
            cmd_extractrhok += "source " + os.path.join(res_dir_prev, 'env_rhok.sh') + "\n"

        # tumor settings for current level
        t_params['num_nodes']             = n;
        t_params['mpi_pernode']           = p;
        t_params['wtime_h']               = h;
        t_params['wtime_m']               = m
        t_params['ibrun_man']             = (level <= 64);
        t_params['results_path']          = res_dir_out;
        t_params['N']                     = level;
        t_params['grad_tol']              = args.opttol;
        t_params['sparsity_lvl']          = sparsity_lvl_per_component;
        t_params['multilevel']            = 1;
        t_params['ls_max_func_evals']     = ls_max_func;
        t_params['diffusivity_inversion'] = diff_inv;
        t_params['data_thres']            = d_thresh;
        t_params['rho_inv']               = rho_default if (level == 64) else '${RHO_INIT}';
        t_params['k_inv']                 = k_default   if (level == 64) else '${K_INIT}';
        t_params['gist_maxit']            = 4           if (level == 64) else 2;
        t_params['linesearchtype']        = 'mt'    if (level == 64) else 'armijo';
        t_params['newton_maxit']          = 30 if (level == 256) else 50;
        t_params['gvf']                   = gvf_;
        t_params['beta']                  = betap_prev;
        t_params['dd_fac']                = sigma_fac;
        t_params['predict_flag']          = pred;
        t_params['csf_path']              = os.path.join(inp_dir, 'patient_seg_csf.nc');
        t_params['gm_path']               = os.path.join(inp_dir, 'patient_seg_gm.nc');
        t_params['wm_path']               = os.path.join(inp_dir, 'patient_seg_wm_wt.nc');
        t_params['data_path']             = os.path.join(inp_dir, 'patient_seg_tc.nc');
        if gaussian_selection_mode == "C0" or level == 64:
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

        #   ------------------------------------------------------------------------
        #    - get command line for tumor inversion
        cmdline_tumor, err = TumorParams.getTumorRunCmd(t_params)
        cmdline_tumor += " &> " + t_params["results_path"] + "tumor_solver_log_nx"+str(level)+".txt";
        if err:
            warnings.warn("Error in tumor parameters\n");
            quit();
        cmd     += cmd_lvl;
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
        cmd_postproc  = pythoncmd + basedir + '/scripts/postprocess.py -input_path ' + args.results_directory + ' -reference_image_path ' + args.patient_image_path + " -patient_labels " +  args.patient_segmentation_labels
        cmd_postproc += " -rdir obs ";
        cmd_postproc += " -convert_images -gridcont ";
        cmd_postproc += " -compute_tumor_stats ";
        cmd_postproc += " -analyze_concomps ";
        cmd_postproc += " -generate_slices " + "\n";

        # cmd_postproc += " -compute_dice_healthy " + "\n";

        if level == 256:
            cmd += "\n# postproc, compute dice\n" + cmd_postproc + "\n\n";

        opt = {}
        opt['compute_sys']  = args.compute_cluster;
        opt['output_dir']  = res_dir_out;
        opt['input_dir']   = inp_dir;
        opt['num_nodes']   = n;
        opt['mpi_pernode'] = p;
        opt['wtime_h']     = h;
        opt['wtime_m']     = m;

        job_file = createJobsubFile(cmd, opt, level);
        if submit:
            if level == 64:
                if args.compute_cluster == 'hazelhen':
                  process = subprocess.check_output(['qsub',job_file]).strip();
                else:
                  process = subprocess.check_output(['sbatch',job_file]).strip();
                print(process)
            else:
                if args.compute_cluster == 'hazelhen':
                  process = subprocess.check_output(['qsub', '-W depend=afterok:'+str(pid_prev), job_file]).strip();
                else:
                  process = subprocess.check_output(['sbatch', '--dependency=afterok:'+str(pid_prev), job_file]).strip();
                print(process)
            print(str(process).split(".")[0])
            pid_prev = int(str(process,'utf-8').split(".")[0])




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
    r_args.add_argument ('-cluster',        '--compute_cluster',             type = str, help = 'compute cluster name for creation of job script (ex. stampede2, hazelhen, cbica etc)', required=True);
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
    parser.add_argument (                   '--opttol',                      type = float, default = 1e-5,   help = 'optimizer tolerance (gradient reduction)');
    parser.add_argument (                   '--multiple_patients',           action='store_true', help = 'process multiple patients, -patient_path should be the base directory containing patient folders which contain patient image(s).');
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
