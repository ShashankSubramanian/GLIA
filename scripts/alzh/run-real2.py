import os, sys, warnings, argparse, subprocess
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../common/'))
import TumorParams
from shutil import copyfile
import preprocess as prep
import nibabel as nib
import numpy as np
import imageTools as imgtools
import nibabel as nib
import file_io as fio
import claire


###
### ------------------------------------------------------------------------ ###
def createJobsubFile(cmd, opt, level):
    """
    @short - create job submission file on tacc systems
    """
    # construct bash file name
    bash_filename = "job.sh";

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
        bash_file.write("#SBATCH -J alzh\n");
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
            if opt['gpu']:
                bash_file.write('#SBATCH -p rtx\n');
            else:
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
        bash_file.write("#SBATCH -o " + os.path.join(opt['output_dir'], "alszh-run.out ") + "\n");
        bash_file.write("#SBATCH -t " + str(opt['wtime_h']) + ":" + str(opt['wtime_m']) + ":00\n");
        #bash_file.write("#SBATCH --mail-user=nutexas.edu\n");
        #bash_file.write("#SBATCH --mail-type=fail\n");
        if opt['compute_sys'] != 'frontera':
            bash_file.write("#SBATCH -A PADAS\n");


    bash_file.write("\n\n");
    bash_file.write("source ~/.bashrc\n");
    if opt['gpu']:
        bash_file.write("\nmodule load cuda")
        bash_file.write("\nmodule load cudnn")
        bash_file.write("\nmodule load nccl")
        bash_file.write("\nmodule load petsc/3.11-rtx")
        bash_file.write("\nexport ACCFFT_DIR=/work/04678/scheufks/frontera/libs/accfft/build_gpu/")
        bash_file.write("\nexport ACCFFT_LIB=${ACCFFT_DIR}/lib/")
        bash_file.write("\nexport ACCFFT_INC=${ACCFFT_DIR}/include/")
        bash_file.write("\nexport CUDA_DIR=${TACC_CUDA_DIR}/")

    bash_file.write("\nexport OMP_NUM_THREADS=1\n");
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
def set_params(basedir, args, case_dir='d_nc', gpu=False):
    
    TF = { 
       '022_S_6013' :  {'t1' : 0.40,  't2' : 0.1, 't02' : 0.2},     
       '023_S_1190' :  {'t1' : 0.47,  't2' : 0.1, 't02' : 0.2},     
       '127_S_4301' :  {'t1' : 0.47,  't2' : 0.63, 't02' : 1.63},   
       '127_S_2234' :  {'t1' : 0.66,  't2' : 0.63, 't02' : 1.63},     
       '012_S_6073' :  {'t1' : 0.35,  't2' : 0.63, 't02' : 1.63},     
       '033_S_4179' :  {'t1' : 0.37,  't2' : 0.63, 't02' : 1.63},     
       '941_S_4036' :  {'t1' : 0.46,  't2' : 0.63, 't02' : 1.63},     
       '032_S_5289' :  {'t1' : 0.74,  't2' : 0.63, 't02' : 1.63},     
       '035_S_4114' :  {'t1' : 0.42,  't2' : 0.63, 't02' : 1.63},     
    }
    reg_dict = {'127_S_4301' : False, '127_S_2234' : False, '022_S_6013' : True, '023_S_1190' : True, '012_S_6073' : False, '033_S_4179' : False, '941_S_4036' : True,
    '032_S_5289' : False, '035_S_4114' : True }

    case = str(case_dir.split('CASE_')[-1])
   
    th_dict = {0.6 : 'th06', 0.4 : 'th04', 0.2 : 'th02'}

    basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
    submit             = True;
    ###########
    opttol  = 1E-2;
    scale   = 1E-1;
    rho_inv = 4;
    k_inv   = 1E-2/scale;
    dt_inv  = 0.01;
    nt_inv  = int(TF[case]['t1']/dt_inv);
    k_lb    = 1E-4/scale;
    k_ub    = 1/scale;
    rho_lb  = 1E-4;
    rho_ub  = 15;
    model   = 2;
    obs_th_1= 0.4;
    obs_th_0= 0.4;
    smooth_fac    = 1.5;
    smooth_fac_c0 = 1.5;
    pred_t0 = TF[case]['t1']
    pred_t1 = -1; #TF[case]['t02']
    pred_t2 = -1; #TF[case]['t2']
    pre_adv_time = -1;
    b_name  = 'inverse_gpu_scale_real'
    #d1      = 'time_point_1_tau_'+th_dict[obs_th_1]+'.nc'
    #d0      = 'time_point_0_tau_'+th_dict[obs_th_0]+'.nc'
    d1      = 'time_point_1_tau.nc'
    d0      = 'time_point_0_tau.nc'
    solver  = 'QN'
    prefix  = 'time_point_0'
    prefix_p= 'time_point_1'
    adv     = False #reg_dict[case];
    vx1     = 'reg-1-0_more_accurate/velocity-field-x1.nc'
    vx2     = 'reg-1-0_more_accurate/velocity-field-x2.nc'
    vx3     = 'reg-1-0_more_accurate/velocity-field-x3.nc'
    vx1_p   = ''#'reg-2-1/velocity-field-x1.nc'
    vx2_p   = ''#'reg-2-1/velocity-field-x2.nc'
    vx3_p   = ''#'reg-2-1/velocity-field-x3.nc'
    obs_mask_path = ''
    ###########
    
    # make paths
    inp_dir = os.path.join(os.path.join(args.results_directory, case_dir), 'data');
    dat_dir = os.path.join(os.path.join(args.results_directory, case_dir), 'data');
    
    d1_true  = os.path.join(inp_dir, 'time_point_1_tau.nc')
    d12_true = os.path.join(inp_dir, 'time_point_2_tau.nc')
    d15_true = os.path.join(inp_dir, 'time_point_2_tau.nc')

    # define results path
    if not adv:
        res_dir = os.path.join(os.path.join(args.results_directory, case_dir),'_reg-sensitivity_inv-obs-th-'+str(obs_th_1)+'-smc0-'+str(smooth_fac_c0)+'-iguess[r-'+str(rho_inv)+'-k-'+str(k_inv*scale)+']-rho-lb-'+str(rho_lb)+'-fd-lbfgs-10-bounds-scale-tol'+str(opttol)+'/');
    else:
        res_dir = os.path.join(os.path.join(args.results_directory,case_dir),'_reg-sensitivity_inv-obs-adv-obs-th-'+str(obs_th_1)+'-smc0-'+str(smooth_fac_c0)+'-iguess[r-'+str(rho_inv)+'-k-'+str(k_inv*scale)+']-rho-lb-'+str(rho_lb)+'-fd-lbfgs-10-bounds-scale-tol-'+str(opttol)+'/');
    
    opt = {}
    opt['compute_sys']  = args.compute_cluster;
    opt['gpu']          = gpu; 
    opt['output_dir']   = res_dir;
    opt['input_dir']    = inp_dir;
    opt['num_nodes']    = 1 if gpu else 2;
    opt['mpi_pernode']  = 1 if gpu else 96;
    opt['wtime_h']      = 6;
    opt['wtime_m']      = 0;

    output_path = args.results_directory
    if not os.path.exists(res_dir):
        print("results folder doesn't exist, creating one!\n");
        os.mkdir(res_dir);
    # python command
    pythoncmd = "python ";
    if args.compute_cluster in ["stampede2", "frontera", "local"]:
        pythoncmd = "python3 ";
    
    t_params = {}
    t_params['binary_name']           = b_name;
    t_params['code_path']             = os.path.join(basedir, '..'); 
    t_params['compute_sys']           = args.compute_cluster;
    t_params['num_nodes']             = opt['num_nodes'];
    t_params['mpi_pernode']           = opt['mpi_pernode'];
    t_params['wtime_h']               = opt['wtime_h'];
    t_params['wtime_m']               = opt['wtime_m'];
    t_params['ibrun_man']             = False;
    t_params['results_path']          = res_dir;
    t_params['N']                     = 256;
    t_params['solvertype']            = solver;
    t_params['grad_tol']              = opttol;
    t_params['sparsity_lvl']          = 5;
    t_params['multilevel']            = 0;
    t_params['model']                 = model; 
    t_params['obs_thres']             = obs_th_1;
    t_params['obs_thres_0']           = obs_th_0;
    t_params['solve_rho_k']           = 1;
    t_params['inject_solution']       = 0;
    t_params['two_snapshot']          = 1;
    t_params['low_res_data']          = 0;
    t_params['pre_reacdiff_solve']    = 0;
    t_params['create_synthetic']      = 0;
    t_params['ls_max_func_evals']     = 10;
    t_params['diffusivity_inversion'] = 1;
    t_params['data_thres']            = 1E-4;
    t_params['rho_inv']               = rho_inv;
    t_params['k_inv']                 = k_inv;
    t_params['nt_inv']                = nt_inv;
    t_params['dt_inv']                = dt_inv;
    t_params['gist_maxit']            = 2;
    t_params['linesearchtype']        = 'armijo';
    t_params['newton_maxit']          = 50;
    t_params['gvf']                   = 0.9;
    t_params['beta']                  = 1E-4;
    t_params['lower_bound_kappa']     = k_lb;
    t_params['upper_bound_kappa']     = k_ub;
    t_params['lower_bound_rho']       = rho_lb;
    t_params['upper_bound_rho']       = rho_ub;
    t_params['dd_fac']                = 1;
    t_params['predict_flag']          = 1;
    t_params['pred_t0']               = pred_t0;
    t_params['pred_t1']               = pred_t1;
    t_params['pred_t2']               = pred_t2;
    t_params['pre_adv_time']          = pre_adv_time;
    t_params['csf_path']              = os.path.join(inp_dir, prefix + '_seg_csf.nc');
    t_params['gm_path']               = os.path.join(inp_dir, prefix + '_seg_gm.nc');
    t_params['wm_path']               = os.path.join(inp_dir, prefix + '_seg_wm.nc');
    t_params['csf_pred_path']         = os.path.join(inp_dir, prefix_p + '_seg_csf.nc');
    t_params['gm_pred_path']          = os.path.join(inp_dir, prefix_p + '_seg_gm.nc');
    t_params['wm_pred_path']          = os.path.join(inp_dir, prefix_p + '_seg_wm.nc');
    t_params['data_path_t1']          = os.path.join(dat_dir, d1);
    t_params['data_path_t0']          = os.path.join(dat_dir, d0);
    t_params['obs_mask_path']         = obs_mask_path;
    t_params['data_path_mri']         = ""
    t_params['data_path_pred_t0']     = d1_true;
    t_params['data_path_pred_t1']     = d12_true;
    t_params['data_path_pred_t2']     = d15_true;
    t_params['forward_flag']          = 0
    t_params['smooth_f']              = smooth_fac;
    t_params['smooth_f_c0']           = smooth_fac_c0;
    t_params['model']                 = model;
    t_params['gaussian_cm_path']      = os.path.join(dat_dir, 'phi-mesh-p-syn.txt');
    t_params['pvec_path']             = os.path.join(dat_dir, 'p-rec-p-syn.txt');
    t_params['velocity_x1']           = "" if not adv else os.path.join(os.path.join(args.results_directory, case_dir), vx1);
    t_params['velocity_x2']           = "" if not adv else os.path.join(os.path.join(args.results_directory, case_dir), vx2);
    t_params['velocity_x3']           = "" if not adv else os.path.join(os.path.join(args.results_directory, case_dir), vx3);
    t_params['velocity_x1_p']         = "" if not adv else os.path.join(os.path.join(args.results_directory, case_dir), vx1_p);
    t_params['velocity_x2_p']         = "" if not adv else os.path.join(os.path.join(args.results_directory, case_dir), vx2_p);
    t_params['velocity_x3_p']         = "" if not adv else os.path.join(os.path.join(args.results_directory, case_dir), vx3_p);
 
    #   ------------------------------------------------------------------------
    #    - get command line for tumor inversion
    cmd = ''
    cmdline_tumor, err = TumorParams.getTumorRunCmd(t_params)
    cmdline_tumor += " &> " + os.path.join(t_params["results_path"], "tumor_solver_log.txt");
    if err:
        warnings.warn("Error in tumor parameters\n");
        quit();
    cmd += cmdline_tumor + "\n\n";
 
    ONEJOB = "\n ### ALZH INVERSION ### \n\n" + cmd
 
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
    r_args.add_argument ('-cluster',        '--compute_cluster',             type = str, help = 'compute cluster name for creation of job script (ex. stampede2, frontera, hazelhen, cbica etc)', required=True);
    parser.add_argument ('-x',              '--results_directory',           type = str, default = os.path.join(basedir, 'results/'), help = 'path to results directory');
    parser.add_argument (                   '--tumor_code_dir',                    type = str, help = 'path to tumor solver code directory')
    args = parser.parse_args();

    #case_dir = 'CASE_022_S_6013'   
    #case_dir = 'CASE_127_S_2234'   
    #case_dir = 'CASE_127_S_4301'   
    #case_dir = 'CASE_023_S_1190'   
    #case_dir = 'CASE_012_S_6073'   
    #case_dir = 'CASE_941_S_4036'   
    #case_dir = 'CASE_033_S_4179'   
    #case_dir = 'CASE_032_S_5289'   
    #case_dir = 'CASE_035_S_4114'  
    #CASES = ['CASE_022_S_6013','CASE_127_S_2234', 'CASE_127_S_4301', 'CASE_023_S_1190', 'CASE_012_S_6073', 'CASE_941_S_4036', 'CASE_033_S_4179', 'CASE_032_S_5289','CASE_035_S_4114']   
    CASES = ['CASE_022_S_6013', 'CASE_023_S_1190', 'CASE_941_S_4036','CASE_035_S_4114']   
    for case_dir in CASES:
        set_params(basedir, args, case_dir, gpu=True);
