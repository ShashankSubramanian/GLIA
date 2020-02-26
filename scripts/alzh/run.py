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
def set_params(basedir, args):

    basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
    submit             = True;
    ###########
    rho_inv = 0;
    k_inv   = 0;
    nt_inv  = 40;
    dt_inv  = 0.025;
    k_lb    = 1E-4;
    k_ub    = 0.5;
    rho_lb  = 4;
    rho_ub  = 15;
    model   = 2;
    b_name  = 'inverse_adv'
    d1      = 'd_nc/dataBeforeObservation.nc'
    d0      = 'd_nc/c0True.nc'
    solver  = 'QN'
    prefix  = 'atlas'
    vx1     = 'reg/velocity-field-x1.nc'
    vx2     = 'reg/velocity-field-x2.nc'
    vx3     = 'reg/velocity-field-x3.nc'
    ###########
    
    #d1      = 'd_nc/data_t1_64_256.nc'
    #d0      = 'd_nc/data_t0_64_256.nc'
    
    # test02
    #d1      = 'd_nc/data_t1_noise-01-118.nc'
    #d0      = 'd_nc/data_t0_noise-01-45.nc'
    #d1      = 'd_nc/data_t1_noise-02-468.nc'
    #d0      = 'd_nc/data_t0_noise-02-60.nc'
    
    # test01
    #d1      = 'd_nc/data_t1_noise-01-130.nc'
    #d0      = 'd_nc/data_t0_noise-01-130.nc'
    #d1      = 'd_nc/data_t1_noise-02-515.nc'
    #d0      = 'd_nc/data_t0_noise-02-61.nc'

    res_dir = os.path.join(args.results_directory, 'inv-adv-nonoise-iguess[r-'+str(rho_inv)+'-k-'+str(k_inv)+']-fd-lbfgs-3-bounds-tight/');
    inp_dir = os.path.join(args.results_directory, 'data');
    dat_dir = os.path.join(args.results_directory, 'tc');


    opt = {}
    opt['compute_sys']  = args.compute_cluster;
    opt['output_dir']  = res_dir;
    opt['input_dir']   = inp_dir;
    opt['num_nodes']   = 2;
    opt['mpi_pernode'] = 96;
    opt['wtime_h']     = 4;
    opt['wtime_m']     = 0;

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
    t_params['grad_tol']              = 1E-4;
    t_params['sparsity_lvl']          = 5;
    t_params['multilevel']            = 0;
    t_params['model']                 = 2;
    t_params['solve_rho_k']           = 1;
    t_params['inject_solution']       = 0;
    t_params['two_snapshot']          = 1;
    t_params['low_res_data']          = 0;
    t_params['pre_reacdiff_solve']    = 0;
    t_params['create_synthetic']      = 0;
    t_params['ls_max_func_evals']     = 20;
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
    t_params['csf_path']              = os.path.join(inp_dir, prefix + '_seg_csf.nc');
    t_params['gm_path']               = os.path.join(inp_dir, prefix + '_seg_gm.nc');
    t_params['wm_path']               = os.path.join(inp_dir, prefix + '_seg_wm.nc');
    t_params['data_path_t1']          = os.path.join(dat_dir, d1);
    t_params['data_path_t0']          = os.path.join(dat_dir, d0);
    t_params['forward_flag']          = 0
    t_params['smooth_f']              = 1.5
    t_params['model']                 = model;
    t_params['gaussian_cm_path']      = os.path.join(dat_dir, 'phi-mesh-p-syn.txt');
    t_params['pvec_path']             = os.path.join(dat_dir, 'p-rec-p-syn.txt');
    t_params['velocity_x1']           = os.path.join(args.results_directory, vx1);
    t_params['velocity_x2']           = os.path.join(args.results_directory, vx2);
    t_params['velocity_x3']           = os.path.join(args.results_directory, vx3);
 
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

    set_params(basedir, args);
