
import os, sys, warnings, argparse, subprocess
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../common/'))
from shutil import copyfile
import nibabel as nib
import numpy as np
import nibabel as nib
import claire
import pandas as pd

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

        bash_file.write("#PBS -N reg\n");
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
        bash_file.write("#SBATCH -o " + os.path.join(opt['output_dir'], "job-script-out"+str(level)+".out ") + "\n");
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
def registration(args, basedir, data_dir, patient, atlas):
    '''
    Function to run registration
    '''

    # create output folder
    base_output_dir = args.results_directory
    if not os.path.exists(base_output_dir):
        os.mkdir(base_output_dir);
    if not os.path.exists(data_dir):
        os.mkdir(data_dir);

    reg_dir = os.path.join(os.path.join(base_output_dir, str(patient)+"_to_"+str(atlas),'registration'));
    if not os.path.exists(reg_dir):
        os.mkdir(reg_dir)

    patient_labels = ",".join([x.split('=')[0] for x in args.patient_segmentation_labels.split(',')])
    reg_param = {}
    reg_param['reg_code_dir'] = args.reg_code_dir
    reg_param['compute_sys'] = args.compute_cluster
    claire.set_parameters(reg_param, base_output_dir, "")
    reg_cmd = "#### define paths\n"
    reg_cmd += "CLAIRE_BDIR=" + reg_param["reg_code_dir"] + "/bin\n"
    reg_cmd += "DATA_DIR=" + reg_param['data_dir'] + "\n"
    reg_cmd += "OUTPUT_DIR=" + reg_param['output_dir'] + "\n"
    # reg_cmd += "TUMOR_DIR=" + reg_param['tumor_output_dir'] + "\n"

    # registration command
    reg_cmd += claire.createCmdLineReg(reg_param)
    # transport labels command
    reg_cmd += claire.createCmdLineTransport(reg_param, task='tlabelmap', labels=patient_labels, input_filename="$DATA_DIR/patient_seg.nii.gz", output_filename="$OUTPUT_DIR/patient_seg_in_Aspace.nii.gz")
    # reg_cmd += "\n#convert netcdf to nifti\npython3 " + basedir + "/grid-cont/utils.py -convert_netcdf_to_nii --name_old $TUMOR_DIR/c0Recon.nc --name_new $TUMOR_DIR/c0Recon.nii.gz --reference_image $DATA_DIR/patient_wm.nii.gz\n\n"
    # reg_cmd += "#convert netcdf to nifti\npython3 " + basedir + "/grid-cont/utils.py -convert_netcdf_to_nii --name_old $TUMOR_DIR/cRecon.nc --name_new $TUMOR_DIR/cRecon.nii.gz --reference_image $DATA_DIR/patient_wm.nii.gz\n\n\n"

    # transport c0 command
    # reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$TUMOR_DIR/c0Recon.nii.gz", output_filename="$TUMOR_DIR/c0Recon_in_Aspace.nii.gz")
    # transport c1 command
    # reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$TUMOR_DIR/cRecon.nii.gz", output_filename="$TUMOR_DIR/cRecon_in_Aspace.nii.gz")
    # transport ventricles command
    # reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$DATA_DIR/patient_vt.nii.gz", output_filename="$OUTPUT_DIR/patient_vt_in_Aspace.nii.gz")
    # transport CSF command
    # reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$DATA_DIR/patient_csf_no_vt.nii.gz", output_filename="$OUTPUT_DIR/patient_csf_no_vt_in_Aspace.nii.gz")
    # transport gray matter command
    # reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$DATA_DIR/patient_gm.nii.gz", output_filename="$OUTPUT_DIR/patient_gm_in_Aspace.nii.gz")
    # transport white matter command
    # reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$DATA_DIR/patient_wm.nii.gz", output_filename="$OUTPUT_DIR/patient_wm_in_Aspace.nii.gz")
    # transport tumor command
    # reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$DATA_DIR/patient_tu.nii.gz", output_filename="$OUTPUT_DIR/patient_tu_in_Aspace.nii.gz")
    # transport edema+white matter command
    # reg_cmd += claire.createCmdLineTransport(reg_param, task='deformimage', input_filename="$DATA_DIR/patient_ed_wm.nii.gz", output_filename="$OUTPUT_DIR/patient_ed_wm_in_Aspace.nii.gz")

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
def run_bratsreg(basedir, args, patient, atlas):

    # ########### SETTINGS ############
    patients_per_job   = 10;
    level              = 256;
    nodes              = 5;
    procs              = 256;
    wtime_h            = 16;
    wtime_m            = 0;
    submit             = True;
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



    #   [0] --------------------------------------------------
    #   - setup, create output folder
    output_path = args.results_directory
    if not os.path.exists(output_path):
        print("results folder doesn't exist, creating one!\n");
        os.mkdir(output_path);
    # python command
    pythoncmd = "python ";
    if args.compute_cluster in ["stampede2", "frontera", "local"]:
        pythoncmd = "python3 ";
    # create input folder
    input_folder = os.path.join(output_path, "input")
    if not os.path.exists(input_folder):
        os.mkdir(input_folder);

    #   [1] --------------------------------------------------
    #   - resample atlas segmentation to 256x256x256
    #   - resample patient segmentation into 256x256x256
    #   - for both, create label images and probmap images
    cmd_preproc = pythoncmd + basedir_real + '/../common/preprocess.py -atlas_image_path ' + args.atlas_image_path  + ' -patient_image_path ' + args.patient_image_path + ' -output_path ' + input_folder + ' -N ' + str(args.resolution) + ' -patient_labels ' + args.patient_segmentation_labels + ' -atlas_labels ' + args.atlas_segmentation_labels;
    if args.use_atlas_segmentation:
        cmd_preproc += ' --use_atlas_segmentation'
    if args.use_patient_segmentation:
        cmd_preproc += ' --use_patient_segmentation'
    cmd = "\n# ### generate maps, resample ###\n" + cmd_preproc + "\nwait\n";

    #   [2] --------------------------------------------------
    #   - CLAIRE, register wm, gm, ve  with weights [0.25, 0.25, 0.5] with masking of wt
    #   - transport patient tc, wt to atlas (labelmap and probmap)
    reg_dir = os.path.join(output_path,'registration');
    if not os.path.exists(reg_dir):
        os.mkdir(reg_dir)

    patient_labels = ",".join([x.split('=')[0] for x in args.patient_segmentation_labels.split(',')])
    reg_param = {}
    reg_param['reg_code_dir'] = args.reg_code_dir
    reg_param['compute_sys'] = args.compute_cluster
    claire.set_parameters(reg_param, output_path, "")
    reg_cmd =  "CLAIRE_BDIR=" + reg_param["reg_code_dir"] + "/bin\n"
    reg_cmd += "DATA_DIR=" + reg_param['data_dir'] + "\n"
    reg_cmd += "OUTPUT_DIR=" + reg_param['output_dir'] + "\n"
    # registration command
    reg_cmd += claire.createCmdLineReg(reg_param)

    #   [3] --------------------------------------------------
    #   - impose patient tc, wt into atlas
    #   - transport atlas wm, gm, csf, ve, [wt, tc] to patient
    # transport labels command
    reg_cmd += claire.createCmdLineTransport(reg_param, task='tlabelmap', labels=patient_labels, input_filename="$DATA_DIR/patient_seg.nii.gz", output_filename="$OUTPUT_DIR/patient_seg_in_Aspace.nii.gz")

    cmd += "\n\n###  REGISTRATION (CLAIRE)  ###\n" + reg_cmd + "\n";

    cmd_integrate = pythoncmd + basedir_real + '/../common/utils.py -integrate_tumor_in_atlas --healthy_seg ' + args.atlas_image_path + ' --tumor_seg ' + os.path.join(reg_dir, 'patient_seg_in_Aspace.nii.gz') + ' --out_seg ' + os.path.join(reg_dir, 'atlas_with_warped_tumor.nii.gz') + ' --patient_labels ' + args.patient_segmentation_labels;
    cmd += "# ### integrate tumor ### \n" + cmd_integrate + "\n";

    cmd_transp = claire.createCmdLineTransport(reg_param, task='tlabelmap', labels=patient_labels, input_filename="$OUTPUT_DIR/atlas_with_warped_tumor.nii.gz", output_filename="$OUTPUT_DIR/atlas_with_warped_tumor_in_Pspace.nii.gz", direction='r2t')
    cmd_transp += claire.createCmdLineTransport(reg_param, task='deformimage', labels=patient_labels, input_filename=args.atlas_t1_image, output_filename="$OUTPUT_DIR/atlas_t1_in_Pspace.nii.gz", direction='r2t')
    cmd += "\n# ### transport to patient ### \n" + cmd_transp;


    ONEJOB  += cmd;
    opt = {}
    opt['compute_sys']  = args.compute_cluster;
    opt['output_dir']  = output_path;
    opt['num_nodes']   = nodes;
    opt['mpi_pernode'] = procs;
    opt['wtime_h']     = wtime_h;
    opt['wtime_m']     = wtime_m;

    if batch_end:
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
    basedir_real = os.path.dirname(os.path.realpath(__file__));
    # parse arguments
    parser = argparse.ArgumentParser(description='Process input images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r_args = parser.add_argument_group('required arguments')
    r_args.add_argument ('-patient_path',   '--patient_image_path',          type = str, help = 'path to patient image directory containing the T1,T2,FLAIR,T1CE and segmentation images\n (format- PatientName_{t1,t2,t1ce,flair,segmented}.ext)', required=True)
    r_args.add_argument ('-patient_labels', '--patient_segmentation_labels', type=str,   help = 'comma separated patient segmented image labels. for ex.\n  0=bg,1=nec,2=ed,4=enh,5=wm,6=gm,7=vt,8=csf\n for BRATS type segmentation. DISCLAIMER vt and every extra label mentioned will be merged with csf');
    r_args.add_argument ('-atlas_path',     '--atlas_image_path',            type = str, help = 'path to a segmented atlas image (affinely registered to given patient)', required=True)
    parser.add_argument ('-atlas_t1_image', '--atlas_t1_image',              type = str, help = 'path to t1 atlas image (affinely registered to given patient)')
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
    parser.add_argument (                   '--multiple_patients',           action='store_true', help = 'process multiple patients, -patient_path should be the base directory containing patient folders which contain patient image(s).');
    parser.add_argument (                   '--tumor_code_dir',              type = str, help = 'path to tumor solver code directory')
    parser.add_argument (                   '--reg_code_dir',                type = str, help = 'path to registration solver code directory')
    parser.add_argument ('-csvfile',                                         type = str, help = 'path to CSV file with patient/atlas pairing')
    args = parser.parse_args();

    base_results_dir = args.results_directory;
    base_patient_image_path = args.patient_image_path
    base_atlas_image_path = args.atlas_image_path
    
    MAX_JOBS=50

    if args.csvfile is not None:
        pa_pairs = pd.read_csv(args.csvfile);

    counter = 0;
    for index, row in pa_pairs.iterrows():
        patient = row['bid']
        atlas = row['aid']
        p_path = row['patient_path']
        split = row['set']
        if row['reg'] == True:
             print('..skipping patient/atlas pair [{} / {}]: already registered'.format(patient, atlas))
             continue;
        if counter >= MAX_JOBS*10:
            print("SUBMITTED MAX NUMBER OF JOBS");
            break;
        counter = counter + 1;
        pa_pairs.set_value(index, 'reg', True)
        print('processing patient/atlas pair [{} / {}]'.format(patient, atlas))
        args.patient_image_path = os.path.join(os.path.join(os.path.join(base_patient_image_path, p_path), "affreg2-"+str(atlas)), patient + '_seg_dl_tu_256x256x256_aff2'+str(atlas)+'.nii.gz');
        args.atlas_image_path = os.path.join(os.path.join(base_atlas_image_path, '256x256x256_nii'), atlas + '_segmented.nii.gz');
        args.atlas_t1_image   = os.path.join(os.path.join(base_atlas_image_path, '256x256x256_nii'), atlas + '_cbq_n3.nii.gz');
        args.results_directory = os.path.join(os.path.join(base_results_dir, p_path), "{}_regto_{}".format(patient,atlas));
        # run
        run_bratsreg(basedir, args, patient, atlas);

    pa_pairs.to_csv(args.csvfile, index=False);
