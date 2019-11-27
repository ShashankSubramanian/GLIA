# sets up the parameters for running CLAIRE
import os,warnings,subprocess
from sys import exit


##############################################################################
# function to set default parameters for diffeomorphic registration
##############################################################################
def set_parameters(param, basedir, tumor_output_dir):

    param.setdefault('N', 256);
    param.setdefault('opttol', 1E-2);
    param.setdefault('regnorm', 'h1s-div');
    param.setdefault('betaw', 1e-4);
    param.setdefault('maxit', 25);
    param.setdefault('krylovmaxit', 50);
    param.setdefault('num_nodes', 2);
    param.setdefault('mpi_pernode', 48);
    param.setdefault('jbound',5e-2);
    param.setdefault('sigma', 0)
    param.setdefault('train', 'reduce')
    param.setdefault('compute_sys', 'frontera')
    param.setdefault('wtime_h', '24');
    param.setdefault('wtime_m', '00');
    param.setdefault('format', 'nifti');
    param.setdefault('extension',  '.nii.gz');
    param.setdefault('num_images', 3);
    param.setdefault('tumor_inversion_space', 'patient');
    param.setdefault('precombine_edema_healthy', True)
    param.setdefault('objwts', [0.5,0.25,0.25])
    param.setdefault('output_prefix', None)
    param.setdefault('output_dir', os.path.join(basedir,'registration'))
    param.setdefault('data_dir', os.path.join(basedir, 'input'))
    if not os.path.exists(param['output_dir']):
        os.makedirs(param['output_dir'])
    if 'reg_code_dir' not in param:
        param.setdefault('reg_code_dir', '/work/04678/scheufks/frontera/code/tumor-tools/3rdparty/claire-dev/')    
    param.setdefault('tumor_output_dir', tumor_output_dir)

def createCmdLineReg(param):
        '''
        creates command line arguments given the registration options given by the user
        '''
        cmd = '#perform patient(template) to atlas(reference) diffeomorphic registration\n'
        cmd += 'if [ ! -f "$OUTPUT_DIR/velocity-field-x1.nii.gz" ]; then\n'
        # add executable env based on which cluster you are running
        claire_bin = '$CLAIRE_BDIR/claire';
        if param['compute_sys'] == 'lonestar':
            cmd += 'ibrun ' + claire_bin;
        if param['compute_sys'] == 'stampede2':
            cmd += 'ibrun ' + claire_bin;
        if param['compute_sys'] == 'frontera':
            cmd += 'ibrun ' + claire_bin;
        elif param['compute_sys'] == 'maverick':
            cmd += 'ibrun ' + claire_bin;
        elif param['compute_sys'] == 'hazelhen':
            cmd += "aprun -n " + str(param['mpi_pernode']) + " -N 12 " + claire_bin;
        elif param['compute_sys'] == 'cbica':
            cmd += "mpirun --prefix $OPENMPI -np $NSLOTS $MACHINES --mca plm_base_verbose 1 --mca orte_forward_job_control 1 " + claire_bin;
        else:
            cmd += 'mpirun ' + '-np ' + str(param['mpi_pernode']);
            cmd += ' ' + claire_bin;

        # output directory
        if param['output_prefix'] is not None:
            cmd += ' -x ' + "$OUTPUT_DIR/" + param['output_prefix'];
        else:
            cmd += ' -x ' + "$OUTPUT_DIR/"

        # if using vecor registration mention the number of components
        if 'num_images' in param:
                cmd += ' -mtc ' + str(param['num_images']) + " $DATA_DIR/patient_csf.nii.gz $DATA_DIR/patient_gm.nii.gz $DATA_DIR/patient_ed_wm.nii.gz"
                cmd += ' -mrc ' + str(param['num_images']) + " $DATA_DIR/atlas_csf.nii.gz $DATA_DIR/atlas_gm.nii.gz $DATA_DIR/atlas_wm.nii.gz"
                cmd += ' -objwts ' + str(param['objwts'][0]) + "," + str(param['objwts'][1]) + "," + str(param['objwts'][2])
                cmd += ' -disablerescaling';

        if param['sigma'] == 0:
            cmd += ' -disablesmoothing';
        else:
            cmd += ' -sigma ' + str(param['sigma'])

        cmd += ' -mask $DATA_DIR/patient_mask.nii.gz'

        cmd += ' -nx ' + str(param['N']);
        cmd += ' -regnorm ' + param['regnorm'];
        cmd += ' -maxit ' + str(param['maxit']);
        cmd += ' -krylovmaxit ' + str(param['krylovmaxit']);
        cmd += ' -beta-div ' + '{0:.6e}'.format(param['betaw']);
        cmd += ' -opttol ' + '{0:.6e}'.format(param['opttol']);
        cmd += ' -train ' + param['train'];
        cmd += ' -jbound ' + str(param['jbound']);
        cmd += ' -format ' + param['format'];

        cmd += ' -velocity';
        cmd += ' -deffield';
        cmd += ' -residual';
        cmd += ' -defmap';
        cmd += ' -detdefgrad';
        cmd += ' -verbosity ' + str(2);

        # logfile
        cmd += ' > ' + '$OUTPUT_DIR/solver_log.txt\n';
        cmd += 'fi\n\n\n'
        return cmd;


def createCmdLineTransport(param, task, labels=None, input_filename=None, output_filename=None, direction=None):
        '''
        labels is a dictionary comma separated input given at the command line
        '''

        cmd = "#Transport Image\n"
        if output_filename is not None:
            cmd += 'if [ ! -f "'+output_filename +'" ]; then\n'

        clairetools_bin = '$CLAIRE_BDIR/clairetools';


        if param['compute_sys'] == 'lonestar':
            cmd += 'ibrun ' + clairetools_bin;
        elif param['compute_sys'] == 'maverick':
            cmd += 'ibrun ' + clairetools_bin;
        elif param['compute_sys'] == 'hazelhen':
            cmd += "aprun 1 " + clairetools_bin;
        elif param['compute_sys'] == 'stampede2':
            cmd += 'ibrun ' + clairetools_bin;
        elif param['compute_sys'] == 'frontera':
            cmd += 'ibrun ' + clairetools_bin;
        elif param['compute_sys'] == 'cbica':
            cmd += "mpirun --prefix $OPENMPI -np $NSLOTS $MACHINES --mca plm_base_verbose 1 --mca orte_forward_job_control 1 " + clairetools_bin;
        else:
            cmd += 'mpirun ' + '-np ' +  str(param['mpi_pernode']);
            cmd += ' ' + clairetools_bin;

        cmd += ' -' + task

        if task=='tlabelmap' and labels is not None:
            cmd += ' -labels ' + labels;

        if task=='tlabelmap' or task=='deformimage':
            cmd += ' -ifile ' + input_filename
            cmd += ' -xfile ' + output_filename

        if direction is not None:
            cmd += ' -r2t'
        cmd += ' -nx ' + str(param['N'])
        if output_filename is None:
            cmd += ' -x $OUTPUT_DIR/'
        cmd += ' -v1 $OUTPUT_DIR/velocity-field-x1' + param['extension'];
        cmd += ' -v2 $OUTPUT_DIR/velocity-field-x2' + param['extension'];
        cmd += ' -v3 $OUTPUT_DIR/velocity-field-x3' + param['extension'];
        cmd += ' -verbosity ' + str(2);
        cmd += ' > ' + '$OUTPUT_DIR/transport_image_log.txt\n';
        if output_filename is not None:
            cmd += 'fi\n'
        return cmd

##############################################################################
#
##############################################################################
def create_cmd_file(cmd,param, create_bash=False):
    # construct bash file name
    bash_filename = "claire_job.sh";

    bash_filename = os.path.join(param['output_dir'], bash_filename);

    # create bash file
    bash_file = open(bash_filename,'w');

    # header
    if create_bash:
        bash_file.write("#!/bin/bash\n\n");
    bash_file.write("#### define paths\n");

    if 'reg_code_dir' in param:
            bash_file.write("CLAIRE_BDIR=" + param["reg_code_dir"] + "/bin\n");
            bash_file.write("DATA_DIR=" + param['data_dir'] + "\n");
            bash_file.write("OUTPUT_DIR=" + param['output_dir'] + "\n");
            bash_file.write("TUMOR_DIR=" + param['tumor_output_dir'] + "\n");
    else:
            warnings.warn("directory for claire needs to be set");
            quit();

    bash_file.write("\n\n"+cmd+"\n");

    # write out done
    bash_file.close();
    # make it executable
    subprocess.call(['chmod','a+x',bash_filename]);
    return bash_filename

##############################################################################
# function to create job submission file for tacc systems
##############################################################################
def createJobsubFile(cmd,param,submit=False):
    # construct bash file name
    bash_filename = "job-submission.sh";

    bash_filename = os.path.join(param['output_dir'], bash_filename);

    # create bash file
    bash_file = open(bash_filename,'w');

    # header
    bash_file.write("#!/bin/bash\n\n");
    bash_file.write("#### sbatch parameters\n");
    bash_file.write("#SBATCH -J claire\n");
    if param['compute_sys'] == 'lonestar':
        bash_file.write("#SBATCH -p normal\n");
        param['num_nodes'] = 2;
        bash_file.write("#SBATCH -N " + str(param['num_nodes']) + "\n");

    elif param['compute_sys'] == 'stampede2':
        bash_file.write('#SBATCH -p skx-normal\n');
        param['num_nodes'] = 3;
        bash_file.write("#SBATCH -N " + str(param['num_nodes']) + "\n");

    elif param['compute_sys'] == 'frontera':
        bash_file.write('#SBATCH -p normal\n');
        param['num_nodes'] = 2;
        bash_file.write("#SBATCH -N " + str(param['num_nodes']) + "\n");
        bash_file.write("#SBATCH -A FTA-Biros\n");

    elif param['compute_sys'] == 'local':
        bash_file.write('#SBATCH -p rebels\n');
        param['num_nodes'] = 1;
        bash_file.write("#SBATCH -N " + str(param['num_nodes']) + "\n");

    else:
        bash_file.write("#SBATCH -p normal\n");
        param['num_nodes'] = 1;
        bash_file.write("#SBATCH -N " + str(param['num_nodes']) + "\n");

    bash_file.write("#SBATCH -n " + str(param['num_nodes']*param['mpi_pernode']) + "\n");
    bash_file.write("#SBATCH -t " + param['wtime_h'] + ":" + param['wtime_m'] + ":00\n");
    bash_file.write("#SBATCH --mail-user=naveen@ices.utexas.edu\n");
    bash_file.write("#SBATCH --mail-type=fail\n\n\n");

    bash_file.write("\n\n"+cmd+"\n");

    # write out done
    bash_file.close();
    # make it executable
    if submit:
        subprocess.call(['sbatch',bash_filename]);
