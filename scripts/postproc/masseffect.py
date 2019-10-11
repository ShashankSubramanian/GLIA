import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
import TumorParams
from netCDF4 import Dataset
from numpy import linalg as la

### Invert in patient-space to get (rho, kappa, c0)
### Register patient to some atlas and transport c0 to this atlas
### Grow some tumors with (rho, kappa, c0-transported) and some gamma
### Register atlas+tumor amd patient and report deformation and final mismatches
### Report best gamma according to mismatch (of ventricles?)

def createNetCDFFile(filename, dimensions, variable):
    file = Dataset(filename, mode='w', format="NETCDF3_CLASSIC");
    x = file.createDimension("x", dimensions[0]);
    y = file.createDimension("y", dimensions[1]);
    z = file.createDimension("z", dimensions[2]);
    data = file.createVariable("data", "f8", ("x","y","z",));
    data[:,:,:] = variable[:,:,:];
    file.close();

def createBashFileHeader(results_path, compute_sys='frontera'):
    bash_filename = results_path + "/coupling_job_submission.sh"
    print("creating job file in ", results_path)

    if compute_sys == 'frontera':
        queue = "normal"
        num_nodes = str(4)
        num_cores = str(128)
    elif compute_sys == 'maverick2':
        queue = "p100"
        num_nodes = str(1)
        num_cores = str(1)
    elif compute_sys == 'stampede2':
        queue = "skx-normal"
        num_nodes = str(6)
        num_cores = str(128)
    else:
        queue = "normal"
        num_nodes = str(1)
        num_cores = str(1)

    bash_file = open(bash_filename, 'w')
    bash_file.write("#!/bin/bash\n\n");
    bash_file.write("#SBATCH -J mass-effect-cpl\n");
    bash_file.write("#SBATCH -o " + results_path + "/coupling_solver_log.txt\n")
    bash_file.write("#SBATCH -p " + queue + "\n")
    bash_file.write("#SBATCH -N " + num_nodes + "\n")
    bash_file.write("#SBATCH -n " + num_cores + "\n")
    bash_file.write("#SBATCH -t 03:00:00\n\n")
    bash_file.write("source ~/.bashrc\n")

    bash_file.write("\n\n")
    bash_file.close()

    return bash_filename

def performRegistration(atlas_image_path, patient_image_path, claire_bin_path, results_path, bash_filename, compute_sys='frontera', mask=True):
    atlas_name = "atlas"
    bash_file = open(bash_filename, 'a')

    ## create registration inputs: mode 1
    bash_file.write("python " + tu_code_path + "/scripts/postproc/postproc-utils.py -m 1 -x " + results_path + " -p " + patient_image_path + " -a " + atlas_image_path + "\n\n")

    ## -defmap for deformation map: not implemented yet in claire
    if mask:
        cmd = "ibrun " + claire_bin_path + "/claire -mrc 3 " + results_path + "/" + atlas_name + "_csf.nii.gz " + results_path + "/" + \
                    atlas_name + "_gm.nii.gz " + results_path + "/" + atlas_name + "_wm.nii.gz " \
                    + "-mtc 3 " + results_path + "/patient_csf.nii.gz " + results_path + "/patient_gm.nii.gz " + results_path + \
                    "/patient_wm.nii.gz -mask " + results_path + "/patient_mask.nii.gz \
                    -nx 256 -train reduce -jbound 5e-2 -regnorm h1s-div -opttol 1e-2 -maxit 50 -krylovmaxit 50 -velocity -detdefgrad -deffield -residual -x " \
                    + results_path + "/"\
                    + " -monitordefgrad -verbosity 1 -disablerescaling -format nifti -sigma 2" + " &> " + results_path + "/registration_log.txt";
    else:
        # cmd = "ibrun " + claire_bin_path + "/claire -mrc 4 " + results_path + "/" + atlas_name + "_csf.nii.gz " + results_path + "/" + atlas_name \
        #             + "_gm.nii.gz " + results_path + "/" + atlas_name + "_wm.nii.gz " + results_path + "/" + atlas_name + "_tu.nii.gz "\
        #             + "-mtc 4 " + results_path + "/patient_csf.nii.gz " + results_path + "/patient_gm.nii.gz " + results_path + "/patient_wm.nii.gz " \
        #             + results_path + "/patient_tu.nii.gz -objwts 0.7,0.1,0.1,0.1 \
        #             -nx 256 -train reduce -jbound 5e-2 -regnorm h1s-div -opttol 1e-2 -maxit 50 -krylovmaxit 50 -velocity -detdefgrad -deffield -residual -x "\
        #             + results_path + "/"\
        #             + " -monitordefgrad -verbosity 1 -disablerescaling -format nifti -sigma 2" + " &> " + results_path + "/registration_log.txt";
        cmd = "ibrun " + claire_bin_path + "/claire -mrc 1 " + results_path + "/" + atlas_name + "_csf.nii.gz " \
                    + "-mtc 1 " + results_path + "/patient_csf.nii.gz " + \
                    "-nx 256 -train reduce -jbound 5e-2 -regnorm h1s-div -opttol 1e-2 -maxit 50 -krylovmaxit 50 -velocity -detdefgrad -deffield -residual -x "\
                    + results_path + "/"\
                    + " -monitordefgrad -verbosity 1 -disablerescaling -format nifti -sigma 2" + " &> " + results_path + "/registration_log.txt";

    bash_file.write(cmd)
    bash_file.write("\n\n")
    bash_file.close()

    # return so that transport can be done immediately after
    return bash_filename

# transport
def transportMaps(claire_bin_path, results_path, bash_filename, transport_file_name):
    # run this after reg always
    bash_file = open(bash_filename, 'a')

    cmd = "ibrun " + claire_bin_path + "/clairetools -v1 " + results_path + "/velocity-field-x1.nii.gz -v2 " + results_path + "/velocity-field-x2.nii.gz -v3 " + results_path + "/velocity-field-x3.nii.gz -ifile "\
                   + results_path + "/" + transport_file_name + ".nii.gz -xfile " + results_path + "/" + transport_file_name + "_transported.nii.gz -deformimage" + " &> " + results_path + "/transport_log.txt";

    bash_file.write(cmd)
    bash_file.write("\n\n")
    bash_file.close()

    return bash_filename

def runTumorForwardModel(tu_code_path, atlas_image_path, results_path, inv_params, bash_filename, compute_sys='frontera'):
    bash_file = open(bash_filename, 'a')

    # call postprocs utils to create tumor input netcdf files :)
    bash_file.write("python " + tu_code_path + "/scripts/postproc/postproc-utils.py -m 2 -x " + results_path + "\n\n")

    atlas_name = "atlas"
    t_params = dict()
    t_params['code_path'] = tu_code_path
    t_params['rho_data'] = inv_params['rho_inv']
    t_params['k_data'] = inv_params['k_inv']
    t_params['nt_data'] = 50
    t_params['dt_data'] = 0.02
    t_params['forward_flag'] = 1
    t_params['create_synthetic'] = 1
    t_params['N'] = 256
    t_params['smooth_f'] = 1
    t_params['fac'] = 1
    t_params['model'] = 4

    gamma = inv_params['gamma']

    gm_path_nc = results_path + "/" + atlas_name + "_gm.nc"
    wm_path_nc = results_path + "/" + atlas_name + "_wm.nc"
    csf_path_nc = results_path + "/" + atlas_name + "_csf.nc"
    c0_path_nc = results_path + "/c0Recon_transported.nc"

    t_params['gm_path'] = gm_path_nc
    t_params['wm_path'] = wm_path_nc
    t_params['csf_path'] = csf_path_nc
    t_params['init_tumor_path'] = c0_path_nc
    t_params['compute_sys'] = compute_sys

    ### run four forward models
    for g in gamma:
        t_params['forcing_factor'] = g
        t_params['results_path'] = results_path + "/tumor-forward-gamma-" + str(int(g)) + "/"
        cmdline_tumor, err = TumorParams.getTumorRunCmd(t_params)
        cmdline_tumor += " &> " + t_params["results_path"] + "/tumor_solver_log.txt";
        bash_file.write(cmdline_tumor)
        bash_file.write("\n\n")

    return bash_filename

def convertTuToBratsSeg(tu_seg):
    brats_seg = 0 * tu_seg
    brats_seg[tu_seg == 4] = 7
    brats_seg[tu_seg == 3] = 5
    brats_seg[tu_seg == 2] = 6
    brats_seg[tu_seg == 1] = 4

    return brats_seg

def computeMismatch(c1, c2):
    c1 = c1.flatten()
    c2 = c2.flatten()

    num = la.norm(c1 - c2)
    den = la.norm(c1)

    return num/den


if __name__=='__main__':
    basedir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Process images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r_args = parser.add_argument_group('required arguments')
    r_args.add_argument ('-p',   '--patient_image_path', type = str, help = 'path to patient segmentation', required=True)
    r_args.add_argument ('-r',   '--tu_results_path', type = str, help = 'path to tumor inversion results directory containing log files', required=True)
    r_args.add_argument ('-a',   '--atlas_image_path', type = str, help = 'path to altas segmentation', required=True)
    r_args.add_argument ('-x',   '--results_path', type = str, help = 'path to results directory', required=True)
    r_args.add_argument ('-cl-path',   '--claire_bin_path', type = str, help = 'path to claire bin directory', required=True)
    r_args.add_argument ('-tu-path',   '--tu_code_path', type = str, help = 'path to tumor solver code directory', required=True)
    parser.add_argument ('-comp',   '--my_compute_sys', type = str, default = 'frontera', help = 'compute system')
    args = parser.parse_args();

    my_compute_sys = args.my_compute_sys

    if args.patient_image_path is None:
        parser.error("patient image path needs to be set")
    else:
        patient_image_path = args.patient_image_path
    if args.tu_results_path is None:
        parser.error("tumor inversion results path needs to be set")
    else:
        tu_results_path = args.tu_results_path
    if args.atlas_image_path is None:
        parser.error("altas image path needs to be set")
    else:
        atlas_image_path = args.atlas_image_path
    if args.claire_bin_path is None:
        parser.error("claire bin path needs to be set")
    else:
        claire_bin_path = args.claire_bin_path
    if args.results_path is None:
        parser.error("results path needs to be set")
    else:
        results_path = args.results_path
    if args.tu_code_path is None:
        parser.error("tumor bin path needs to be set")
    else:
        tu_code_path = args.tu_code_path

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    info_file = os.path.join(tu_results_path, 'info.dat')
    inv_params = dict()
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 0:
                values = lines[1].split(" ")
                inv_params['rho_inv'] = float(values[0])
                inv_params['k_inv'] = float(values[1])
            else:
                print( "  WARNING: output file info.dat is empty for tumor inversion of patient " + tu_results_path);
    else:
        print("  WARNING: no output file info.dat for tumor inversion of patient " + level_path );

    gamma = [3E4, 6E4, 9E4, 12E4]
    inv_params['gamma'] = gamma

    # create the bash file
    bash_filename = createBashFileHeader(results_path, my_compute_sys)

    # register patient to atlas
    bash_filename = performRegistration(atlas_image_path, patient_image_path, claire_bin_path, results_path, bash_filename, compute_sys=my_compute_sys)
    # convert c0Recon nc to nifti 
    c0_nc = tu_results_path + "/c0Recon.nc"
    file = Dataset(c0_nc, mode='r', format="NETCDF3_CLASSIC")
    c0 = np.transpose(file.variables['data'])
    nii = nib.load(patient_image_path)
    nib.save(nib.Nifti1Image(c0, nii.affine), results_path + "/c0Recon.nii.gz")
    # transport c0Recon to atlas
    bash_filename = transportMaps(claire_bin_path, results_path, bash_filename, "c0Recon")

    # run tumor solver with c0recon, rho, kappa and a few gamme values
    bash_filename = runTumorForwardModel(tu_code_path, atlas_image_path, results_path, inv_params, bash_filename, compute_sys=my_compute_sys)

    # registration the other way: register patient to each atlas
    # create many batch scripts as these registrations can run in parallel
    for g in gamma:
        results_path_reverse = results_path + "/reg-gamma-" + str(int(g)) + "/"
        bash_file = open(bash_filename, 'a')
        bash_file.write("python " + tu_code_path + "/scripts/postproc/postproc-utils.py -m 3 -x " + results_path + "\n\n")
        bash_file.close()
        new_seg_path = results_path_reverse + "/tu-seg.nii.gz"
        bash_filename = performRegistration(new_seg_path, patient_image_path, claire_bin_path, results_path_reverse, bash_filename, compute_sys=my_compute_sys, mask=False)
        bash_filename = transportMaps(claire_bin_path, results_path_reverse, bash_filename, "patient_csf")

    # find the mass-effect parameter
    bash_file = open(bash_filename, 'a')
    bash_file.write("python " + tu_code_path + "/scripts/postproc/postproc-utils.py -m 4 -x " + results_path + "\n\n")
    bash_file.close()

    # #submit the job
    # subprocess.call(['sbatch',bash_filename]);
