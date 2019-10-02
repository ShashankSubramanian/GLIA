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
### Register atlas+tumor to patient and report deformation and final mismatches
### Report best gamma according to mismatch (of ventricles?)

def createNetCDFFile(filename, dimensions, variable):
    file = Dataset(filename, mode='w', format="NETCDF3_CLASSIC");
    x = file.createDimension("x", dimensions[0]);
    y = file.createDimension("y", dimensions[1]);
    z = file.createDimension("z", dimensions[2]);
    data = file.createVariable("data", "f8", ("x","y","z",));
    data[:,:,:] = variable[:,:,:];
    file.close();

def performRegistration(atlas_image_path, patient_image_path, claire_bin_path, results_path, compute_sys='frontera', mask=True):
    # create atlas vector labels
    atlas_name = "atlas"
    nii = nib.load(atlas_image_path)
    altas_seg = nii.get_fdata()
    altas_mat_img = 0 * altas_seg
    altas_mat_img[np.logical_or(altas_seg == 7, altas_seg == 8)] = 1
    nib.save(nib.Nifti1Image(altas_mat_img, nii.affine), results_path + "/" + atlas_name + "_csf.nii.gz")
    altas_mat_img = 0 * altas_seg
    altas_mat_img[altas_seg == 5] = 1
    nib.save(nib.Nifti1Image(altas_mat_img, nii.affine), results_path + "/" + atlas_name + "_gm.nii.gz")
    altas_mat_img = 0 * altas_seg
    altas_mat_img[altas_seg == 6] = 1
    nib.save(nib.Nifti1Image(altas_mat_img, nii.affine), results_path + "/" + atlas_name + "_wm.nii.gz")
    if not mask:
        # atlas has a tumor too; save it
        altas_mat_img = 0 * altas_seg
        altas_mat_img[altas_seg == 4] = 1
        nib.save(nib.Nifti1Image(altas_mat_img, nii.affine), results_path + "/" + atlas_name + "_tu.nii.gz")

    nii = nib.load(patient_image_path)
    patient_seg = nii.get_fdata()
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 7] = 1
    nib.save(nib.Nifti1Image(patient_mat_img, nii.affine), results_path + "/patient_csf.nii.gz")
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 5] = 1
    nib.save(nib.Nifti1Image(patient_mat_img, nii.affine), results_path + "/patient_gm.nii.gz")
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 6] = 1
    nib.save(nib.Nifti1Image(patient_mat_img, nii.affine), results_path + "/patient_wm.nii.gz")
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 4] = 1
    nib.save(nib.Nifti1Image(patient_mat_img, nii.affine), results_path + "/patient_tu.nii.gz")

    if mask:
        #create tumor masking file
        patient_mat_img = 0 * patient_seg + 1
        patient_mat_img[patient_seg == 4] = 0 #enhancing tumor mask
        patient_mat_img = gaussian_filter(patient_mat_img, sigma=2) # claire requires smoothing of masks
        nib.save(nib.Nifti1Image(patient_mat_img, nii.affine), results_path + "/patient_mask.nii.gz")

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
    bash_file.write("#SBATCH -o " + results_path + "/claire_solver_log.txt\n")
    bash_file.write("#SBATCH -p " + queue + "\n")
    bash_file.write("#SBATCH -N " + num_nodes + "\n")
    bash_file.write("#SBATCH -n " + num_cores + "\n")
    bash_file.write("#SBATCH -t 03:00:00\n\n")
    bash_file.write("source ~/.bashrc\n")

    if mask:
        cmd = "ibrun " + claire_bin_path + "/claire -mtc 3 " + results_path + "/" + atlas_name + "_csf.nii.gz " + results_path + "/" + \
                    atlas_name + "_gm.nii.gz " + results_path + "/" + atlas_name + "_wm.nii.gz " \
                    + "-mrc 3 " + results_path + "/patient_csf.nii.gz " + results_path + "/patient_gm.nii.gz " + results_path + \
                    "/patient_wm.nii.gz -mask " + results_path + "/patient_mask.nii.gz \
                    -nx 256 -train reduce -jbound 5e-2 -regnorm h1s-div -opttol 1e-2 -maxit 20 -krylovmaxit 50 -velocity -detdefgrad -deffield -residual -defmap -x " \
                    + results_path + "/"\
                    + " -monitordefgrad -verbosity 2 -disablerescaling -format nifti -sigma 2"
    else:
        cmd = "ibrun " + claire_bin_path + "/claire -mtc 4 " + results_path + "/" + atlas_name + "_csf.nii.gz " + results_path + "/" + atlas_name \
                    + "_gm.nii.gz " + results_path + "/" + atlas_name + "_wm.nii.gz " + results_path + "/" + atlas_name + "_tu.nii.gz "\
                    + "-mrc 4 " + results_path + "/patient_csf.nii.gz " + results_path + "/patient_gm.nii.gz " + results_path + "/patient_wm.nii.gz " \
                    + results_path + "/patient_tu.nii.gz \
                    -nx 256 -train reduce -jbound 5e-2 -regnorm h1s-div -opttol 1e-2 -maxit 20 -krylovmaxit 50 -velocity -detdefgrad -deffield -residual -defmap -x "\
                    + results_path + "/"\
                    + " -monitordefgrad -verbosity 2 -disablerescaling -format nifti -sigma 2"

    bash_file.write(cmd)
    bash_file.write("\n\n")
    bash_file.close()

    return bash_filename

# transport
def transportMaps(claire_bin_path, results_path, bash_filename, transport_file_name):
    bash_file = open(bash_filename, 'a')

    cmd = "ibrun " + claire_bin_path + "/clairetools -v1 " + results_path + "/velocity-field-x1.nii.gz -v2 " + results_path + "/velocity-field-x2.nii.gz -v3 " + results_path + "/velocity-field-x3.nii.gz -ifile "\
                   + results_path + "/" + transport_file_name + ".nii.gz -xfile " + results_path + "/" + transport_file_name + "_transported.nii.gz -deformimage" 

    bash_file.write(cmd)
    bash_file.write("\n\n")
    bash_file.close()

    return bash_filename

def runTumorForwardModel(tu_code_path, atlas_image_path, results_path, inv_params, bash_filename, compute_sys='frontera'):
    bash_file = open(bash_filename, 'a')

    # modify petsc
    if compute_sys == 'frontera':
        bash_file.write("module load petsc/3.11-single")
        bash_file.write("\n\n")

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

    nii = nib.load(results_path + "/" + atlas_name + "_gm.nii.gz")
    gm = nii.get_fdata()
    gm_path_nc = results_path + "/" + atlas_name + "_gm.nc"
    dimensions = 256 * np.ones(3)
    createNetCDFFile(gm_path_nc, dimensions, gm)
    nii = nib.load(results_path + "/" + atlas_name + "_wm.nii.gz")
    wm = nii.get_fdata()
    wm_path_nc = results_path + "/" + atlas_name + "_wm.nc"
    createNetCDFFile(wm_path_nc, dimensions, wm)
    nii = nib.load(results_path + "/" + atlas_name + "_csf.nii.gz")
    csf = nii.get_fdata()
    csf_path_nc = results_path + "/" + atlas_name + "_csf.nc"
    createNetCDFFile(csf_path_nc, dimensions, csf)
    nii = nib.load(results_path + "/c0Recon_transported.nii.gz")
    c0 = nii.get_fdata()
    c0_path_nc = results_path + "/c0Recon_transported.nc"
    createNetCDFFile(c0_path_nc, dimensions, c0)

    t_params['gm_path'] = gm_path_nc
    t_params['wm_path'] = wm_path_nc
    t_params['csf_path'] = csf_path_nc
    t_params['init_tumor_path'] = c0_path_nc
    t_params['compute_sys'] = compute_sys

    gamma = [1E4, 4E4, 8E4, 12E4]

    ### run four forward models
    for g in gamma:
        t_params['forcing_factor'] = g
        t_params['results_path'] = results_path + "/tumor-forward-gamma-" + str(g) + "/"
        cmdline_tumor, err = TumorParams.getTumorRunCmd(t_params)
        bash_file.write(cmdline_tumor)
        bash_file.write("\n\n")

    # modify petsc
    if compute_sys == 'frontera':
        bash_file.write("module unload petsc/3.11-single")
        bash_file.write("\n\n")

    return bash_filename, gamma

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
    parser.add_argument ('-m',   '--mode', type = int, default = 1, help = 'mode 1: register P-A; transport; run tumor models; mode 2: register AP-P; transport csf; mode 3: compute metrics')
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

    mode = args.mode
    gamma = []
    if mode == 1:
        # register patient to atlas
        bash_filename = performRegistration(atlas_image_path, patient_image_path, claire_bin_path, results_path, compute_sys=my_compute_sys)
        
        # convert c0Recon nc to nifti 
        c0_nc = tu_results_path + "/c0Recon.nc"
        file = Dataset(c0_nc, mode='r', format="NETCDF3_CLASSIC")
        c0 = np.transpose(file.variables['data'])

        nii = nib.load(results_path + "/patient_mask.nii.gz")
        nib.save(nib.Nifti1Image(c0, nii.affine), results_path + "/c0Recon.nii.gz")

        # transport c0Recon to atlas
        bash_filename = transportMaps(claire_bin_path, results_path, bash_filename, "c0Recon")
        # run tumor solver with c0recon, rho, kappa and a few gamme values
        bash_filename, gamma = runTumorForwardModel(tu_code_path, atlas_image_path, results_path, inv_params, bash_filename, compute_sys=my_compute_sys)
        # #submit the job
        # subprocess.call(['sbatch',bash_filename]);
    elif mode == 2:
        # registration the other way: register atlas_tumor to patient
        # create many batch scripts as these registrations can run in parallel
        if len(gamma) == 0:
            print("tumor forward models have not been run")
        else:
            for g in gamma:
                results_path_reverse = results_path + "/reg-gamma-" + str(g) + "/"

                if not os.path.exists(results_path_reverse):
                    os.makedirs(results_path_reverse)

                tu_path = results_path + "/tumor-forward-gamma-" + str(g) + "/"
                max_time = 0
                for f in os.listdir(tu_path):
                    f_split = f.split("_")
                    if f_split[0] == "seg":
                        f_split_2 = f_split[1].split("[")[1]
                        f_split_2 = f_split_2.split("]")[0]
                        time_step = int(f_split_2)
                        if time_step >= max_time:
                            max_time = time_step

                tu_img = tu_path + "seg_t[" + max_time + "].nc"
                # make it nifti for registration
                file = Dataset(tu_img, mode='r', format="NETCDF3_CLASSIC")
                tu_seg = np.transpose(file.variables['data'])
                brats_seg = convertTuToBratsSeg(tu_seg)
                nii = nib.load(results_path + "/patient_csf.nii.gz")
                new_seg_path = results_path_reverse + "/tu-seg.nii.gz"
                nib.save(nib.Nifti1Image(brats_seg, nii.affine), new_seg_path)
                bash_filename = performRegistration(patient_image_path, new_seg_path, claire_bin_path, results_path_reverse, compute_sys=my_compute_sys, mask=False)
                bash_filename = transportMaps(claire_bin_path, results_path_reverse, bash_filename, "patient_csf")
                # #submit the job
                # subprocess.call(['sbatch',bash_filename]);
    elif mode == 3:
        # compute metrics
        if len(gamma) == 0:
            print("tumor forward models have not been run")
        else:
            for g in gamma:
                pat_csf_path = results_path + "/patient_csf.nii.gz"
                nii = nib.load(pat_csf_path)
                pat_csf = nii.get_fdata()

                sim_csf_path = results_path_reverse + "/patient_csf_transported.nii.gz"
                nii = nib.load(sim_csf_path)
                sim_csf = nii.get_fdata()

                rel_err = computeMismatch(pat_csf, sim_csf)
                print("Relative error in csf for gamma = {} is {}".format(g, rel_err))
    else:
        print("run-mode not valid; use either 1 or 2")

