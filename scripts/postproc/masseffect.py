import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
import TumorParams
from netCDF4 import Dataset

### Invert in patient-space to get (rho, kappa, c0)
### Register patient to some atlas and transport c0 to this atlas
### Grow some tumors with (rho, kappa, c0-transported) and some gamma
### Register atlas+tumor to patient and report deformation and final mismatches
### Report best gamma according to mismatch (of ventricles?)

def performRegistration(atlas_image_path, patient_image_path, claire_bin_path, results_path):
    # create atlas vector labels
    atlas_name = atlas_image_path.split("/")[-1]
    atlas_name = atlas_name.split(".")[0]
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
    patient_mat_img = 0 * patient_seg + 1
    patient_mat_img[patient_seg == 4] = 0 #enhancing tumor mask
    patient_mat_img = gaussian_filter(patient_mat_img, sigma=2) # claire requires smoothing of masks
    nib.save(nib.Nifti1Image(patient_mat_img, nii.affine), results_path + "/patient_mask.nii.gz")

    bash_filename = results_path + "/coupling_job_submission.sh"
    print("creating job file in ", results_path)
    bash_file = open(bash_filename, 'w')
    bash_file.write("#!/bin/bash\n\n");
    bash_file.write("#SBATCH -J mass-effect-cpl\n");
    bash_file.write("#SBATCH -o " + results_path + "/claire_solver_log.txt\n")
    bash_file.write("#SBATCH -p normal\n")
    bash_file.write("#SBATCH -N 3\n")
    bash_file.write("#SBATCH -n 64\n")                                                                                                                                                                                                                                                                          
    bash_file.write("#SBATCH -t 05:00:00\n\n")
    bash_file.write("source ~/.bashrc\n")

    cmd = "ibrun " + claire_bin_path + "/claire -mtc 3 " + results_path + "/" + atlas_name + "_csf.nii.gz " + results_path + "/" + atlas_name + "_gm.nii.gz " + results_path + "/" + atlas_name + "_wm.nii.gz " \
                    + "-mrc 3 " + results_path + "/patient_csf.nii.gz " + results_path + "/patient_gm.nii.gz " + results_path + "/patient_wm.nii.gz -mask " + results_path + "/patient_mask.nii.gz \
                    -nx 256 -train reduce -jbound 5e-2 -regnorm h1s-div -opttol 1e-2 -maxit 20 -krylovmaxit 50 -velocity -detdefgrad -deffield -residual -defmap -x " + results_path \
                    + " -monitordefgrad -verbosity 2 -disablerescaling -format nifti -sigma 2"

    bash_file.write(cmd)
    bash_file.write("\n\n")
    bash_file.close()

    # #submit the job
    # subprocess.call(['sbatch',bash_filename]);

    return bash_filename

# transport
def transportMaps(claire_bin_path, results_path, tu_results_path, bash_filename):
    bash_file = open(bash_filename, 'a')

    # convert c0Recon nc to nifti
    c0_nc = tu_results_path + "/c0Recon.nc"
    file = Dataset(c0_nc, mode='r', format="NETCDF3_CLASSIC")
    c0 = np.transpose(file.variables['data'])

    nii = nib.load(results_path + "/patient_mask.nii.gz")
    nib.save(nib.Nifti1Image(c0, nii.affine), results_path + "/c0Recon.nii.gz")

    cmd = "ibrun " + claire_bin_path + "/clairetools -v1 " + results_path + "/velocity-field-x1.nii.gz -v2 " + results_path + "/velocity-field-x2.nii.gz -v3 " + results_path + "/velocity-field-x3.nii.gz -ifile "\
                   + results_path + "/c0Recon.nii.gz -xfile " + results_path + "/c0Recon_transported.nii.gz -deformimage" 

    bash_file.write(cmd)
    bash_file.write("\n\n")
    bash_file.close()

    return bash_filename

def createNetCDFFile(filename, dimensions, variable):
    file = Dataset(filename, mode='w', format="NETCDF3_CLASSIC");
    x = file.createDimension("x", dimensions[0]);
    y = file.createDimension("y", dimensions[1]);
    z = file.createDimension("z", dimensions[2]);
    data = file.createVariable("data", "f8", ("x","y","z",));
    data[:,:,:] = variable[:,:,:];
    file.close();

def runTumorForwardModel(tu_code_path, atlas_image_path, results_path, inv_params, bash_filename):
    bash_file = open(bash_filename, 'a')
    atlas_name = atlas_image_path.split("/")[-1]
    atlas_name = atlas_name.split(".")[0]
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
    t_params['compute_sys'] = 'frontera'

    gamma = [1E4, 4E4, 8E4, 12E4]

    ### run four forward models
    for g in gamma:
        t_params['forcing_factor'] = g
        t_params['results_path'] = results_path + "/tumor-forward-gamma-" + str(g)
        cmdline_tumor, err = TumorParams.getTumorRunCmd(t_params)
        bash_file.write(cmdline_tumor)
        bash_file.write("\n\n")

    return bash_filename

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
    args = parser.parse_args();

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


    bash_filename = performRegistration(atlas_image_path, patient_image_path, claire_bin_path, results_path)
    bash_filename = transportMaps(claire_bin_path, results_path, tu_results_path, bash_filename)
    bash_filename = runTumorForwardModel(tu_code_path, atlas_image_path, results_path, inv_params, bash_filename)



