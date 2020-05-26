import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset
from numpy import linalg as la
import math
from postproc_utils import writeNII, createNetCDFFile



def convert_tu_brats_seg(tu_seg):
    brats_seg = 0 * tu_seg
    brats_seg[tu_seg == 5] = 8
    brats_seg[tu_seg == 4] = 7
    brats_seg[tu_seg == 3] = 5
    brats_seg[tu_seg == 2] = 6
    brats_seg[tu_seg == 1] = 4

    return brats_seg


def create_sbatch_header(results_path, compute_sys='frontera', suff=0):
    bash_filename = results_path + "/reg-and-transport_" + str(suff) + ".sh"
    print("creating job file in ", results_path)

    if compute_sys == 'frontera':
        queue = "normal"
        num_nodes = str(4)
        num_cores = str(128)
    elif compute_sys == 'maverick2':
        queue = "v100"
        num_nodes = str(1)
        num_cores = str(1)
    elif compute_sys == 'longhorn':
        queue = "v100"
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
    bash_file.write("#SBATCH -o " + results_path + "/coupling_solver_log_" + str(suff) + ".txt\n")
    bash_file.write("#SBATCH -p " + queue + "\n")
    bash_file.write("#SBATCH -N " + num_nodes + "\n")
    bash_file.write("#SBATCH -n " + num_cores + "\n")
    bash_file.write("#SBATCH -t 03:00:00\n\n")
    bash_file.write("source ~/.bashrc\n")

    bash_file.write("\n\n")
    bash_file.close()

    return bash_filename


def create_label_maps(atlas_image_path, patient_image_path, results_path, atlas_name):
    nii = nib.load(atlas_image_path)
    altas_seg = nii.get_fdata()
    writeNII(altas_seg, results_path + "/" + atlas_name + "_labels.nii.gz", ref_image=nii)
    altas_mat_img = 0 * altas_seg
    altas_mat_img[altas_seg == 7] = 1
    writeNII(altas_mat_img, results_path + "/" + atlas_name + "_vt.nii.gz", ref_image=nii)
    altas_mat_img = 0 * altas_seg
    altas_mat_img[altas_seg == 8] = 1
    writeNII(altas_mat_img, results_path + "/" + atlas_name + "_csf.nii.gz", ref_image=nii)
    altas_mat_img = 0 * altas_seg
    altas_mat_img[altas_seg == 5] = 1
    writeNII(altas_mat_img, results_path + "/" + atlas_name + "_gm.nii.gz", ref_image=nii)
    altas_mat_img = 0 * altas_seg
    altas_mat_img[np.logical_or(altas_seg == 6, altas_seg == 8)] = 1
    writeNII(altas_mat_img, results_path + "/" + atlas_name + "_wm.nii.gz", ref_image=nii)

    nii = nib.load(patient_image_path)
    patient_seg = nii.get_fdata()
    writeNII(patient_seg, results_path + "/patient_labels.nii.gz", ref_image=nii)
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 7] = 1
    writeNII(patient_mat_img, results_path + "/patient_vt.nii.gz", ref_image=nii)
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 8] = 1
    writeNII(patient_mat_img, results_path + "/patient_csf.nii.gz", ref_image=nii)
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 5] = 1
    writeNII(patient_mat_img, results_path + "/patient_gm.nii.gz", ref_image=nii)
    patient_mat_img = 0 * patient_seg
    patient_mat_img[(np.logical_or(patient_seg == 6, patient_seg == 8))] = 1
#    patient_mat_img[np.logical_or(np.logical_or(patient_seg == 6, patient_seg == 8), patient_seg == 4)] = 1
    writeNII(patient_mat_img, results_path + "/patient_wm.nii.gz", ref_image=nii)
    ### masking file
    patient_mat_img = 0 * patient_seg + 1
    patient_mat_img[np.logical_or(np.logical_or(patient_seg == 4, patient_seg == 1), patient_seg == 2)] = 0 #enhancing tumor mask
    patient_mat_img = gaussian_filter(patient_mat_img, sigma=2) # claire requires smoothing of masks
    writeNII(patient_mat_img, results_path + "/patient_mask.nii.gz", ref_image=nii)

def register(claire_bin_path, results_path, atlas_name, bash_filename, idx, multigpu = False):
    bash_file = open(bash_filename, 'a')
    if multigpu:
      cmd = "CUDA_VISIBLE_DEVICES=" + str((idx-1)%4) + " "
    else:
      cmd = ""

    cmd += "ibrun " + claire_bin_path + "/claire -mrc 3 " + results_path + "/" + atlas_name + "_vt.nii.gz " + results_path + "/" + atlas_name \
                + "_gm.nii.gz " + results_path + "/" + atlas_name + "_wm.nii.gz "\
                + "-mtc 3 " + results_path + "/patient_vt.nii.gz " + results_path + "/patient_gm.nii.gz " + results_path + "/patient_wm.nii.gz " \
                + "-mask " + results_path + "/patient_mask.nii.gz " \
                + "-nx 256 -train binary -jbound 0.2 -regnorm h1s-div -opttol 5e-2 -maxit 25 -krylovmaxit 50 -beta-div 1e-4 -velocity -detdefgrad -deffield -defmap -residual -x "\
                + results_path + "/"\
                + " -monitordefgrad -verbosity 1 -disablerescaling -format nifti -sigma 2" + " > " + results_path + "/registration_log.txt &";
    bash_file.write(cmd)
    bash_file.write("\n\n")
    bash_file.close()

    # return so that transport can be done immediately after
    return bash_filename

# transport
def transport(claire_bin_path, results_path, bash_filename, transport_file_name, idx, r2t=False, multigpu=False):
    # run this after reg always
    bash_file = open(bash_filename, 'a')
    if multigpu:
      cmd = "CUDA_VISIBLE_DEVICES=" + str((idx-1)%4) + " "
    else:
      cmd = ""

    if r2t == True:
      cmd += "ibrun " + claire_bin_path + "/clairetools -v1 " + results_path + "/velocity-field-x1.nii.gz -v2 " + results_path + "/velocity-field-x2.nii.gz -v3 " + results_path + "/velocity-field-x3.nii.gz -r2t -tlabelmap -labels 0,1,2,4,5,6,7,8 -ifile "\
              + results_path + "/" + transport_file_name + ".nii.gz -xfile " + results_path + "/" + transport_file_name + "_transported.nii.gz -iporder 1" + " > " + results_path + "/transport_log_" + transport_file_name + ".txt &";
    else:
### enable -r2t if reference to transport is needed. keep disabled for now.
      if transport_file_name == "patient_labels":
        cmd += "ibrun " + claire_bin_path + "/clairetools -v1 " + results_path + "/velocity-field-x1.nii.gz -v2 " + results_path + "/velocity-field-x2.nii.gz -v3 " + results_path + "/velocity-field-x3.nii.gz -tlabelmap -labels 0,1,2,4,5,6,7,8 -ifile "\
                + results_path + "/" + transport_file_name + ".nii.gz -xfile " + results_path + "/" + transport_file_name + "_transported.nii.gz -iporder 1" + " > " + results_path + "/transport_log_" + transport_file_name + ".txt &";
      else:
        cmd += "ibrun " + claire_bin_path + "/clairetools -v1 " + results_path + "/velocity-field-x1.nii.gz -v2 " + results_path + "/velocity-field-x2.nii.gz -v3 " + results_path + "/velocity-field-x3.nii.gz -ifile "\
                + results_path + "/" + transport_file_name + ".nii.gz -xfile " + results_path + "/" + transport_file_name + "_transported.nii.gz -deformimage -iporder 1" + " > " + results_path + "/transport_log_" + transport_file_name + ".txt &";

    bash_file.write(cmd)
    bash_file.write("\n\n")
    bash_file.close()

    return bash_filename


def convert_and_move_moi(path, out_path):
    file = Dataset(path, mode='r', format="NETCDF3_CLASSIC")
    c0 = np.transpose(file.variables['data'])
    nii = nib.load(out_path + "/patient_vt.nii.gz") 
    writeNII(c0, out_path + "/c0Recon.nii.gz", ref_image=nii)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r_args = parser.add_argument_group('required arguments')
    r_args.add_argument ('-tp',   '--transport', type = int, help = 'atlas list', default = 0) 
    r_args.add_argument ('-mg',   '--multigpu', type = int, help = 'use multiple gpus', default = 1) 
    args = parser.parse_args();
    tp = args.transport
    mg = args.multigpu

    scripts_path    = os.getcwd() + "/.."
    patient_name    = "Brats18_CBICA_ABO_1"
    patient_path    = scripts_path + "/../brain_data/real_data/" + patient_name + "/data/" + patient_name + "_seg_tu_aff2jakob.nii.gz"
    map_of_interest = scripts_path + "/../brain_data/real_data/" + patient_name + "/data/" + patient_name + "_c0Recon_aff2jakob.nc"
    submit_job      = True

    patient = nib.load(patient_path).get_fdata()
    atlas_path = scripts_path + "/../brain_data/atlas/"
    results_path = scripts_path + "/../results/reg-Brats18_CBICA_ABO_1-check/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    claire_path = scripts_path + "/../../claire-dev/bingpu/"
    bash_file = create_sbatch_header(results_path, compute_sys = "longhorn", suff=tp)
    if tp == 0:
      atlist = [1,2,3,4]
    else:
      atlist = [5,6,7,8]

    for i in atlist:
        atlas = "atlas-" + str(i)
        r_path = results_path + atlas
        if not os.path.exists(r_path):
            os.makedirs(r_path)
        ### create inputs for registration
        create_label_maps(atlas_path + atlas + ".nii.gz", patient_path, r_path, atlas_name = atlas)
        convert_and_move_moi(map_of_interest, r_path)
        ### create reg job scripts
        bash_file = register(claire_path, r_path, atlas, bash_file, i, multigpu=mg)
    
    fio = open(bash_file, 'a')
    fio.write("wait\n\n")
    fio.close()
    for i in atlist:
        atlas = "atlas-" + str(i)
        r_path = results_path + atlas
        ### create reg-transport script
        bash_file = transport(claire_path, r_path, bash_file, "c0Recon", i, r2t=False, multigpu=mg)
        bash_file = transport(claire_path, r_path, bash_file, "patient_labels", i, r2t=False, multigpu=mg)
        bash_file = transport(claire_path, r_path, bash_file, atlas + "_labels", i, r2t=True, multigpu=mg)

    fio = open(bash_file, 'a')
    fio.write("wait\n\n")
    fio.close()
    if submit_job:
      subprocess.call(['sbatch', bash_file])
