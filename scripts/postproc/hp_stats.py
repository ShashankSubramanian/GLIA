import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
import TumorParams
from netCDF4 import Dataset
from numpy import linalg as la
import math
from postproc_utils import writeNII, createNetCDFFile
from shutil import copyfile

def writeNII(img, filename, affine=None, ref_image=None):
    '''
    function to write a nifti image, creates a new nifti object
    '''
    if ref_image is not None:
        data = nib.Nifti1Image(img, affine=ref_image.affine, header=ref_image.header);
        data.header['datatype'] = 64
        data.header['glmax'] = np.max(img)
        data.header['glmin'] = np.min(img)
    elif affine is not None:
        data = nib.Nifti1Image(img, affine=affine);
    else:
        data = nib.Nifti1Image(img, np.eye(4))

    nib.save(data, filename);
def convert_tu_brats_seg(tu_seg):
    brats_seg = 0 * tu_seg
    brats_seg[tu_seg == 5] = 8
    brats_seg[tu_seg == 4] = 7
    brats_seg[tu_seg == 3] = 5
    brats_seg[tu_seg == 2] = 6
    brats_seg[tu_seg == 1] = 4

    return brats_seg

def create_sbatch_header(results_path, compute_sys='longhorn'):
    bash_filename = results_path + "/reg-and-transport.sh"
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
    bash_file.write("#SBATCH -o " + results_path + "/coupling_solver_log.txt\n")
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
    altas_mat_img[altas_seg == 4] = 1
    writeNII(altas_mat_img, results_path + "/" + atlas_name + "_tu.nii.gz", ref_image=nii)
    altas_mat_img = 0 * altas_seg
    altas_mat_img[np.logical_or(altas_seg == 6, altas_seg == 8)] = 1
    writeNII(altas_mat_img, results_path + "/" + atlas_name + "_wm.nii.gz", ref_image=nii)

    nii = nib.load(patient_image_path)
    patient_seg = nii.get_fdata()
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 7] = 1
    writeNII(patient_mat_img, results_path + "/patient_vt.nii.gz", ref_image=nii)
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 8] = 1
    writeNII(patient_mat_img, results_path + "/patient_csf.nii.gz", ref_image=nii)
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 4] = 1
    writeNII(patient_mat_img, results_path + "/patient_tu.nii.gz", ref_image=nii)
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 5] = 1
    writeNII(patient_mat_img, results_path + "/patient_gm.nii.gz", ref_image=nii)
    patient_mat_img = 0 * patient_seg
    patient_mat_img[(np.logical_or(patient_seg == 6, patient_seg == 8))] = 1
    writeNII(patient_mat_img, results_path + "/patient_wm.nii.gz", ref_image=nii)


#    ### masking file
#    patient_mat_img = 0 * patient_seg + 1
#    patient_mat_img[np.logical_or(np.logical_or(patient_seg == 4, patient_seg == 1), patient_seg == 2)] = 0 #enhancing tumor mask
#    patient_mat_img = gaussian_filter(patient_mat_img, sigma=2) # claire requires smoothing of masks
#    writeNII(patient_mat_img, results_path + "/patient_mask.nii.gz", ref_image=nii)

def register(claire_bin_path, results_path, atlas_name, bash_filename):
    bash_file = open(bash_filename, 'a')
    cmd = "ibrun " + claire_bin_path + "/claire -mtc 1 " + results_path + "/" + atlas_name + "_vt.nii.gz " \
                + "-mrc 1 " + results_path + "/patient_vt.nii.gz " \
                + "-nx 256 -train reduce -jbound 5e-2 -regnorm h1s-div -opttol 1e-2 -maxit 50 -krylovmaxit 50 -velocity -detdefgrad -deffield -defmap -residual -x "\
                + results_path + "/"\
                + " -monitordefgrad -verbosity 1 -disablerescaling -format nifti -sigma 2" + " &> " + results_path + "/registration_log.txt";
#    cmd = "ibrun " + claire_bin_path + "/claire -mtc 4 " + results_path + "/" + atlas_name + "_vt.nii.gz " + results_path + "/" + atlas_name \
#                + "_gm.nii.gz " + results_path + "/" + atlas_name + "_wm.nii.gz " + results_path + "/" + atlas_name + "_tu.nii.gz "\
#                + "-mrc 4 " + results_path + "/patient_vt.nii.gz " + results_path + "/patient_gm.nii.gz " + results_path + "/patient_wm.nii.gz " \
#                + results_path + "/patient_tu.nii.gz " \
#                + "-nx 256 -train reduce -jbound 5e-2 -regnorm h1s-div -opttol -objwts 1,0,0,0 -maxit 50 -krylovmaxit 50 -velocity -detdefgrad -deffield -defmap -residual -x "\
#                + results_path + "/"\
#                + " -monitordefgrad -verbosity 1 -disablerescaling -format nifti -sigma 2" + " &> " + results_path + "/registration_log.txt";
#                + "-mask " + results_path + "/patient_mask.nii.gz " \
#    cmd = "ibrun " + claire_bin_path + "/claire -mrc 4 " + results_path + "/" + atlas_name + "_vt.nii.gz " + results_path + "/" + atlas_name + "_csf.nii.gz " + results_path + "/" + atlas_name \
#                + "_gm.nii.gz " + results_path + "/" + atlas_name + "_wm.nii.gz "\
#                + "-mtc 4 " + results_path + "/patient_vt.nii.gz " + results_path + "/patient_csf.nii.gz " + results_path + "/patient_gm.nii.gz " + results_path + "/patient_wm.nii.gz " \
#                + "-mask " + results_path + "/patient_mask.nii.gz " \
#                + "-nx 256 -train reduce -jbound 5e-2 -regnorm h1s-div -opttol 1e-2 -maxit 30 -krylovmaxit 50 -velocity -detdefgrad -deffield -residual -x "\
#                + results_path + "/"\
#                + " -monitordefgrad -verbosity 1 -disablerescaling -format nifti -sigma 2" + " &> " + results_path + "/registration_log.txt";
    bash_file.write(cmd)
    bash_file.write("\n\n")
    bash_file.close()

    # return so that transport can be done immediately after
    return bash_filename

# transport
# transport
def transport(claire_bin_path, results_path, bash_filename, transport_file_name):
    # run this after reg always
    bash_file = open(bash_filename, 'a')

    cmd = "ibrun " + claire_bin_path + "/clairetools -v1 " + results_path + "/velocity-field-x1.nii.gz -v2 " + results_path + "/velocity-field-x2.nii.gz -v3 " + results_path + "/velocity-field-x3.nii.gz -ifile "\
                   + results_path + "/" + transport_file_name + ".nii.gz -xfile " + results_path + "/" + transport_file_name + "_transported.nii.gz -deformimage -iporder 1" + " &> " + results_path + "/transport_log.txt";

    bash_file.write(cmd)
    bash_file.write("\n\n")
    bash_file.close()

    return bash_filename


base_dir = os.getcwd() + "/../../"


pat_names = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AAP_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1"]
nm = ["ABO", "AAP", "AMH", "ALU"]
atlases = ["atlas-4", "atlas-5", "atlas-9", "atlas-10"]
c_avg = np.zeros((256,256,256))
u_avg = np.zeros((256,256,256))
num_cases = 0
claire_path = base_dir + "../claire-dev/bingpu/"


pat_names_temp = ["Brats18_CBICA_AAP_1"]
reg_flag = 0
if reg_flag:
    for pat in pat_names_temp:
        r_path = base_dir + "results/stat-" + pat + "/registration-mild/"
        if not os.path.exists(r_path):
            os.makedirs(r_path)
        bash_file = create_sbatch_header(r_path)
        nii = nib.load(base_dir + "brain_data/real_data/" + pat + "/data/" + pat + "_seg_tu_aff2jakob.nii.gz")
        pat_seg = nii.get_fdata()

        for atlas in atlases:
            inv_results = base_dir + "results/inv-" + pat + "-" + atlas + "-mri-KS/"
            if not os.path.exists(inv_results + "seg_brats.nii.gz"):
                file = Dataset(inv_results + "seg_rec_final.nc", mode='r', format="NETCDF3_CLASSIC")
                seg = np.transpose(file.variables['data'])
                seg = convert_tu_brats_seg(seg)
                writeNII(seg, inv_results + "seg_brats.nii.gz", ref_image = nii)
            if not os.path.exists(inv_results + "pat_seg_wt_brats.nii.gz"):
                pat_seg[pat_seg == 1] = 4 ## nec to en
                pat_seg[pat_seg == 2] = 4 ## ed to en
                writeNII(pat_seg, inv_results + "pat_seg_wt_brats.nii.gz", ref_image = nii)
            r_path_ = r_path + atlas
            if not os.path.exists(r_path_):
                os.makedirs(r_path_)
            if not os.path.exists(r_path_ + "/c_rec.nii.gz"):
                file = Dataset(inv_results + "c_rec_final.nc", mode='r', format="NETCDF3_CLASSIC")
                f = np.transpose(file.variables['data'])
                writeNII(f, r_path_ + "/c_rec.nii.gz", ref_image = nii)
            if not os.path.exists(r_path_ + "/u_rec.nii.gz"):
                file = Dataset(inv_results + "displacement_rec_final.nc", mode='r', format="NETCDF3_CLASSIC")
                f = np.transpose(file.variables['data'])
                writeNII(f, r_path_ + "/u_rec.nii.gz", ref_image = nii)
            create_label_maps(inv_results + "seg_brats.nii.gz", inv_results + "pat_seg_wt_brats.nii.gz", r_path_, atlas_name = atlas)
            ### create reg job scripts
            bash_file = register(claire_path, r_path_, atlas, bash_file)
            ### create reg-transport script
            bash_file = transport(claire_path, r_path_, bash_file, "c_rec")
            bash_file = transport(claire_path, r_path_, bash_file, "u_rec")
            bash_file = transport(claire_path, r_path_, bash_file, "patient_vt")


    
c_avg_reg = np.zeros((256,256,256))
u_avg_reg = np.zeros((256,256,256))
compute_avg = 1
num_cases = len(atlases)
if compute_avg:
    for pat in pat_names:
        c_avg = 0 * c_avg
        u_avg = 0 * u_avg
        c_avg_reg = 0 * c_avg_reg
        u_avg_reg = 0 * u_avg_reg
        
        for atlas in atlases:
            inv_results = base_dir + "results/inv-" + pat + "-" + atlas + "-mri-KS/"
            file = Dataset(inv_results + "c_rec_final.nc", mode='r', format="NETCDF3_CLASSIC")
            c_avg += np.transpose(file.variables['data'])
            file = Dataset(inv_results + "displacement_rec_final.nc", mode='r', format="NETCDF3_CLASSIC")
            u_avg += np.transpose(file.variables['data'])
        c_avg /= num_cases
        u_avg /= num_cases
        r_path = base_dir + "results/stat-" + pat + "/"
        if not os.path.exists(r_path):
            os.makedirs(r_path)
        nii = nib.load(base_dir + "brain_data/real_data/" + pat + "/data/" + pat + "_seg_tu_aff2jakob.nii.gz")
        writeNII(c_avg, r_path + "c_avg.nii.gz", ref_image = nii)
        writeNII(u_avg, r_path + "u_avg.nii.gz", ref_image = nii)

#        r_path_ = r_path + "registration/" + atlas
#        for atlas in atlases:
#            c_avg_reg += nib.load(r_path_ + "/c_rec_transported.nii.gz").get_fdata()
#            u_avg_reg += nib.load(r_path_ + "/u_rec_transported.nii.gz").get_fdata()
#        c_avg_reg /= num_cases
#        u_avg_reg /= num_cases
#        writeNII(c_avg_reg, r_path + "c_avg_reg.nii.gz", ref_image = nii)
#        writeNII(u_avg_reg, r_path + "u_avg_reg.nii.gz", ref_image = nii)
#
        
#    copyfile(base_dir + "brain_data/real_data/" + pat + "/data/" + pat + "_t1ce_aff2jakob.nii.gz", r_path + pat + "_t1ce_aff2jakob.nii.gz")
#    copyfile(base_dir + "brain_data/real_data/" + pat + "/data/" + pat + "_t1_aff2jakob.nii.gz", r_path + pat + "_t1_aff2jakob.nii.gz")
#    copyfile(base_dir + "brain_data/real_data/" + pat + "/data/" + pat + "_t2_aff2jakob.nii.gz", r_path + pat + "_t2_aff2jakob.nii.gz")
#    copyfile(base_dir + "brain_data/real_data/" + pat + "/data/" + pat + "_flair_aff2jakob.nii.gz", r_path + pat + "_flair_aff2jakob.nii.gz")
#    copyfile(base_dir + "brain_data/real_data/" + pat + "/data/" + pat + "_seg_tu_aff2jakob.nii.gz", r_path + pat + "_seg_tu_aff2jakob.nii.gz")
