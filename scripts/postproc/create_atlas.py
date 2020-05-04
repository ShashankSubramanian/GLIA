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


def create_netcdf(filename, dimensions, variable):
    file = Dataset(filename, mode='w', format="NETCDF3_CLASSIC");
    x = file.createDimension("x", dimensions[0]);
    y = file.createDimension("y", dimensions[1]);
    z = file.createDimension("z", dimensions[2]);
    data = file.createVariable("data", "f8", ("x","y","z",));
    data[:,:,:] = variable[:,:,:];
    file.close();

def convert_tu_brats_seg(tu_seg):
    brats_seg = 0 * tu_seg
    brats_seg[tu_seg == 5] = 8
    brats_seg[tu_seg == 4] = 7
    brats_seg[tu_seg == 3] = 5
    brats_seg[tu_seg == 2] = 6
    brats_seg[tu_seg == 1] = 4

    return brats_seg



##pat_names = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AAP_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1"]
pat_names = ["Brats18_CBICA_AAP_1"]
scripts_path = os.getcwd() + "/.."
idx = 0
for n in pat_names:
    data_path = scripts_path + "/../results/reg-" + n + "/"
    pnm = pat_names[idx]
  ###  atlases = ["atlas-1", "atlas-2", "atlas-3"]
#atlases = ["atlas-1"]

    patient_seg = nib.load(scripts_path + "/../brain_data/real_data/" + pnm + "/data/" + pnm + "_seg_tu_aff2jakob.nii.gz").get_fdata()
    tc = np.logical_or(patient_seg == 1, patient_seg == 4)
    for i in range(1,9):
        atlas = "atlas-" + str(i)
        r_path = data_path + atlas
        atlas_path = scripts_path + "/../brain_data/atlas/" + atlas + ".nii.gz"
        altas_seg = nib.load(atlas_path).get_fdata()
        print("transporting maps for {}".format(atlas))
        c0_path = r_path + "/c0Recon_transported.nii.gz"
        c0 = nib.load(c0_path).get_fdata()
        createNetCDFFile(r_path + "/c0Recon_transported.nc", 256 * np.ones(3), np.transpose(c0))
       ### create nc files
        nm = r_path + "/" + atlas + "_vt_new"
        altas_mat_img = 0 * altas_seg
        altas_mat_img[altas_seg == 7] = 1
        createNetCDFFile(nm + ".nc", 256 * np.ones(3), np.transpose(altas_mat_img))
        nm = r_path + "/" + atlas + "_csf_new"
        altas_mat_img = 0 * altas_seg
        altas_mat_img[altas_seg == 8] = 1
        createNetCDFFile(nm + ".nc", 256 * np.ones(3), np.transpose(altas_mat_img))
        nm = r_path + "/" + atlas + "_gm_new"
        altas_mat_img = 0 * altas_seg
        altas_mat_img[altas_seg == 5] = 1
        createNetCDFFile(nm + ".nc", 256 * np.ones(3), np.transpose(altas_mat_img))
        nm = r_path + "/" + atlas + "_wm_new"
        altas_mat_img = 0 * altas_seg
#        altas_mat_img[np.logical_or(altas_seg == 6, tc == 1)] = 1  ### add tc to wm of atlas
        altas_mat_img[altas_seg == 6] = 1
        createNetCDFFile(nm + ".nc", 256 * np.ones(3), np.transpose(altas_mat_img))
    idx += 1      
