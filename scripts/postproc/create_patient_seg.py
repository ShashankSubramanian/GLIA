import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
import TumorParams
from netCDF4 import Dataset
from numpy import linalg as la

from postproc_utils import writeNII, createNetCDFFile


def convertTuToBratsSeg(tu_seg):
    """ converts tumor solver segmentation from single-species into brats labels """
    brats_seg = 0 * tu_seg
    brats_seg[tu_seg == 5] = 8          # glm from tu solver is just csf
    brats_seg[tu_seg == 4] = 7          # vt
    brats_seg[tu_seg == 3] = 5          # gray matter
    brats_seg[tu_seg == 2] = 6          # white matter
    brats_seg[tu_seg == 1] = 4          # tumor is en

    return brats_seg

def computeMismatch(c1, c2):
    c1 = c1.flatten()
    c2 = c2.flatten()

    num = la.norm(c1 - c2)
    den = la.norm(c1)

    return num/den


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Create patient segmentation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r_args = parser.add_argument_group('required arguments')
    r_args.add_argument ('-p',   '--patient_image_path', type = str, help = 'path to patient segmentation', required=True)
    r_args.add_argument ('-a',   '--atlas_image_path', type = str, help = 'path to affinely registered atlas segmentation', required=True)
    args = parser.parse_args();
    patient_path = args.patient_image_path
    atlas_path = args.atlas_image_path

    seg_file = patient_path + "/seg_t[50].nc"
    seg = Dataset(seg_file, mode='r', format="NETCDF3_CLASSIC")
    tu_seg = np.transpose(seg.variables['data'])
    brats_seg = convertTuToBratsSeg(tu_seg)

    atlas = atlas_path + "/jakob_segmented_with_cere_lps_256x256x256.nii.gz"
    nii = nib.load(atlas)
    new_seg_path = patient_path + "/patient_seg.nii.gz"
    writeNII(brats_seg, new_seg_path, ref_image=nii)
