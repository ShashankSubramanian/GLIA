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


def computeNorm(img_1, img_2):
    return la.norm(img_1 - img_2)

def printStats(altas, patient):
    a_mat = 0 * altas
    a_mat[atlas == 5] = 1
    p_mat = 0 * patient
    p_mat[patient == 5] = 1

    diff = computeNorm(a_mat, p_mat)
    print("GM difference is {}".format(diff))

    a_mat = 0 * altas
    a_mat[atlas == 6] = 1
    p_mat = 0 * patient
    p_mat[patient == 6] = 1

    diff = computeNorm(a_mat, p_mat)
    print("WM difference is {}".format(diff))

    a_mat = 0 * altas
    a_mat[atlas == 7] = 1
    p_mat = 0 * patient
    p_mat[patient == 7] = 1

    diff = computeNorm(a_mat, p_mat)
    print("VT difference is {}".format(diff))

    a_mat = 0 * altas
    a_mat[atlas == 8] = 1
    p_mat = 0 * patient
    p_mat[patient == 8] = 1

    diff = computeNorm(a_mat, p_mat)
    print("CSF difference is {}".format(diff))


scripts_path = os.getcwd() + "/.."
patient_path = scripts_path + "/../brain_data/t16/t16-m12-seg.nii.gz"
patient = nib.load(patient_path).get_fdata()
atlas_path = scripts_path + "/../brain_data/atlas/" 
for atlas_file in os.listdir(atlas_path):
    atlas = nib.load(atlas_path + atlas_file).get_fdata()
    print("Printing statistics for atlas {}...".format(atlas_file.split(".")[0]))
    printStats(atlas, patient)

