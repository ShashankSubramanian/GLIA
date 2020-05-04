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

def printStats(altas, patient, f):
    a_mat = 0 * altas
    a_mat[atlas == 5] = 1
    p_mat = 0 * patient
    p_mat[patient == 5] = 1

    diff_gm = computeNorm(a_mat, p_mat)

    a_mat = 0 * altas
    a_mat[atlas == 6] = 1
    p_mat = 0 * patient
    p_mat[patient == 6] = 1

    diff_wm = computeNorm(a_mat, p_mat)

    a_mat = 0 * altas
    a_mat[atlas == 7] = 1
    p_mat = 0 * patient
    p_mat[patient == 7] = 1

    diff_vt = computeNorm(a_mat, p_mat)

    a_mat = 0 * altas
    a_mat[atlas == 8] = 1
    p_mat = 0 * patient
    p_mat[patient == 8] = 1

    diff_csf = computeNorm(a_mat, p_mat)

    diff = diff_gm + diff_wm + diff_csf + diff_vt
    f.write("{} \t\t\t\t {} \t\t\t\t {} \t\t\t\t {} \t\t\t\t {}\n".format(diff_wm, diff_gm, diff_csf, diff_vt, diff))


scripts_path = os.getcwd() + "/.."
pat_name = "Brats18_CBICA_ABO_1"
patient_path = scripts_path + "/../brain_data/real_data/" + pat_name + "/data/" + pat_name + "_seg_tu_aff2jakob.nii.gz"
patient = nib.load(patient_path).get_fdata()
tumor_mask = np.logical_or(np.logical_or(patient == 1, patient == 4), patient == 2)
atlas_path = scripts_path + "/../brain_data/atlas/"
afile = scripts_path + "/../results/inv-" + pat_name + "/atlas_stat.txt"
f = open(afile, "w+")
f.write("WM \t\t\t\t GM \t\t\t\t CSF \t\t\t\t VT \t\t\t\t Total \n")
for i in range(1,9):
    atlas_file = "atlas-" + str(i) + ".nii.gz"
    atlas = nib.load(atlas_path + atlas_file).get_fdata()
    atlas[tumor_mask == 1] = 0      ### mask the tumor region
    printStats(atlas, patient, f)
f.close()

