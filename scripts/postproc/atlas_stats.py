import os, sys, warnings, argparse, subprocess
import math
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
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

def computeVolume(mat):
  sz      = mat.shape[0]
  h       = (2.0 * math.pi) /  sz
  measure = h * h * h
  vol = np.sum(mat.flatten())
  vol *= measure
  return vol

def computeMatVolumes(seg):
  ### vt
  vt = 0 * seg
  vt[seg == 7] = 1
  vt_vol = computeVolume(vt)
  ### wm
  wm = 0 * seg
  wm[seg == 6] = 1
  wm_vol = computeVolume(wm)
  ### gm
  gm = 0 * seg
  gm[seg == 5] = 1
  gm_vol = computeVolume(gm)
  ### csf
  csf = 0 * seg
  csf[seg == 8] = 1
  csf_vol = computeVolume(csf)

  return vt_vol, wm_vol, gm_vol, csf_vol

def printAtStats(atlas, pat_trans, at_tu, f):
  vt_vol, wm_vol, gm_vol, csf_vol = computeMatVolumes(atlas)
  vt_vol_pt, wm_vol_pt, gm_vol_pt, csf_vol_pt = computeMatVolumes(pat_trans)
  vt_vol_atu, wm_vol_atu, gm_vol_atu, csf_vol_atu = computeMatVolumes(at_tu)

  f.write("{},{},{},".format(vt_vol, vt_vol_pt, vt_vol_atu))
  f.write("{},{},{},".format(wm_vol, wm_vol_pt, wm_vol_atu))
  f.write("{},{},{},".format(gm_vol, gm_vol_pt, gm_vol_atu))
  f.write("{},{},{}".format(csf_vol, csf_vol_pt, csf_vol_atu))
  f.write("\n")

def printPatStats(seg, f):
  vt_vol, wm_vol, gm_vol, csf_vol = computeMatVolumes(seg)
  f.write("Volumes (vt, wm, gm, csf) = {}, {}, {}, {}\n".format(vt_vol, wm_vol, gm_vol, csf_vol))

scripts_path = os.path.dirname(os.path.realpath(__file__)) + "/.."
atlas_path   = scripts_path + "/../brain_data/atlas/"
pat_names = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AAP_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1"]
for pat_name in pat_names:
  print("Computing stats for {}".format(pat_name))
  pat_path     = scripts_path + "/../brain_data/real_data/" + pat_name + "/data/" + pat_name + "_seg_tu_aff2jakob.nii.gz"
  afile        = scripts_path + "/../results/stat-" + pat_name + "/atlas_stat-rand.csv"
  pfile        = scripts_path + "/../results/stat-" + pat_name + "/pat_stat.csv"

  f = open(pfile, "w+")
### print patient stats
  patient      = nib.load(pat_path).get_fdata()
  printPatStats(patient, f)
  f.close()
  tumor_mask = np.logical_or(np.logical_or(patient == 1, patient == 4), patient == 2)
### print atlas stats
  f = open(afile, "w+")
  inv_suff = "-rand"
  for i in range(1,9):
    atlas_name = "atlas-" + str(i)
    atlas = nib.load(atlas_path + atlas_name + ".nii.gz").get_fdata()
    atlas[tumor_mask == 1] = 0 ## mask the tumor region
    ## get recon A+T
    inv_results = scripts_path + "/../results/inv-" + pat_name + "/" + atlas_name + inv_suff
    reg_results = scripts_path + "/../results/reg-" + pat_name + "-nav/" + atlas_name
    at_tu       = nib.load(inv_results + "/seg_rec_final.nii.gz").get_fdata()
    pat_trans   = nib.load(reg_results + "/patient_labels_transported.nii.gz").get_fdata()
    printAtStats(atlas, pat_trans, at_tu, f)


