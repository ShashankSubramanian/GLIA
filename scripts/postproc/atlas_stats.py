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

def printAtlasStats(atlas, f, idx):
  vt_vol, wm_vol, gm_vol, csf_vol = computeMatVolumes(atlas)
  f.write("{},{},{},{},{}\n".format(idx, vt_vol, wm_vol, gm_vol, csf_vol))

def printAtStats(idx, atlas, pat_trans, at_tu, f):
  vt_vol, wm_vol, gm_vol, csf_vol = computeMatVolumes(atlas)
  vt_vol_pt, wm_vol_pt, gm_vol_pt, csf_vol_pt = computeMatVolumes(pat_trans)
  vt_vol_atu, wm_vol_atu, gm_vol_atu, csf_vol_atu = computeMatVolumes(at_tu)

  f.write("{},{},{},".format(idx,vt_vol, vt_vol_atu))
#  f.write("{},{},{},".format(wm_vol, wm_vol_pt, wm_vol_atu))
#  f.write("{},{},{},".format(gm_vol, gm_vol_pt, gm_vol_atu))
#  f.write("{},{},{}".format(csf_vol, csf_vol_pt, csf_vol_atu))
  f.write("\n")

def printPatStats(seg, f):
  vt_vol, wm_vol, gm_vol, csf_vol = computeMatVolumes(seg)
#  f.write("Volumes (vt, wm, gm, csf) = {}, {}, {}, {}\n".format(vt_vol, wm_vol, gm_vol, csf_vol))
  f.write(str(vt_vol))
  f.write("\n")
  f.write(str(wm_vol)) 
  f.write("\n")
  f.write(str(gm_vol)) 
  f.write("\n")
  f.write(str(csf_vol))
  f.write("\n")
  


scripts_path = os.path.dirname(os.path.realpath(__file__)) + "/.."
atlas_path   = scripts_path + "/../../data/adni/"
pat_names = ["Brats18_CBICA_ALU_1"]
#pat_names = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AAP_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1"]
flag_real = True

### vanilla stats
#adni_dir  = scripts_path + "/../../data/adni/"
##cbica_dir = scripts_path + "/../../data/cbica-fsl/"
#cbica_dir = scripts_path + "/../brain_data/"
##f         = open(adni_dir + "/adni-atlas-stats.csv", "w+")
#f_c       = open(cbica_dir + "/cbica-pat-stats.csv", "w+")
##for atlas in os.listdir(adni_dir):
##  if atlas.find("seg") is not -1:
##    idx = atlas.replace("_seg_aff2jakob_ants.nii.gz", "")
##    at  = nib.load(adni_dir + atlas).get_fdata()
##    printAtlasStats(at, f, idx)
##f.close() 
#
#for pat in pat_names:
#  seg = nib.load(cbica_dir + "real_data/" + pat + "/data/" + pat + "_seg_tu_aff2jakob.nii.gz").get_fdata()
#  printAtlasStats(seg, f_c, pat)
#f_c.close()
#
##for pat in os.listdir(cbica_dir):
##  if pat.find("seg_tu_aff2jakob_fsl") is not -1:
##    idx = pat.replace("_seg_tu_aff2jakob_fsl.nii.gz", "")
##    p = nib.load(cbica_dir + pat).get_fdata()
##    printAtlasStats(p, f_c, idx)
##f_c.close()
#

### stats for patients and reconstructed patients

suff = ""
if flag_real:
  for pat_name in pat_names:
    print("Computing stats for {}".format(pat_name))
    pat_path     = scripts_path + "/../brain_data/real_data/" + pat_name + "/data/" + pat_name + "_seg_tu_aff2jakob.nii.gz"
    afile        = scripts_path + "/../results/" + pat_name + "/stat/atlas_stat" + suff + ".csv"
    patient      = nib.load(pat_path).get_fdata()
###    tumor_mask = np.logical_or(np.logical_or(patient == 1, patient == 4), patient == 2)
### print atlas stats
    f = open(afile, "w+")
    inv_suff = suff
    invdir       = scripts_path + "/../results/" + pat_name + "/tu/"
    for atlas_name in os.listdir(invdir):
      if atlas_name[0] == "5": ### adni atlases
        atlas = nib.load(atlas_path + atlas_name + "_seg_aff2jakob_ants.nii.gz").get_fdata()
        ###atlas[tumor_mask == 1] = 0 ## mask the tumor region
        ## get recon A+T
        inv_results = invdir + "/" + atlas_name + inv_suff
        reg_results = scripts_path + "/../results/" + pat_name + "/reg/" + atlas_name
        at_tu       = nib.load(inv_results + "/seg_rec_final.nii.gz").get_fdata()
        pat_trans   = nib.load(reg_results + "/" + pat_name + "_labels_transported.nii.gz").get_fdata()
        printAtStats(atlas_name, atlas, pat_trans, at_tu, f)
else:
  inv_dir = scripts_path + "/../results/"
  inv_folder = "atlas-2-tu-case3"
  patient = nib.load(inv_dir + inv_folder + "/seg_final.nii.gz").get_fdata()
  f = open(inv_dir + inv_folder + "/pat-stats.csv", "w+")
  printPatStats(patient, f)
  f.close()
  
