import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import math

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


#--------------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Mass effect inversion',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-p', '--patient_dir', type = str, help = 'path to patients', required = True) 
  args   = parser.parse_args();
  f_c    = open(args.patient_dir + "/pat_stats.csv", "w+")
  fail   = []
  for pat in os.listdir(args.patient_dir):
    if os.path.exists(args.patient_dir + "/" + pat + "/" + pat + "_t1.nii.gz"):
      idx = pat
      print("stating pat ", idx)
      nm = args.patient_dir + "/" + pat + "/aff2jakob/" + pat + "_seg_ants_aff2jakob.nii.gz"
      if not os.path.exists(nm):
        print("pat {} does not exist".format(pat))
        fail.append(pat)
        continue
      p = nib.load(nm).get_fdata()
      printAtlasStats(p, f_c, idx)
  f_c.close()
  print("failed patients: ", fail)

