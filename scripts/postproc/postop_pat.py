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

scripts_path = os.path.dirname(os.path.realpath(__file__)) + "/.."

flag_real = True
if flag_real:
  #pat_names = ["Brats18_CBICA_ALU_1"]
  pat_names = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AAP_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1"]
  inv_suff = "-noreg"
  for pat_name in pat_names:
    print("postop for patient {}".format(pat_name))
    pat_path = scripts_path + "/../brain_data/real_data/" + pat_name + "/data/" + pat_name + "_seg_tu_aff2jakob.nii.gz"
    nii = nib.load(pat_path)
    invdir = scripts_path + "/../results/" + pat_name + "/tu/"
    for atlas_name in os.listdir(invdir):
      if atlas_name[0] == "5" and atlas_name.find(inv_suff) is not -1:
        inv_results = invdir + atlas_name
        infile = inv_results + "/seg_rec_final.nc"
        ncfile = Dataset(infile, mode='r', format="NETCDF3_CLASSIC")
        dat = np.transpose(ncfile.variables['data'])
        outfile = infile.replace(".nc", ".nii.gz")
##        dat = convert_tu_brats_seg(dat)
        writeNII(dat, outfile, ref_image=nii)
        infile = inv_results + "/c_rec_final.nc"
        ncfile = Dataset(infile, mode='r', format="NETCDF3_CLASSIC")
        dat = np.transpose(ncfile.variables['data'])
        outfile = infile.replace(".nc", ".nii.gz")
        writeNII(dat, outfile, ref_image=nii)
        infile = inv_results + "/displacement_rec_final.nc"
        ncfile = Dataset(infile, mode='r', format="NETCDF3_CLASSIC")
        dat = np.transpose(ncfile.variables['data'])
        outfile = infile.replace(".nc", ".nii.gz")
        writeNII(dat, outfile, ref_image=nii)
else:
    nii = nib.load(scripts_path + "/../brain_data/atlas/atlas-2.nii.gz")
    pats = ["atlas-2-case2", "atlas-2-case3"]
    atsuff = "hp"
    cus_str = "rec_"
    for pat in pats:
      for idx in range(1,9):
        inv_results = scripts_path + "/../results/inv-" + pat + "/atlas-" + str(idx) + "-" + atsuff + "/"
        infile = inv_results + "/seg_" + cus_str + "final.nc"
        ncfile = Dataset(infile, mode='r', format="NETCDF3_CLASSIC")
        dat = np.transpose(ncfile.variables['data'])
        outfile = infile.replace(".nc", ".nii.gz")
        writeNII(dat, outfile, ref_image=nii)
#    infile = inv_results + "/mri_" + cus_str +"final.nc"
#    ncfile = Dataset(infile, mode='r', format="NETCDF3_CLASSIC")
#    dat = np.transpose(ncfile.variables['data'])
#    outfile = infile.replace(".nc", ".nii.gz")
#    writeNII(dat, outfile, ref_image=nii)
        infile = inv_results + "/c_" + cus_str +"final.nc"
        ncfile = Dataset(infile, mode='r', format="NETCDF3_CLASSIC")
        dat = np.transpose(ncfile.variables['data'])
        outfile = infile.replace(".nc", ".nii.gz")
        writeNII(dat, outfile, ref_image=nii)
        infile = inv_results + "/displacement_" + cus_str + "final.nc"
        ncfile = Dataset(infile, mode='r', format="NETCDF3_CLASSIC")
        dat = np.transpose(ncfile.variables['data'])
        outfile = infile.replace(".nc", ".nii.gz")
        writeNII(dat, outfile, ref_image=nii)
