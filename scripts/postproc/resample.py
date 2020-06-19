import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset
from numpy import linalg as la
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['axes.formatter.use_mathtext'] = True
import matplotlib.pyplot as plt
import cv2
import skimage
import skimage.transform
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import pandas as pd
import seaborn as sns
import math

### helpers
def convertTuToBratsSeg(tu_seg):
    brats_seg = 0 * tu_seg
    brats_seg[tu_seg == 5] = 8
    brats_seg[tu_seg == 4] = 7
    brats_seg[tu_seg == 3] = 5
    brats_seg[tu_seg == 2] = 6
    brats_seg[tu_seg == 1] = 4

    return brats_seg

def convertHToBratsSeg(tu_seg):
    brats_seg = 0 * tu_seg
    brats_seg[tu_seg == 10] = 8
    brats_seg[tu_seg == 50] = 7
    brats_seg[tu_seg == 150] = 5
    brats_seg[tu_seg == 250] = 6

    return brats_seg

def createNetCDFFile(filename, dimensions, variable):
    file = Dataset(filename, mode='w', format="NETCDF3_CLASSIC");
    x = file.createDimension("x", dimensions[0]);
    y = file.createDimension("y", dimensions[1]);
    z = file.createDimension("z", dimensions[2]);
    data = file.createVariable("data", "f8", ("x","y","z",));
    data[:,:,:] = variable[:,:,:];
    file.close();

def resizeImage(wfile, name, size, outdir, inorder=0):
    ncimg      = Dataset(wfile, mode='r', format="NETCDF3_CLASSIC")
    img        = np.transpose(ncimg.variables['data'])
    dimensions = size * np.ones(3)
#    img        = skimage.transform.resize(img, dimensions, order=inorder)
    fac = float(size)/float(256)
    if size is not 256:
      img = ndimage.zoom(img, fac, order=inorder)
    wfile_new  = outdir + "/" + name + "_" + str(size) + ".nc"
    createNetCDFFile(wfile_new, dimensions, np.transpose(img))

tumor_dir = os.getcwd() + "/../../"

### resample all files needed for mass-effect inversion
### atlases
patient_name = "Brats18_CBICA_AAP_1"
pat          = False
new_size     = 256
#pat_prefix   = tumor_dir + "brain_data/real_data/" + patient_name + "/data/" + patient_name
#invdir       = tumor_dir + "results/" + patient_name
#regdir       = tumor_dir + "/reg/"
atlasdir     = "/scratch1/05027/shas1693/adni-nc/"
#atlasdir     = tumor_dir + "../data/adni-nc/"
outdir       = "/scratch1/05027/shas1693/adni-nc/" + str(new_size) + "/"
if not os.path.exists(outdir):
  os.makedirs(outdir)

for atlas in os.listdir(atlasdir):
  if atlas.find("t1_aff2jakob") is not -1:
    atlas_name = atlas.replace("_t1_aff2jakob.nc", "")
    print("resampling {}".format(atlas_name))
    t1 = atlasdir + atlas_name + "_t1_aff2jakob.nc"
    resizeImage(t1, atlas_name + "_t1_aff2jakob", new_size, outdir, inorder=3)
    seg = atlasdir + atlas_name + "_seg_aff2jakob_ants.nc"
    resizeImage(seg, atlas_name + "_seg_aff2jakob_ants", new_size, outdir, inorder=0)

#for atlas_name in os.listdir(regdir):
#    if atlas_name[0] == "5":
#      print("resampling atlas {}".format(atlas_name))
#      atlas_prefix =  regdir + atlas_name + "/" + atlas_name
#      wfile = atlas_prefix + "_wm.nc"
#      resizeImage(wfile, new_size)
#      wfile = atlas_prefix + "_gm.nc"
#      resizeImage(wfile, new_size)
#      wfile = atlas_prefix + "_csf.nc"
#      resizeImage(wfile, new_size)
#      wfile = atlas_prefix + "_vt.nc"
#      resizeImage(wfile, new_size)
#      wfile = regdir + atlas_name + "/c0Recon_transported.nc"
#      resizeImage(wfile, new_size)
#
#if pat:
#    patwfile = pat_prefix + "_tu_aff2jakob.nc"
#    resizeImage(patwfile, new_size)
#    patwfile = pat_prefix + "_obs_aff2jakob.nc"
#    resizeImage(patwfile, new_size)
#    patwfile = pat_prefix + "_wm_aff2jakob.nc"
#    resizeImage(patwfile, new_size)
#    patwfile = pat_prefix + "_gm_aff2jakob.nc"
#    resizeImage(patwfile, new_size)
#    patwfile = pat_prefix + "_csf_aff2jakob.nc"
#    resizeImage(patwfile, new_size)
#    patwfile = pat_prefix + "_vt_aff2jakob.nc"
#    resizeImage(patwfile, new_size)
#    patwfile = pat_prefix + "_c0Recon_aff2jakob.nc"
#    resizeImage(patwfile, new_size)
