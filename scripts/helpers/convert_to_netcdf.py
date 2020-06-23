import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset
import cv2
import skimage
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import math

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../utils/')
from file_io import writeNII, createNetCDF
from image_tools import resizeImage, resizeNIIImage


if __name__=='__main__':
  parser = argparse.ArgumentParser(description='convert nifti to netcdf',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument ('-i',   '--inputfile', type = str, help = 'path to nifti file', default = "") 
  r_args.add_argument ('-resample', action='store_true', help = 'resample the output') 
  r_args.add_argument ('-n', type = int, help = 'size of image', default = 256) 
  args = parser.parse_args();

  if args.resample:
    print("converting and resampling" + args.inputfile + "\n")
  else:
    print("converting " + args.inputfile + "\n")

  infile = args.inputfile
  nii = nib.load(infile)
  img = nii.get_fdata()
  n = args.n
  if not args.resample:
    outfile = infile.replace(".nii.gz", ".nc")
    createNetCDF(outfile, args.n * np.ones(3), np.transpose(img))
  else:
    outfile = infile.replace(".nii.gz", "_" + str(n) + ".nc")
    if n is not 256:
      img_rsz = resizeImage(img, [n,n,n], interp_order=1)
    else:
      img_rsz = img
    createNetCDF(outfile, args.n * np.ones(3), np.transpose(img_rsz))
