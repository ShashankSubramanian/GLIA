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


def createNetCDFFile(filename, dimensions, variable):
  file = Dataset(filename, mode='w', format="NETCDF3_CLASSIC");
  x = file.createDimension("x", dimensions[0]);
  y = file.createDimension("y", dimensions[1]);
  z = file.createDimension("z", dimensions[2]);
  data = file.createVariable("data", "f8", ("x","y","z",));
  data[:,:,:] = variable[:,:,:];
  file.close();



if __name__=='__main__':
  parser = argparse.ArgumentParser(description='convert nifti to netcdf',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument ('-i',   '--inputfile', type = str, help = 'path to nifti file', default = "") 
  args = parser.parse_args();
  
  infile = args.inputfile
  nii = nib.load(infile)
  img = nii.get_fdata()
  outfile = infile.replace(".nii.gz", ".nc")
  createNetCDFFile(outfile, 256 * np.ones(3), np.transpose(img))
