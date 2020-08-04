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


def writeNII(img, filename, affine=None, ref_image=None):
    '''
    function to write a nifti image, creates a new nifti object
    '''
    if ref_image is not None:
        data = nib.Nifti1Image(img, affine=ref_image.affine, header=ref_image.header);
        data.header['datatype'] = 64
        data.header['glmax'] = np.max(img)
        data.header['glmin'] = np.min(img)
    elif affine is not None:
        data = nib.Nifti1Image(img, affine=affine);
    else:
        data = nib.Nifti1Image(img, np.eye(4))

    nib.save(data, filename);

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='create a tumor mask from a segmentation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument ('-i',   '--inputfile', type = str, help = 'path to tumor segmentation', required = True) 
  args = parser.parse_args();
  
  infile  = args.inputfile
  outfile = infile.replace("seg_tu", "mask")
  nii     = nib.load(infile)
  seg     = nii.get_fdata()
  mask    = 1 - np.logical_or(seg == 1, np.logical_or(seg == 2, seg == 4))
  writeNII(mask, outfile, ref_image=nii)

