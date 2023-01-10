import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset
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
  parser = argparse.ArgumentParser(description='convert netcdf to nifti',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument ('-i',   '--inputfolder', type = str, help = 'path to netcdf file', required = True) 
  r_args.add_argument ('-r',   '--reffile', type = str, help = 'path to ref nifti file', required = True) 
  args = parser.parse_args();
  
  infolder  = args.inputfolder

  for f in os.listdir(infolder):
    #if '.nc' in f and ("i_t" in f or "seg_t" in f or "p_t" in f or "c_t" in f or "vt_t" in f):
    if '.nc' in f:
      print("Converting ", f)
      infile = os.path.join(infolder, f)
      reffile = args.reffile
      nii     = nib.load(reffile)
      outfile = infile.replace(".nc", ".nii.gz")
      ncfile = Dataset(infile, mode='r', format="NETCDF3_CLASSIC")
      dat = np.transpose(ncfile.variables['data'])
      writeNII(dat, outfile, ref_image=nii)

