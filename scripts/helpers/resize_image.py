import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import math
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../utils/')
import file_io as fio
import image_tools as imgtools
###
### ------------------------------------------------------------------------ ###
def resample(fname, ndim, order):
    """ Resamples image.
    """
    ndims = tuple([ndim, ndim, ndim]);
    ext = ".n" + fname.split('.n')[-1]
    fname_out = fname.split('.n')[0]
    if ext in ['.nii.gz', '.nii']:
        img = nib.load(fname)
        scale = img.get_fdata().shape[0]/ndim 
        new_affine = np.copy(img.affine)
        row,col = np.diag_indices(new_affine.shape[0])
        new_affine[row,col] = np.array([-scale,-scale,scale,1]);
        img_resized = nib.processing.resample_from_to(img, (np.multiply(1./scale, img.shape).astype(int), new_affine), order=order)
        fio.writeNII(img_resized.get_fdata(), fname_out + "_nx" + str(ndim) + ext, affine=img_resized.affine, ref_image=img_resized);
    else:
        img_d = fio.readNetCDF(fname);
        img_resized = imgtools.resizeImage(img_d, ndims, order);
        fio.createNetCDF(fname_out + "_nx" + str(ndim) + ext, ndims, img_resized);

###
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='resample image',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument ('-i',   '--inputfile', type = str, help = 'path to file', required = True) 
  r_args.add_argument ('-n',   '--ndim', type = int, help = 'new dimension', required = True) 
  r_args.add_argument ('-ord',   '--iporder', type = int, help = 'interp order', required = True) 
  args = parser.parse_args();
  
  infile  = args.inputfile
  n = args.ndim
  resample(infile, n, args.iporder) 

