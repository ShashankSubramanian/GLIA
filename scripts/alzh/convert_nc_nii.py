#!/bin/python
import nibabel as nib
import os, sys
import ntpath
import numpy as np
import netCDF4
from netCDF4 import Dataset

### ------------------------------------------------------------------------ ###
def createNetCDF(filename,dimensions,variable):
    '''
    function to write a netcdf image file and return its contents
    '''
    imgfile = Dataset(filename,mode='w',format="NETCDF3_CLASSIC");
    x = imgfile.createDimension("x",dimensions[0]);
    y = imgfile.createDimension("y",dimensions[1]);
    z = imgfile.createDimension("z",dimensions[2]);
    data = imgfile.createVariable("data","f8",("x","y","z",));
    data[:,:,:] = variable[:,:,:];
    imgfile.close();

###
### ------------------------------------------------------------------------ ###
def readNetCDF(filename):
    '''
    function to read a netcdf image file and return its contents
    '''
    imgfile = Dataset(filename);
    img = imgfile.variables['data'][:]
    imgfile.close();
    return img

###
### ------------------------------------------------------------------------ ###
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

###
### ------------------------------------------------------------------------ ###
import argparse

parser = argparse.ArgumentParser(description='read objective')
parser.add_argument ('-x',           type = str,          help = 'path to the results folder');
args = parser.parse_args();

template = nib.load(os.path.join('data', 'atlas_seg_wm.nii.gz'));
for dir in os.listdir(args.x):
    if not ".nc" in dir:
        continue
    print("converting {}".format(dir))
    dat = readNetCDF(os.path.join(args.x, dir));
    dat = np.swapaxes(dat,0,2);
    output_size = tuple(dat.shape)
    filename = ntpath.basename(dir);
    filename = filename.split('.nc')[0]
    newfilename = filename + '.nii.gz';
    writeNII(dat, os.path.join(args.x,newfilename), template.affine);

