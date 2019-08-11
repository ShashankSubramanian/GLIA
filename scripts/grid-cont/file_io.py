import os
import numpy as np
from netCDF4 import Dataset
import nibabel as nib


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
def writeNII(img, filename, affine=None):
    '''
    function to wrote a nifti image, creates a new nifti object
    '''
    if affine is None:
        data = nib.Nifti1Image(img, np.eye(4));
    else:
        data = nib.Nifti1Image(img, affine);
    nib.save(data, filename);
