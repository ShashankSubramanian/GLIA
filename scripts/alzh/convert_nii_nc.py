#!/bin/python


import nibabel as nib
import os, sys
import ntpath
import numpy as np
import netCDF4
from netCDF4 import Dataset
import argparse 

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
parser = argparse.ArgumentParser(description='read objective')
parser.add_argument ('-x',           type = str,          help = 'path to the results folder');
args = parser.parse_args();

for dir in os.listdir(args.x):
    if not ".nii.gz" in dir:
        continue
    print("converting {}".format(dir))
    dat = nib.load(os.path.join(args.x,dir)).get_fdata()
    filename = ntpath.basename(dir);
    filename = filename.split('.nii.gz')[0]
    newfilename = filename + '.nc';
    createNetCDF(os.path.join(args.x, newfilename), np.shape(dat), np.swapaxes(dat,0,2));
