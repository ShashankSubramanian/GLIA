#!/bin/python
import nibabel as nib
import os, sys
import ntpath
import numpy as np
import scipy.ndimage as ndimage
import netCDF4
from netCDF4 import Dataset
import skimage
from skimage.util import random_noise
import scipy as sp


###
### ------------------------------------------------------------------------ ###
def resizeImage(img, new_size, interp_order):
    '''
    resize image to new_size
    '''
    factor = tuple([float(x)/float(y) for x,y in zip(list(new_size), list(np.shape(img)))]);
    return ndimage.zoom(img, factor, order=interp_order);

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
def add_noise(dat, Nx_in, Nx_out, noise_level):
    # Noise level should be in the range of [0, 1.5]
    # Empirical function between noise density and noise level
    d = (noise_level*np.linalg.norm(dat.flatten())/(70))**(2)
    sigma = 2
    dummy = np.amax(dat)*np.array(random_noise(np.zeros((Nx_in, Nx_in, Nx_in)), mode="s&p",amount = d))
    d_til = np.array(dummy) + np.array(dat)
    d_tiltil = sp.ndimage.gaussian_filter(d_til, sigma)
    d_noise = resizeImage(d_tiltil, (Nx_out, Nx_out, Nx_out) , 1)    # downsample bilinear

    d_noise = resizeImage(d_tiltil, (Nx_in, Nx_in, Nx_in) , 0)       # upsample NN
    # compute noise
    noise_level_out = np.linalg.norm(dat.flatten() - d_noise.flatten(), 2) / np.linalg.norm(dat.flatten(), 2)
    return d_noise,  noise_level_out


###
### ------------------------------------------------------------------------ ###
template = nib.load(os.path.join('data', 'atlas_seg_wm.nii.gz'));
dat = readNetCDF(os.path.join('tc', 'dataBeforeObservation.nc'));
dat_c0 = readNetCDF(os.path.join('tc', 'c0True.nc'));
d_128      = resizeImage(dat,    tuple([128,128,128]), 1);
d_128_256  = resizeImage(d_128,  tuple([256,256,256]), 0);
d_64       = resizeImage(dat,    tuple([64,64,64]),    1);
d_64_256   = resizeImage(d_64,   tuple([256,256,256]), 0);
d0_64      = resizeImage(dat_c0, tuple([64,64,64]),    1);
d0_64_256  = resizeImage(d0_64,  tuple([256,256,256]), 0);

d1_noise, nlvl = add_noise(dat, 256, 64, 0.1)
print("noise level d1: {}".format(nlvl))
d0_noise, nlvl = add_noise(dat_c0, 256, 64, 0.1)
print("noise level d0: {}".format(nlvl))


createNetCDF('data_t1_noise.nc', tuple([256,256,256]), d1_noise);
createNetCDF('data_t0_noise.nc', tuple([256,256,256]), d0_noise);
createNetCDF('data_t1_64_256.nc', tuple([256,256,256]), d_64_256);
createNetCDF('data_t0_64_256.nc', tuple([256,256,256]), d0_64_256);
createNetCDF('data_t1_128_256.nc', tuple([256,256,256]), d_128_256);
createNetCDF('data_t1_128.nc', tuple([128,128,128]), d_128);
createNetCDF('data_t1_64.nc', tuple([64,64,64]), d_64);
writeNII(np.swapaxes(d_64_256, 0,2), 'data_t1_64_256.nii.gz', template.affine);
writeNII(np.swapaxes(d_128_256, 0,2), 'data_t1_128_256.nii.gz', template.affine);
writeNII(np.swapaxes(d1_noise, 0,2), 'data_t1_noise.nii.gz', template.affine);
writeNII(np.swapaxes(d0_noise, 0,2), 'data_t0_noise.nii.gz', template.affine);

print("64:  norm ||d - PR d|| / ||d|| = {}".format(np.linalg.norm(dat.flatten() - d_64_256.flatten(), 2) / np.linalg.norm(dat.flatten(), 2)))
print("128: norm ||d - PR d|| / ||d|| = {}".format(np.linalg.norm(dat.flatten() - d_128_256.flatten(), 2) / np.linalg.norm(dat.flatten(), 2)))

