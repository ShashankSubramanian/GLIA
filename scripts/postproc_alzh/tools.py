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
def add_noise_gauss(dat, Nx_in, Nx_out, noise_level, mask, d0=False):

    #d = (noise_level*np.linalg.norm(dat.flatten())/(70))**(2)
    m = np.amax(dat)
    d = (np.sum(np.where(dat > 0.1 *m, 1, 0).flatten())/ np.sum(np.where(mask > 0,1,0))) *noise_level
   # d = (noise_level*np.linalg.norm(dat.flatten())/(70))**(2)
    print("     - d = {}".format(d))
    sigma = 2 if not d0 else 1
    maxval = 0.8 * np.amax(dat)
    sp_noise = maxval*np.array(random_noise(np.zeros((Nx_in, Nx_in, Nx_in)), mode="s&p",amount = d))
    sp_noise = np.multiply(sp_noise, resizeImage(np.random.rand(64,64,64), (Nx_in, Nx_in, Nx_in), 0 ))
    sp_noise = np.multiply(sp_noise, mask)
    sp_noise_smooth = sp.ndimage.gaussian_filter(sp_noise, 1)

    d_til = sp_noise_smooth + dat
    #d_til_smooth = sp.ndimage.gaussian_filter(d_til, sigma)
    d_til_smooth = sp.ndimage.gaussian_filter(d_til, 1)

    d_noise_lres = resizeImage(d_til_smooth, (Nx_out, Nx_out, Nx_out) , 1)    # downsample bilinear
    d_noise      = resizeImage(d_noise_lres, (Nx_in, Nx_in, Nx_in) , 0)       # upsample NN
    # compute noise
    noise_level_out = np.linalg.norm(dat.flatten() - d_noise.flatten(), 2) / np.linalg.norm(dat.flatten(), 2)
    return d_noise, d_noise_lres,  noise_level_out


###
### ------------------------------------------------------------------------ ###



