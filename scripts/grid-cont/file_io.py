import os
from sys import exit
import os.path as op
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
def writeNII(img, filename, affine=None, ref_image=None):
    '''
    function to wrote a nifti image, creates a new nifti object
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


# class FileIO:
#     '''
#     A simple class for loading and saving medical image types e.g. niftii(.nii.gz) and netcdf(.nc)
#     '''
#     def __init__(self, filename):
#         self.filename = filename
#         self.foldername = os.path.dirname(filename)
#         self.img = None
#         if op.splitext(self.filename)[1] == '.nc':
#             self.is_nifti = False            
#         elif op.splitext(op.splitext(self.filename)[0])[1] == '.nii' or op.splitext(self.filename)[1] == '.nii':            
#             self.is_nifti = True
#         else:
#             print("Unrecognised image type, should be either .nc or .nii.gz or .nii\n")
#             exit()


#     def load_image(self):
#         if is_nifti:
#             self._load_nii_image()
#         else:
#             self._load_netcdf_image()

#         return self.img

#     def save_image(self):
#         if is_nifti:
#             self._save_nii_image()
#         else:
#             self._save_netcdf_image()

#     def _load_nii_image(self):
#         img = nib.load(self.filename)
#         self.img = img.get_fdata()
#         self.affine = img.affine
        


#     def _get_nii_header(self):        

#     def _save_nii_image(self, img, new_filename,affine):
#         if affine is None:
#             data = nib.Nifti1Image(img, np.eye(4));
#         else:
#             data = nib.Nifti1Image(img, affine);
#         nib.save(data, new_filename);

#     def _load_netcdf_image(self):
#         imgfile = Dataset(self.filename);
#         self.img = imgfile.variables['data'][:]
#         imgfile.close();        

#     def _save_netcdf_image(self, img, new_filename):
#         imgfile = Dataset(new_filename, mode='w',format="NETCDF3_CLASSIC");
#         x = imgfile.createDimension("x",dimensions[0]);
#         y = imgfile.createDimension("y",dimensions[1]);
#         z = imgfile.createDimension("z",dimensions[2]);
#         data = imgfile.createVariable("data","f8",("x","y","z",));
#         data[:,:,:] = img[:,:,:];
#         imgfile.close();






