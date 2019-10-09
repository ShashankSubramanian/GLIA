import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
import TumorParams
from netCDF4 import Dataset
from numpy import linalg as la

def createNetCDFFile(filename, dimensions, variable):
    file = Dataset(filename, mode='w', format="NETCDF3_CLASSIC");
    x = file.createDimension("x", dimensions[0]);
    y = file.createDimension("y", dimensions[1]);
    z = file.createDimension("z", dimensions[2]);
    data = file.createVariable("data", "f8", ("x","y","z",));
    data[:,:,:] = variable[:,:,:];
    file.close();


def createTumorInputs(results_path):
	nii = nib.load(results_path + "/" + atlas_name + "_gm.nii.gz")
    gm = nii.get_fdata()
    gm_path_nc = results_path + "/" + atlas_name + "_gm.nc"
    dimensions = 256 * np.ones(3)
    createNetCDFFile(gm_path_nc, dimensions, np.transpose(gm))
    nii = nib.load(results_path + "/" + atlas_name + "_wm.nii.gz")
    wm = nii.get_fdata()
    wm_path_nc = results_path + "/" + atlas_name + "_wm.nc"
    createNetCDFFile(wm_path_nc, dimensions, np.transpose(wm))
    nii = nib.load(results_path + "/" + atlas_name + "_csf.nii.gz")
    csf = nii.get_fdata()
    csf_path_nc = results_path + "/" + atlas_name + "_csf.nc"
    createNetCDFFile(csf_path_nc, dimensions, np.transpose(csf))
    nii = nib.load(results_path + "/c0Recon_transported.nii.gz")
    c0 = nii.get_fdata()
    c0_path_nc = results_path + "/c0Recon_transported.nc"
    createNetCDFFile(c0_path_nc, dimensions, np.transpose(c0))

if __name__=='__main__':
	arser = argparse.ArgumentParser(description='Process images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r_args = parser.add_argument_group('required arguments')
    r_args.add_argument ('-x',   '--results_path', type = str, help = 'path to results directory', required=True)
    parser.add_argument ('-m',   '--mode', type = int, default = 1, help = '1: createTumorInputs', required=True)
    args = parser.parse_args();

    if args.mode == 1:
    	createTumorInputs(arg.results_path)
    elif args.mode == 2:
    	gamma = [3E4, 9E4, 15E4]
    	# compute metrics
        min_jacobian_norm = 1E10
        min_gamma = 0
        for g in gamma:
            results_path_reverse = args.results_path + "/reg-gamma-" + str(int(g)) + "/"
            jacobian_path = results_path_reverse + "/det-deformation-grad.nii.gz"
            nii = nib.load(jacobian_path)
            jacobian = nii.get_fdata()
            nii = nib.load(results_path_reverse + "/patient_csf.nii.gz")
            p_csf = nii.get_fdata()
            mask = sc.ndimage.morphology.binary_dilation(p_csf, iterations=2)
            jacobian = np.multiply(jacobian, mask)
            nrm = la.norm(jacobian)
            print("jacobian norm for gamma = {} is {}".format(g, nrm))
            if nrm < min_jacobian_norm:
                min_jacobian_norm = nrm
                min_gamma = g 
        print("Mass-effect parameter is {}".format(min_gamma))
        if min_gamma > 9E4:
            print("Mass-effect is high")
        elif min_gamma > 3E4:
            print("Mass-effect is moderate")
        else:
            print("Mass-effect is low")
    else:
    	print("utils mode incorrect. exiting...")