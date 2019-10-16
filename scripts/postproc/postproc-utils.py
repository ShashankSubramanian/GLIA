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

def convertTuToBratsSeg(tu_seg):
    brats_seg = 0 * tu_seg
    brats_seg[tu_seg == 5] = 8
    brats_seg[tu_seg == 4] = 7
    brats_seg[tu_seg == 3] = 5
    brats_seg[tu_seg == 2] = 6
    brats_seg[tu_seg == 1] = 4

    return brats_seg

def computeDisplacement(d1, d2, d3):
    d = d1**2 + d2**2 + d3**2
    return np.sqrt(d)


def createRegistrationInputs(atlas_image_path, patient_image_path, results_path):
    atlas_name = "atlas"
    nii = nib.load(atlas_image_path)
    altas_seg = nii.get_fdata()
    altas_mat_img = 0 * altas_seg
    altas_mat_img[altas_seg == 7] = 1
    nib.save(nib.Nifti1Image(altas_mat_img, nii.affine), results_path + "/" + atlas_name + "_vt.nii.gz")
    altas_mat_img = 0 * altas_seg
    altas_mat_img[altas_seg == 8] = 1
    nib.save(nib.Nifti1Image(altas_mat_img, nii.affine), results_path + "/" + atlas_name + "_csf.nii.gz")
    altas_mat_img = 0 * altas_seg
    altas_mat_img[altas_seg == 5] = 1
    nib.save(nib.Nifti1Image(altas_mat_img, nii.affine), results_path + "/" + atlas_name + "_gm.nii.gz")
    altas_mat_img = 0 * altas_seg
    altas_mat_img[altas_seg == 6] = 1
    nib.save(nib.Nifti1Image(altas_mat_img, nii.affine), results_path + "/" + atlas_name + "_wm.nii.gz")

    # atlas has a tumor too; save it
    altas_mat_img = 0 * altas_seg
    altas_mat_img[altas_seg == 4] = 1
    nib.save(nib.Nifti1Image(altas_mat_img, nii.affine), results_path + "/" + atlas_name + "_tu.nii.gz")

    nii = nib.load(patient_image_path)
    patient_seg = nii.get_fdata()
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 7] = 1
    nib.save(nib.Nifti1Image(patient_mat_img, nii.affine), results_path + "/patient_vt.nii.gz")
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 8] = 1
    nib.save(nib.Nifti1Image(patient_mat_img, nii.affine), results_path + "/patient_csf.nii.gz")
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 5] = 1
    nib.save(nib.Nifti1Image(patient_mat_img, nii.affine), results_path + "/patient_gm.nii.gz")
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 6] = 1
    nib.save(nib.Nifti1Image(patient_mat_img, nii.affine), results_path + "/patient_wm.nii.gz")
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 4] = 1
    nib.save(nib.Nifti1Image(patient_mat_img, nii.affine), results_path + "/patient_tu.nii.gz")

    #create tumor masking file
    patient_mat_img = 0 * patient_seg + 1
    patient_mat_img[patient_seg == 4] = 0 #enhancing tumor mask
    patient_mat_img = gaussian_filter(patient_mat_img, sigma=2) # claire requires smoothing of masks
    nib.save(nib.Nifti1Image(patient_mat_img, nii.affine), results_path + "/patient_mask.nii.gz")


def createTumorInputs(results_path):
    atlas_name = "atlas"
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
    nii = nib.load(results_path + "/" + atlas_name + "_vt.nii.gz")
    vt = nii.get_fdata()
    vt_path_nc = results_path + "/" + atlas_name + "_vt.nc"
    createNetCDFFile(vt_path_nc, dimensions, np.transpose(vt))
    nii = nib.load(results_path + "/c0Recon_transported.nii.gz")
    c0 = nii.get_fdata()
    c0_path_nc = results_path + "/c0Recon_transported.nc"
    createNetCDFFile(c0_path_nc, dimensions, np.transpose(c0))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r_args = parser.add_argument_group('required arguments')
    r_args.add_argument ('-x',   '--results_path', type = str, help = 'path to results directory', required=True)
    r_args.add_argument ('-p',   '--patient_image_path', type = str, help = 'path to patient segmentation')
    r_args.add_argument ('-a',   '--atlas_image_path', type = str, help = 'path to altas segmentation')
    parser.add_argument ('-m',   '--mode', type = int, default = 1, help = '1: createTumorInputs', required=True)
    args = parser.parse_args();


    results_path = args.results_path
    gamma = [3E4, 6E4, 9E4, 12E4]
    if args.mode == 1:
        createRegistrationInputs(args.atlas_image_path, args.patient_image_path, args.results_path)
    elif args.mode == 2:
        createTumorInputs(results_path)     
    elif args.mode == 3:
        for g in gamma:
            results_path_reverse = results_path + "/reg-gamma-" + str(int(g)) + "/"

            if not os.path.exists(results_path_reverse):
                os.makedirs(results_path_reverse)

            tu_path = results_path + "/tumor-forward-gamma-" + str(int(g)) + "/"
            max_time = 0
            for f in os.listdir(tu_path):
                f_split = f.split("_")
                if f_split[0] == "seg":
                    f_split_2 = f_split[1].split("[")[1]
                    f_split_2 = f_split_2.split("]")[0]
                    time_step = int(f_split_2)
                    if time_step >= max_time:
                        max_time = time_step

            tu_img = tu_path + "seg_t[" + str(max_time) + "].nc"
            # make it nifti for registration
            file = Dataset(tu_img, mode='r', format="NETCDF3_CLASSIC")
            tu_seg = np.transpose(file.variables['data'])
            brats_seg = convertTuToBratsSeg(tu_seg)
            nii = nib.load(results_path + "/patient_csf.nii.gz")
            new_seg_path = results_path_reverse + "/tu-seg.nii.gz"
            nib.save(nib.Nifti1Image(brats_seg, nii.affine), new_seg_path)
    elif args.mode == 4:
        # compute metrics
        min_jacobian_norm = 1E12
        min_gamma = 0
        for g in gamma:
            results_path_reverse = args.results_path + "/reg-gamma-" + str(int(g)) + "/"
            nii = nib.load(results_path_reverse + "/displacement-field-x1.nii.gz")
            d1 = nii.get_fdata()
            nii = nib.load(results_path_reverse + "/displacement-field-x2.nii.gz")
            d2 = nii.get_fdata()
            nii = nib.load(results_path_reverse + "/displacement-field-x2.nii.gz")
            d3 = nii.get_fdata()

            d = computeDisplacement(d1, d2, d3)
            nib.save(nib.Nifti1Image(d, nii.affine), results_path_reverse + "/displacement.nii.gz")

            # jacobian_path = results_path_reverse + "/det-deformation-grad.nii.gz"
            # nii = nib.load(jacobian_path)
            # jacobian = nii.get_fdata()

            nii = nib.load(results_path_reverse + "/patient_vt.nii.gz")
            p_csf = nii.get_fdata()
            mask = p_csf
            # mask = sc.ndimage.morphology.binary_dilation(p_csf, iterations=2)
            d = np.multiply(d, mask)
            d = d.flatten()
            nrm_d = la.norm(d, ord=1)
            print("Displacement 1-norm for gamma = {} is {}".format(g, nrm_d))

            # jacobian = np.multiply(jacobian, mask)
            # one_vec = np.ones(jacobian.shape)
            # one_vec = np.multiply(one_vec, mask)
            # one_vec = one_vec.flatten()
            # jacobian = jacobian.flatten()
            # jacobian = np.abs(jacobian - one_vec)
            # nrm = la.norm(jacobian, ord=1)
            # print("Jacobian-diff 1-norm for gamma = {} is {}".format(g, nrm))

            if nrm_d < min_jacobian_norm:
                min_jacobian_norm = nrm_d
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