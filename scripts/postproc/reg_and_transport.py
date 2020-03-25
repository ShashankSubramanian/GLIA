import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
import TumorParams
from netCDF4 import Dataset
from numpy import linalg as la
import math
from postproc_utils import writeNII, createNetCDFFile


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


def create_netcdf(filename, dimensions, variable):
    file = Dataset(filename, mode='w', format="NETCDF3_CLASSIC");
    x = file.createDimension("x", dimensions[0]);
    y = file.createDimension("y", dimensions[1]);
    z = file.createDimension("z", dimensions[2]);
    data = file.createVariable("data", "f8", ("x","y","z",));
    data[:,:,:] = variable[:,:,:];
    file.close();

def convert_tu_brats_seg(tu_seg):
    brats_seg = 0 * tu_seg
    brats_seg[tu_seg == 5] = 8
    brats_seg[tu_seg == 4] = 7
    brats_seg[tu_seg == 3] = 5
    brats_seg[tu_seg == 2] = 6
    brats_seg[tu_seg == 1] = 4

    return brats_seg


def create_sbatch_header(results_path, compute_sys='frontera'):
    bash_filename = results_path + "/reg-and-transport.sh"
    print("creating job file in ", results_path)

    if compute_sys == 'frontera':
        queue = "normal"
        num_nodes = str(4)
        num_cores = str(128)
    elif compute_sys == 'maverick2':
        queue = "v100"
        num_nodes = str(1)
        num_cores = str(1)
    elif compute_sys == 'longhorn':
        queue = "v100"
        num_nodes = str(1)
        num_cores = str(1)
    elif compute_sys == 'stampede2':
        queue = "skx-normal"
        num_nodes = str(6)
        num_cores = str(128)
    else:
        queue = "normal"
        num_nodes = str(1)
        num_cores = str(1)

    bash_file = open(bash_filename, 'w')
    bash_file.write("#!/bin/bash\n\n");
    bash_file.write("#SBATCH -J mass-effect-cpl\n");
    bash_file.write("#SBATCH -o " + results_path + "/coupling_solver_log.txt\n")
    bash_file.write("#SBATCH -p " + queue + "\n")
    bash_file.write("#SBATCH -N " + num_nodes + "\n")
    bash_file.write("#SBATCH -n " + num_cores + "\n")
    bash_file.write("#SBATCH -t 03:00:00\n\n")
    bash_file.write("source ~/.bashrc\n")

    bash_file.write("\n\n")
    bash_file.close()

    return bash_filename


def create_label_maps(atlas_image_path, patient_image_path, results_path, atlas_name):
    nii = nib.load(atlas_image_path)
    altas_seg = nii.get_fdata()
    altas_mat_img = 0 * altas_seg
    altas_mat_img[altas_seg == 7] = 1
    writeNII(altas_mat_img, results_path + "/" + atlas_name + "_vt.nii.gz", ref_image=nii)
    altas_mat_img = 0 * altas_seg
    altas_mat_img[altas_seg == 8] = 1
    writeNII(altas_mat_img, results_path + "/" + atlas_name + "_csf.nii.gz", ref_image=nii)
    altas_mat_img = 0 * altas_seg
    altas_mat_img[altas_seg == 5] = 1
    writeNII(altas_mat_img, results_path + "/" + atlas_name + "_gm.nii.gz", ref_image=nii)
    altas_mat_img = 0 * altas_seg
    altas_mat_img[np.logical_or(altas_seg == 6, altas_seg == 8)] = 1
    writeNII(altas_mat_img, results_path + "/" + atlas_name + "_wm.nii.gz", ref_image=nii)

    nii = nib.load(patient_image_path)
    patient_seg = nii.get_fdata()
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 7] = 1
    writeNII(patient_mat_img, results_path + "/patient_vt.nii.gz", ref_image=nii)
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 8] = 1
    writeNII(patient_mat_img, results_path + "/patient_csf.nii.gz", ref_image=nii)
    patient_mat_img = 0 * patient_seg
    patient_mat_img[patient_seg == 5] = 1
    writeNII(patient_mat_img, results_path + "/patient_gm.nii.gz", ref_image=nii)
    patient_mat_img = 0 * patient_seg
    patient_mat_img[(np.logical_or(patient_seg == 6, patient_seg == 8))] = 1
#    patient_mat_img[np.logical_or(np.logical_or(patient_seg == 6, patient_seg == 8), patient_seg == 4)] = 1
    writeNII(patient_mat_img, results_path + "/patient_wm.nii.gz", ref_image=nii)
    ### masking file
    patient_mat_img = 0 * patient_seg + 1
    patient_mat_img[np.logical_or(patient_seg == 4, patient_seg == 1)] = 0 #enhancing tumor mask
    patient_mat_img = gaussian_filter(patient_mat_img, sigma=2) # claire requires smoothing of masks
    writeNII(patient_mat_img, results_path + "/patient_mask.nii.gz", ref_image=nii)

def register(claire_bin_path, results_path, atlas_name, bash_filename):
    bash_file = open(bash_filename, 'a')
    cmd = "ibrun " + claire_bin_path + "/claire -mrc 3 " + results_path + "/" + atlas_name + "_vt.nii.gz " + results_path + "/" + atlas_name \
                + "_gm.nii.gz " + results_path + "/" + atlas_name + "_wm.nii.gz "\
                + "-mtc 3 " + results_path + "/patient_vt.nii.gz " + results_path + "/patient_gm.nii.gz " + results_path + "/patient_wm.nii.gz " \
                + "-mask " + results_path + "/patient_mask.nii.gz " \
                + "-nx 256 -train reduce -jbound 5e-2 -regnorm h1s-div -opttol 5e-2 -maxit 50 -krylovmaxit 50 -velocity -detdefgrad -deffield -defmap -residual -x "\
                + results_path + "/"\
                + " -monitordefgrad -verbosity 1 -disablerescaling -format nifti -sigma 2" + " &> " + results_path + "/registration_log.txt";
#                + "-mask " + results_path + "/patient_mask.nii.gz " \
#    cmd = "ibrun " + claire_bin_path + "/claire -mrc 4 " + results_path + "/" + atlas_name + "_vt.nii.gz " + results_path + "/" + atlas_name + "_csf.nii.gz " + results_path + "/" + atlas_name \
#                + "_gm.nii.gz " + results_path + "/" + atlas_name + "_wm.nii.gz "\
#                + "-mtc 4 " + results_path + "/patient_vt.nii.gz " + results_path + "/patient_csf.nii.gz " + results_path + "/patient_gm.nii.gz " + results_path + "/patient_wm.nii.gz " \
#                + "-mask " + results_path + "/patient_mask.nii.gz " \
#                + "-nx 256 -train reduce -jbound 5e-2 -regnorm h1s-div -opttol 1e-2 -maxit 30 -krylovmaxit 50 -velocity -detdefgrad -deffield -residual -x "\
#                + results_path + "/"\
#                + " -monitordefgrad -verbosity 1 -disablerescaling -format nifti -sigma 2" + " &> " + results_path + "/registration_log.txt";
    bash_file.write(cmd)
    bash_file.write("\n\n")
    bash_file.close()

    # return so that transport can be done immediately after
    return bash_filename

# transport
def transport(claire_bin_path, results_path, bash_filename, transport_file_name):
    # run this after reg always
    bash_file = open(bash_filename, 'a')

    cmd = "ibrun " + claire_bin_path + "/clairetools -v1 " + results_path + "/velocity-field-x1.nii.gz -v2 " + results_path + "/velocity-field-x2.nii.gz -v3 " + results_path + "/velocity-field-x3.nii.gz -ifile "\
                   + results_path + "/" + transport_file_name + ".nii.gz -xfile " + results_path + "/" + transport_file_name + "_transported.nii.gz -deformimage -iporder 1" + " &> " + results_path + "/transport_log.txt";

    bash_file.write(cmd)
    bash_file.write("\n\n")
    bash_file.close()

    return bash_filename


def convert_and_move_moi(path, out_path):
    file = Dataset(path, mode='r', format="NETCDF3_CLASSIC")
    c0 = np.transpose(file.variables['data'])
    nii = nib.load(out_path + "/patient_vt.nii.gz") 
    writeNII(c0, out_path + "/c0Recon.nii.gz", ref_image=nii)


def write_phi_p(phi, p, path):
    phi_name = path + "/phi-mesh-scaled-transported.txt"
    p_name = path + "/p-rec-scaled-transported.txt"

    phi_file = open(phi_name, 'w')
    p_file = open(p_name, 'w')

    phi_file.write("sigma = 0.0245437, spacing = 0.0490874\ncenters = [\n")
    p_file.write("p = [\n")

    ### compute gaussian also
#    x_1d = np.linspace(0, 2*math.pi - 2*math.pi/256, 256)
#    x,y,z = np.meshgrid(x_1d, x_1d, x_1d)
#    gauss = np.zeros((256,256,256))
#    sigma = 0.0245437
    for k in range(len(phi)):
        phi_file.write(str(phi[k][2]) + ", " + str(phi[k][1]) + ", " + str(phi[k][0]) + "\n")
        p_file.write(str(p[k]) + "\n")
#        gauss += p[k] * np.exp(-((x - phi[k][0])**2 + (y - phi[k][1])**2 + (z - phi[k][2])**2) / (2*sigma*sigma))
#    nii = nib.load(path + "/patient_vt.nii.gz")
#    writeNII(gauss, path + "/ic-compute-transported.nii.gz", ref_image = nii)
    phi_file.write("];")
    p_file.write("];")
    phi_file.close()
    p_file.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r_args = parser.add_argument_group('required arguments')
    r_args.add_argument ('-tp',   '--transport', type = int, help = 'mode to transport', default = 0) 
    args = parser.parse_args();
    tp = args.transport

    scripts_path = os.getcwd() + "/.."
    patient_path = scripts_path + "/../brain_data/t16/t16-case7-seg.nii.gz"
    map_of_interest = scripts_path + "/../results/rd-inv-t16-case7/tumor_inversion/nx256/obs-1.0/c0Recon.nc"

    ### use the rec phis and ps directly(?)
    phi_rec = scripts_path + "/../results/rd-inv-t16-case7/tumor_inversion/nx256/obs-1.0/phi-mesh-scaled.txt"
    p_rec = scripts_path + "/../results/rd-inv-t16-case7/tumor_inversion/nx256/obs-1.0/p-rec-scaled.txt"

    phix, phiy, phiz = np.loadtxt(phi_rec, comments = ["]", "#"], delimiter=',', skiprows=2, unpack=True);
    p_vec = np.loadtxt(p_rec,  comments = ["]", "#"], skiprows=1);
    phi = []

    for x,y,z in zip(phix,phiy,phiz):
        phi.append(tuple([z,y,x]));

    patient = nib.load(patient_path).get_fdata()
    atlas_path = scripts_path + "/../brain_data/atlas/"
    results_path = scripts_path + "/../results/reg-t16-case7/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    claire_path = scripts_path + "/../../claire-dev/bingpu/"
    bash_file = create_sbatch_header(results_path, compute_sys = "longhorn")

#    atlases = ["atlas-2", "atlas-4", "atlas-5", "atlas-6"]
#    atlases = ["atlas-4", "atlas-5", "atlas-6"]
    atlases = ["atlas-1"]
    for atlas in atlases:
        r_path = results_path + atlas
        if not os.path.exists(r_path):
            os.makedirs(r_path)
        ### create inputs for registration
        create_label_maps(atlas_path + atlas + ".nii.gz", patient_path, r_path, atlas_name = atlas)
        convert_and_move_moi(map_of_interest, r_path)
        ### create reg job scripts
        bash_file = register(claire_path, r_path, atlas, bash_file)
        ### create reg-transport script
        bash_file = transport(claire_path, r_path, bash_file, "c0Recon")
        bash_file = transport(claire_path, r_path, bash_file, "patient_vt")

    fio = open(bash_file, 'a')
    fio.write("python3 " + scripts_path + "/postproc/reg_and_transport.py -tp 3\n")
    fio.write("python3 " + scripts_path + "/postproc/reg_and_transport.py -tp 2\n")
    fio.close()
    sx = sy = sz = 256 / (2*math.pi)
    ### transport disp
    if tp == 1:
        for atlas in atlases:
            r_path = results_path + atlas
            print("displacing coordinates for {}".format(atlas))
            dx = nib.load(r_path + "/displacement-field-x1.nii.gz").get_fdata()
            dy = nib.load(r_path + "/displacement-field-x2.nii.gz").get_fdata()
            dz = nib.load(r_path + "/displacement-field-x3.nii.gz").get_fdata()

            ### displace the phis
            phi_new = []
            for k in range(len(phi)):
                idx = (int(round(phi[k][0]*sz)), int(round(phi[k][1]*sy)), int(round(phi[k][2]*sz)))
                px = phi[k][0] - dx[idx]
                py = phi[k][1] - dy[idx]
                pz = phi[k][2] - dz[idx]
                phi_new.append(tuple([px,py,pz]))

            write_phi_p(phi_new, p_vec, r_path)
    elif tp == 2:
        ### convert transported c0 to nc
        for atlas in atlases:
            print("transporting maps for {}".format(atlas))
            r_path = results_path + atlas
            c0_path = r_path + "/c0Recon_transported.nii.gz"
            c0 = nib.load(c0_path).get_fdata()
            createNetCDFFile(r_path + "/c0Recon_transported.nc", 256 * np.ones(3), np.transpose(c0))
    elif tp == 3:
        for atlas in atlases:
            print("deforming coordinates for {}".format(atlas))
            r_path = results_path + atlas
            dx = nib.load(r_path + "/deformation-map-x1.nii.gz").get_fdata()
            dy = nib.load(r_path + "/deformation-map-x2.nii.gz").get_fdata()
            dz = nib.load(r_path + "/deformation-map-x3.nii.gz").get_fdata()

            ### displace the phis
            phi_new = []
            for k in range(len(phi)):
                idx = (int(round(phi[k][0]*sz)), int(round(phi[k][1]*sy)), int(round(phi[k][2]*sz)))
                px = dz[idx]
                py = dy[idx]
                pz = dx[idx]
                phi_new.append(tuple([px,py,pz]))

            write_phi_p(phi_new, p_vec, r_path)
