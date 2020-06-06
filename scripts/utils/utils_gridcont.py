import matplotlib as mpl
import matplotlib.pyplot as plt
import os,argparse, sys

import numpy as np
import nibabel as nib
import nibabel.processing

import file_io as fio
import ntpath
import math
from netCDF4 import Dataset
import heapq
import glob

import networkx as nx
import scipy
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import re, shutil, tempfile

import image_tools as imgtools

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


###
### ------------------------------------------------------------------------ ###
def update_config(path_log, path_config):
    """ Extracts reconstructed rho and kappa values from the log file and
        updates the config file for the next level.
    """
    rho, k =  extract_from_log(path_log, "solver_log.txt")
    with open(os.path.join(path_config, "solver_config.txt"), "r") as f:
        lines = f.readlines()
    with open(os.path.join(path_config, "solver_config.txt"), "w")  as f:
        for line in lines:
            if "init_rho=" in line:
                f.write("init_rho="+str(rho)+"\n")
                continue
            if "init_k=" in line:
                f.write("init_k="+str(k)+"\n")
                continue
            f.write(line)
###
### ------------------------------------------------------------------------ ###
def extract_from_log(path, filename):
    empty = False
    exist = False
    try:
      with open(os.path.join(path, 'reconstruction_info.dat'), 'r') as rec_file:
        lines = rec_file.readlines()
        line = lines[-1]
        rho = float(line.split()[0])
        k = float(line.split()[1])
    except Exception as e:
      print("[] reconstruction file {}/reconstruction_info.dat could not be opened:\n  -->  {}".format(path, e))
      print("[] trying to read log file...")
      if os.path.exists(os.path.join(path, filename)):
          exist =True;
          print("reading logifle:", os.path.join(path, filename))
          with open(os.path.join(path,filename), 'r',  encoding="utf-8") as f:
              lines  = f.readlines();
              no = 0;
              if len(lines) > 0:
                  for line in lines:
                      if " ### estimated reaction coefficients:                  ###" in line:
                          rho = float(lines[no+1].split("r1:")[-1].split(",")[0])
                      if " ### estimated diffusion coefficients:                 ### " in line:
                          k = float(lines[no+1].split("k1:")[-1].split(",")[0]);
                      no += 1;
                  print( bcolors.OKBLUE + " ... setting init guess (rho, k) = (",rho,",",k,") " + bcolors.ENDC);
              else:
                  empty = True;
      if empty or not exist:
          print("Error: tumor solver log file in ", path, "does not exist.");
    return rho, k

###
### ------------------------------------------------------------------------ ###
def resample(path, fname, ndim):
    """ Resamples image. If nifty file format, function tries to load reference image.
    """
    order = 1
    ndims = tuple([ndim, ndim, ndim]);
    if "seg" in fname or "obs" in fname or "mask" in fname:
        order = 0
        print('segmentation/mask found, using NN interpolation')
    ext = ".n" + fname.split('.n')[-1]
    fname_out = fname.split('_nx')[0].split('.')[0]
    if ext in ['.nii.gz', '.nii']:
        img = nib.load(os.path.join(path,fname))
        scale = img.get_fdata().shape[0]/ndim 
        new_affine = np.copy(img.affine)
        row,col = np.diag_indices(new_affine.shape[0])
        new_affine[row,col] = np.array([-scale,-scale,scale,1]);
        img_resized = nib.processing.resample_from_to(img, (np.multiply(1./scale, img.shape).astype(int), new_affine), order=order)

        #img_resized = imgtools.resizeNIIImage(img, ndims, interp_order=order)
        fio.writeNII(img_resized.get_fdata(), os.path.join(path, fname_out + "_nx" + str(ndim) + ext), affine=img_resized.affine, ref_image=img_resized);
    else:
        img_d = fio.readNetCDF(os.path.join(path, fname));
        img_resized = imgtools.resizeImage(img_d, ndims, order);
        fio.createNetCDF(os.path.join(path, fname_out + "_nx" + str(ndim) + ext), ndims, img_resized);

###
### ------------------------------------------------------------------------ ###
def resample_input(path, fname, ndim, order=0):
    """ Resamples the template image to new dimension. """
    ext = ".n" + fname.split('.n')[-1]
    filename = fname.split('_nx')[0]
    dim_descr = '_nx120x120x77' if ndim==128 else '_nx60x60x38'
    if ext in ['.nii.gz', '.nii']:
        p_seg = nib.load(os.path.join(path,fname))
        p_seg_d = p_seg.get_fdata()
        header = p_seg.header
        affine = p_seg.affine
        # 256 --> ndim
        scale = 256/ndim #int(p_seg_d.shape[0] / ndim)
        affine_coarse = np.copy(affine)
        row,col = np.diag_indices(affine_coarse.shape[0])
        affine_coarse[row,col] = np.array([-scale,-scale,scale,1]);
        p_seg_coarse = nib.processing.resample_from_to(p_seg, (np.multiply(1./scale, p_seg.shape).astype(int), affine_coarse), order=order)
        # save template (brats hdr)
        nib.save(p_seg_coarse, os.path.join(path, filename + dim_descr + ext));
         
        img_resized_regular_256 = imgtools.resizeNIIImage(p_seg, tuple([256, 256, 256]), interp_order=order)
        if ndim==128:
            fio.writeNII(img_resized_regular_256.get_fdata(), os.path.join(path, filename + '_nx' + str(256) + ext), ref_image=img_resized_regular_256)
        img_resized_regular = nib.processing.resample_from_to(img_resized_regular_256, (np.multiply(1./scale, img_resized_regular_256.shape).astype(int), affine_coarse), order=order)
        #img_resized_regular = imgtools.resizeNIIImage(img_resized_regular_256, tuple([ndim, ndim, ndim]), interp_order=order)
        fio.writeNII(img_resized_regular.get_fdata(), os.path.join(path, filename + '_nx' + str(ndim) + ext), ref_image=img_resized_regular)
    else:
        p_seg_d = fio.readNetCDF(os.path.join(path,fname))
        p_seg_d_coarse  = imgtools.resizeImage(p_seg_d, tuple([ndim, ndim, ndim]), interp_order=order)
        fio.createNetCDF(os.path.join(path, filename + '_nx' + str(ndim) + ext), p_seg_d_coarse.shape, p_seg_d_coarse)


###
### ------------------------------------------------------------------------ ###
def compute_observation_mask(path, segfile, obs_lambda, labels, dim=256, suffix='', ref_image=None):
    ext = ".n" + segfile.split(".n")[-1]
    if ext in [".nii.gz", ".nii"]:
        segi = nib.load(os.path.join(path, segfile))
        seg = segi.get_fdata()
    else:
        seg = np.swapaxis(fio.readNetCDF(os.path.join(path, segfile)), 0, 2)

    # TC and WT masks
    en = (data==labels['en']).astype(float)
    nec = (data==labels['nec']).astype(float)
    ed = (data==labels['ed']).astype(float)
    tc = np.logical_or(en, nec).astype(float)  # TC = EN + NEC
    wt = np.logical_or(tc, ed).astype(float)   # WT = EN + NEC + ED

    obs = tc + obs_lambda * (1 - wt);
    dims = [dim, dim, dim];
    if ext in [".nii.gz", ".nii"]:
        if ref_image is not None:
            ref_img = nib.load(ref_image)
            fio.writeNII(obs, os.path.join(path,"obs_mask" + suffix + ext), affine=ref_img.affine, ref_image=ref_img);
        else:
            fio.writeNII(obs, os.path.join(path,"obs_mask" + suffix + ext));
    else:
        fio.createNetCDF(os.path.join(path,"obs_mask" + suffix + ext), dims, obs);



###
### ------------------------------------------------------------------------ ###
def convert_images_to_orig_size(input_path, reference_image_path, gridcont=False):
    """ Resamples the output images back to the original dimension (same dimension as reference image).
    """
    levels = [64,128,256] if gridcont else [256];
    templates = {}
    ref_img = nib.load(reference_image_path);

    print('[] converting nc files to nii.gz files in brats hdr.')
    refshape = ref_img.shape 
    suf = {}
    for l in levels:
        suf[l] = "_{}x{}x{}.nii.gz".format(int(refshape[0]/scale), int(resfshape[1]/scale), int(refshape[2]/scale))
        tu_out_path = os.path.join(input_path, 'nx'+str(l)+'/');
        # create template image 
        new_affine = np.copy(ref_img.affine)
        row,col = np.diag_indices(new_affine.shape[0])
        new_affine[row,col] = np.array([-scale, -scale, scale,1])
        if l < 256:
            resampled_template = nib.processing.resample_from_to(ref_img, (np.multiply(1./scale, ref_img.shape).astype(int), new_affine))
        else:
            resampled_template = ref_img
        templates[str(l)] = resampled_template
        nib.save(resampled_template, os.path.join(tu_out_path,"template" + suf[l] ));

        template = templates[str(l)];
        print(" == converting files of level {} ==".format(l))
        dirs = os.listdir(tu_out_path)
        for dir in dirs:
            if not "obs" in dir:
                continue;
            ncfiles_tumor = imgtools.getNetCDFImageList(os.path.join(tu_out_path, dir + '/'));
            for f in ncfiles_tumor:
                data = fio.readNetCDF(f);
                data = np.swapaxes(data,0,2);
                order = 0 if 'seg' in f else 1
                newdata = imgtools.resizeImage(data, tuple(template.shape), order);
                output_size = tuple(template.shape)
                filename = ntpath.basename(f);
                filename = filename.split(".")[0]
                newfilename = filename + '_' + "".join([str(x) for x in list(output_size)]) + '.nii.gz';
                print("  .. creating {}".format(newfilename))
                fio.writeNII(newdata, os.path.join(os.path.join(tu_out_path, dir), newfilename), template.affine);


###
### ------------------------------------------------------------------------ ###
def convert_netcdf_to_nii(input_filename, output_filename, affine=None, ref_image=None):
    ref_image = nib.load(ref_image)
    img = fio.readNetCDF(input_filename)
    img = np.swapaxes(img,0,2);
    fio.writeNII(img, output_filename, affine=affine, ref_image=ref_image);

###
### ------------------------------------------------------------------------ ###
def compute_concomp(data, level):
    img = data > 1E-4

    structure = np.ones((3, 3, 3), dtype=np.int);
    comps       = {}
    count       = {}
    sums        = {}
    relmass     = {}
    xcm_data_px = {}
    xcm_data    = {}
    comps_sorted       = {}
    relmass_sorted     = {}
    xcm_data_px_sorted = {}
    xcm_data_sorted    = {}

    total_mass = 0
    labeled, ncomponents = scipy.ndimage.measurements.label(img, structure);
    print("[] computing connected components of TC data:" + bcolors.OKBLUE + " {} components found".format(ncomponents),
    bcolors.ENDC)
    for i in range(ncomponents):
        comps[i] = (labeled == i+1)
        a, b = scipy.ndimage.measurements._stats(comps[i])
        total_mass += b
    for i in range(ncomponents):
        count[i], sums[i]  = scipy.ndimage.measurements._stats(comps[i])
        relmass[i] = sums[i]/float(total_mass);
        xcm_data_px[i] = scipy.ndimage.measurements.center_of_mass(comps[i])
        xcm_data[i] = tuple([2 * math.pi * xcm_data_px[i][0] / float(level), 2 * math.pi * xcm_data_px[i][1] / float(level), 2 * math.pi * xcm_data_px[i][2] / float(level)]);

    # sort components according to their size
    sorted_rmass = sorted(relmass.items(), key=lambda x: x[1], reverse=True);
    perm = {}
    temp = {}
    labeled_sorted = np.zeros_like(labeled);
    for i in range(len(sorted_rmass)):
        perm[i]               = sorted_rmass[i][0] # get key from sorted list
        comps_sorted[i]       = comps[perm[i]];
        relmass_sorted[i]     = relmass[perm[i]];
        xcm_data_px_sorted[i] = xcm_data_px[perm[i]];
        xcm_data_sorted[i]    = xcm_data[perm[i]];

        temp[i] = (labeled == perm[i]+1).astype(int)*(i+1);
        labeled_sorted += temp[i];

    # return labeled, comps, ncomponents, xcm_data_px, xcm_data, relmass;
    return labeled_sorted, comps_sorted, ncomponents, xcm_data_px_sorted, xcm_data_sorted, relmass_sorted


###
### ------------------------------------------------------------------------ ###
def compute_connected_components(args, labels=None, obs_th=-1):
    """ Computes connecte components of tumor data and writes file with component data.
    """
    res_path  = os.path.join(args.input_path);
    init_path = os.path.join(args.input_path, '../init');
    level = int(res_path.split("nx")[-1].split("/")[0]);

    data_not_from_seg = labels==None

    if data_not_from_seg:
        fname = "data"
    else:        
        fname = "patient_seg"
    ref_img = None
    success = True
    try:
        ext = ".nc"
        data = np.swapaxes(fio.readNetCDF(os.path.join(init_path, fname + ext)), 0, 2)
        dims = data.shape
    except Exception as c:
        success = False
    if not success:
        try:
            ext = ".nii.gz"
            ref_img = nib.load(os.path.join(init_path, fname + ext))
            data = ref_img.get_fdata()
            dims = data.shape;
        except Exception as c:
            print(c)
            return
    if labels is not None:
        if 'tc' in labels:
            tc = (data==labels['tc']).astype(float)
        else:
            en = (data==labels['en']).astype(float)
            nec = (data==labels['nec']).astype(float)
            tc = np.logical_or(en, nec).astype(float)
        data = tc
    # observation operator:
    if obs_th > 0:
        print(" .. applying rel. obs-threshold: {}".format(obs_th))
        m = np.amax(data.flatten())
        th = obs_th * m
        data = np.where(data > th, data, 0)
    if ext == ".nc":
        fio.createNetCDF(os.path.join(args.output_path,'target_data_nx'+str(level)+'.nc'), np.shape(data), np.swapaxes(data,0,2));
    else:
        fio.writeNII(data, os.path.join(args.output_path,'target_data_nx'+str(level)+'.nii.gz'), affine=ref_img.affine, ref_image=ref_img);

    if data_not_from_seg:
        data = (data>0).astype(float)
    # compute connected components
    labeled, comps_data, ncomps_data, xcm_data_px, xcm_data, relmass = compute_concomp(data, level);

    # write labels
    fname = os.path.join(args.output_path, 'data_comps_nx'+str(level))
    if ref_img is not None:
        fio.writeNII(labeled, fname + ".nii.gz", affine=ref_img.affine, ref_image=ref_img);
    else:
        fio.createNetCDF(fname + ".nc", np.shape(labeled), np.swapaxes(labeled,0,2));

    # write to file
    concomp_file = open(os.path.join(res_path,'dcomp.dat'),'w');
    concomp_file.write("#components:\n")
    concomp_file.write(str(ncomps_data) + "\n")
    concomp_file.write("center of mass:\n")
    for i in range(ncomps_data):
         concomp_file.write(str(xcm_data[i][2]) + ',' + str(xcm_data[i][1]) + ',' + str(xcm_data[i][0]) + "\n" )
    concomp_file.write("relative mass:\n")
    for i in range(ncomps_data):
        concomp_file.write(str(relmass[i]) + "\n" )
    concomp_file.close();

    hx = 2*math.pi/float(level);
    phi_cm_data_file = open(os.path.join(res_path,'phi-cm-data.txt'),'w');
    p_cm_data_file = open(os.path.join(res_path,'p-cm-data.txt'),'w');
    phi_cm_data_file.write(" sigma = %1.8f, spacing = %1.8f\n" % (args.sigma * hx, 2*hx));
    phi_cm_data_file.write(" centers = [\n")
    p_cm_data_file.write("p = [\n")
    for i in range(ncomps_data):
        phi_cm_data_file.write(" %1.8f, %1.8f, %1.8f\n" % (xcm_data[i][2], xcm_data[i][1], xcm_data[i][0]));
        p_cm_data_file.write(" %1.8f\n" % 0.);
    phi_cm_data_file.write("];")
    p_cm_data_file.write("];")

    # compute Gaussian support based on c(0) intensity ranking, with minimum 10 Gaussians per component
    if args.select_gaussians:
        np_shared = 200;
        hx = 2*math.pi/float(level);
        data_c0 = fio.readNetCDF(os.path.join(init_path, "support_data.nc"));
        data_c0 = np.swapaxes(data_c0,0,2);
        dims = data_c0.shape;
        dtype = [('x', np.int), ('y', np.int), ('z', np.int), ('value', np.float)]
        feasible = np.zeros_like(data_c0);
        dc0_centers = np.zeros((int(dims[0]/2), int(dims[1]/2), int(dims[2]/2)), dtype=dtype);
        print("Compute Gaussian support based on c(0) intensity ranking.")
        print(" max{c(0)}: %1.2e, min{c(0)}: %1.2e" % (np.amax(data_c0.flatten()), np.amin(data_c0.flatten())) );

        d0_centers_comps = {}
        np_comp = {}
        CENTERS = []
        for i in range(ncomps_data):
            d0_centers_comps[i] = np.zeros((int(dims[0]/2), int(dims[1]/2), int(dims[2]/2)), dtype=dtype);
            np_comp[i] = int(relmass[i] * np_shared);

        for x in range(0,dims[0],2):
            for y in range(0,dims[1],2):
                for z in range(0,dims[2],2):
                    dc0_centers[int(x/2),int(y/2),int(z/2)]['x'] = x;
                    dc0_centers[int(x/2),int(y/2),int(z/2)]['y'] = y;
                    dc0_centers[int(x/2),int(y/2),int(z/2)]['z'] = z;
                    dc0_centers[int(x/2),int(y/2),int(z/2)]['value'] = data_c0[x,y,z];
                    feasible[x,y,z] = 1;
                    for i in range(ncomps_data):
                        if labeled[x,y,z] == i+1:
                            d0_centers_comps[i][int(x/2),int(y/2),int(z/2)]['x'] = x;
                            d0_centers_comps[i][int(x/2),int(y/2),int(z/2)]['y'] = y;
                            d0_centers_comps[i][int(x/2),int(y/2),int(z/2)]['z'] = z;
                            d0_centers_comps[i][int(x/2),int(y/2),int(z/2)]['value'] = data_c0[x,y,z];
        heap     = {}
        failures = []
        for i in range(ncomps_data):
            heap[i] = [];
            centers = [];
            for x in range(d0_centers_comps[i].shape[0]):
                for y in range(d0_centers_comps[i].shape[1]):
                    for z in range(d0_centers_comps[i].shape[2]):
                        heapq.heappush(heap[i], (- d0_centers_comps[i][x][y][z]['value'], d0_centers_comps[i][x][y][z]['x'], d0_centers_comps[i][x][y][z]['y'], d0_centers_comps[i][x][y][z]['z'], i))
            print("\nselecting Gaussians for component #%d: selecting 10 + %d Gaussians according to c(0) ranking." % (i, np_comp[i]));
            for k in range(10+np_comp[i]):
                if len(heap[i]) == 0:
                    print(bcolors.WARNING + "Warning: Heap empty. Selected %d of desired %d Gaussians in component %d" % (k, 10+np_comp[i], i) + bcolors.ENDC);
                    break;
                elem = heapq.heappop(heap[i]);
                if abs(elem[0]) == 0:
                    print(bcolors.WARNING + "Warning: Heap element zero. Selected %d of desired %d Gaussians in component %d" % (k, 10+np_comp[i], i) + bcolors.ENDC);
                    break;
                if feasible[elem[1], elem[2], elem[3]]:
                    centers.append(tuple([elem[1], elem[2], elem[3], elem[4]]))
                    feasible[elem[1], elem[2], elem[3]] = 0;
                else:
                    print(bcolors.FAIL + "Error: attempting to select a Gaussian with center already selected." + bcolors.ENDC);
                    failures.append(tuple([elem[1], elem[2], elem[3]]));
            print("selected centers of component #%d:\n" % i, centers);
            CENTERS.extend(centers);
        if (len(failures) > 0):
            ax.scatter(*zip(*failures), c='magenta', marker='s');
            print(bcolors.WARNING + "Processing level %d done. Gaussian selection finished with errors." % level + bcolors.ENDC);
        else:
            print(bcolors.OKGREEN + "Processing level %d done. No errors occured." % level + bcolors.ENDC);

        # write gaussian centers to file phi-selection.txt
        phi_cm_selection_file = open(os.path.join(init_path,'phi-support-c0.txt'),'w');
        phi_cm_selection_file.write(" sigma = %1.8f, spacing = %1.8f\n" % (args.sigma * hx, 2*hx));
        phi_cm_selection_file.write(" centers = [\n")
        for cm in CENTERS:
            phi_cm_selection_file.write(" %1.8f, %1.8f, %1.8f, %d\n" % (cm[2]*hx, cm[1]*hx, cm[0]*hx, cm[3]+1));
        phi_cm_selection_file.write("];")

###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='process input images')
    parser.add_argument ('-input_path', type=str, help = 'input folder');
    parser.add_argument ('-output_path', type=str, help = 'output folder');

    # connected components`
    parser.add_argument ('-concomp_data', action='store_true', help = 'computes connected components of input data, along with center of mass and relative mass of component');
    parser.add_argument ('-select_gaussians', action='store_true', help = 'selects maximal (200 + #nc * 10) Gaussians according to ranking in c(0) of previous level, and relative mass of components.');
    parser.add_argument ('-obs_th', type = float, default = 0.0,   help = 'relative observation threshold for target data  (as used in connected component analysis)');
    parser.add_argument ('-sigma', type = float, default = 1,   help = 'sigma = factor * hx on level');
    parser.add_argument ('-labels', type=str, help = 'patient labels');

    # resample
    parser.add_argument ('-resample_input', action='store_true', help = 'resamples input patient segmentation to different level');
    parser.add_argument ('-resample', action='store_true', help = 'resamples input image to different levels');
    parser.add_argument ('-fname', type = str, help = 'path to patient segmentation file (template)');
    parser.add_argument ('-ndim', type=int, help = 'new resolution');

    # observation mask
    parser.add_argument ('-compute_observation_mask', action='store_true', help = 'computes observation mask OBS = TC + lambda_obs (1-WT)');
    parser.add_argument ('-obs_lambda', type = float, default = 1,   help = 'parameter to control observation operator OBS = TC + lambda (1-WT)');
    parser.add_argument ('-suffix', type=str, help = 'ob name suffix');

    # convert
    parser.add_argument ('-convert',action='store_true', help = 'convert netcdf images to nifti images');
    parser.add_argument ('-reference_image', type=str, help = 'reference nifti image')
    parser.add_argument ('-multilevel', action='store_true', default=False, help = 'If set, convert images on all levels.');

    # read log and extract
    parser.add_argument ('-update_config', action='store_true', help = 'extracts rho and k from logfile of previous level, and updates config');

    args = parser.parse_args();

    labels = {}
    labels_rev = {}
    if args.labels:
        for x in args.labels.split(','):
            labels[int(x.split('=')[0])] = x.split('=')[1];
        labels_rev = {v:k for k,v in labels.items()};
    else:
        labels = None
        labels_rev = None

    if args.resample_input:
        print("[] resampling input image {} to resolution {}^3".format(args.fname, args.ndim))
        resample_input(args.input_path, args.fname, args.ndim);

    if args.resample:
        print("[] resampling image {} to resolution {}^3".format(args.fname, args.ndim))
        resample(args.input_path, args.fname, args.ndim);

    if args.concomp_data:
        compute_connected_components(args, labels_rev, obs_th=args.obs_th)

    if args.compute_observation_mask:
        print("[] computing observation mask OBS = TC + lambda*(1-WT) using lambda=", args.obs_lambda)
        compute_observation_mask(args.input_path, args.fname, args.obs_lambda, labels_rev, args.ndim, args.suffix, args.ref_image)

    if args.update_config:
        print("[] extracting rho and k from ", args.input_path);
        update_config(args.input_path, args.output_path);

    if args.convert:
        convert_images_to_orig_size(args.input_path, args.reference_image, gridcont=args.multilevel)
