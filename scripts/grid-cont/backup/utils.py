import os,argparse, sys
import numpy as np
import nibabel as nib
import nibabel.processing
import imageTools as imgtools
import file_io as fio
import ntpath
import math
from netCDF4 import Dataset
import heapq

import networkx as nx
import scipy
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import matplotlib as mpl
import matplotlib.pyplot as plt


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
def extractRhoK(path, rhofac):
    env_file = open(os.path.join(path,'env_rhok.sh'), 'w')
    env_file.write("#!/bin/bash\n")
    env_file.write("export RHO_INIT=8\n")
    env_file.write("export K_INIT=0\n")
    empty = False;
    exist = False;
    level = int(path.split("nx")[-1].split("/")[0]);
    logfile = 'tumor_solver_log_nx' + str(level) + '.txt'
    if os.path.exists(os.path.join(path,logfile)):
        exist =True;
        print("reading logifle:", os.path.join(path,logfile))
        with open(os.path.join(path,logfile), 'r',  encoding="utf-8") as f:
            lines  = f.readlines();
            no = 0;
            if len(lines) > 0:
                for line in lines:
                    if " ### estimated reaction coefficients:                  ###" in line:
                        rho = rhofac * float(lines[no+1].split("r1:")[-1].split(",")[0])
                    if " ### estimated diffusion coefficients:                 ### " in line:
                        k = float(lines[no+1].split("k1:")[-1].split(",")[0]);
                    no += 1;
                env_file.write("export RHO_INIT="+str(rho)+"\n")
                env_file.write("export K_INIT="+str(k)+"\n")
                os.environ['RHO_INIT'] = str(rho);
                os.environ['K_INIT']   = str(k);
                print( bcolors.OKBLUE + " ... setting init guess (rho, k) = (",rho,",",k,") " + bcolors.ENDC);
            else:
                empty = True;
    if empty or not exist:
        print("Error: tumor solver log file in ", path, "does not exist.");
    env_file.close();


###
### ------------------------------------------------------------------------ ###
def resample(path, name, nname, cdim, ndim):
    img = fio.readNetCDF(os.path.join(path,name));
    s = tuple([ndim, ndim, ndim]);
    if 'seg' in name:
        print('segmentation file found, using NN interpolation')
        img = imgtools.resizeImage(img, s, 0);
    if 'obs' in name:
        print('observation mask found, using NN interpolation')
        img = imgtools.resizeImage(img, s, 0);
    else:
        img = imgtools.resizeImage(img, s, 3);
    dim = [ndim, ndim, ndim];
    fio.createNetCDF(os.path.join(path, nname), dim, img);


###
### ------------------------------------------------------------------------ ###
def computeObservationMask(inp_dir, obs_lambda, suffix=''):
    tc = fio.readNetCDF(os.path.join(inp_dir,"patient_seg_tc.nc"));
    wt = fio.readNetCDF(os.path.join(inp_dir,"patient_seg_wt.nc"));
    tc = (tc > 0).astype(float);
    wt = (wt > 0).astype(float);
    obs = tc + obs_lambda * (1 - wt);
    dim = [256, 256, 256];
    fio.createNetCDF(os.path.join(inp_dir,"obs_mask" + suffix +".nc"), dim, obs);
    return obs;


###
### ------------------------------------------------------------------------ ###
def connectedComponentsData(path_nc, level, target_nc=None, path_nii=None, target_nii=None, nii=False):
    if target_nc == None:
        target_nc = 'patient_seg_tc.nc'

    IMG = []
    data_nc = fio.readNetCDF(os.path.join(path_nc,target_nc));
    data_nc = np.swapaxes(data_nc,0,2);
    dims_nc = data_nc.shape;
    print(".. reading targt data ", os.path.join(path_nc, target_nc), " with dimension", dims_nc)
    if nii:
        data_nii = nib.load(os.path.join(path_nii,target_nii));
        dims_nii = data_nii.shape;
        data_img = data_nii.get_fdata();
        affine = data_nii.affine;
        IMG = [data_img > 1E-1, data_nc > 1E-4];
    else:
        IMG = [data_nc > 1E-4];



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

    for img in IMG:
        total_mass = 0
        labeled, ncomponents = scipy.ndimage.measurements.label(img, structure);
        print(bcolors.OKBLUE, "   - number of components found: ", ncomponents, bcolors.ENDC)

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
    return labeled_sorted, comps_sorted, ncomponents, xcm_data_px_sorted, xcm_data_sorted, relmass_sorted;

def convert_netcdf_to_nii(input_filename, output_filename, affine=None, ref_image=None):
    ref_image = nib.load(ref_image)
    img = fio.readNetCDF(input_filename)
    img = np.swapaxes(img,0,2);
    fio.writeNII(img, output_filename, affine=affine, ref_image=ref_image);

###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='process input images')
    parser.add_argument ('-output_path',         type=str, help = 'output folder');
    parser.add_argument ('-input_path',          type=str, help = 'input folder');
    parser.add_argument ('-rdir',                type = str,          help = 'path to tumor solver results');
    parser.add_argument ('-compute_observation_mask',    action='store_true', help = 'computes observation mask OBS = TC + lambda_obs (1-WT)');
    parser.add_argument ('--obs_lambda',          type = float, default = 1,   help = 'parameter to control observation operator OBS = TC + lambda (1-WT)');
    parser.add_argument ('--suffix',              type=str, help = 'ob name suffix');
    parser.add_argument ('-resample',             action='store_true', help = 'resamples images');
    parser.add_argument ('-convert_netcdf_to_nii',action='store_true', help = 'convert netcdf images to nifti images');
    parser.add_argument ('--reference_image',      type=str, help = 'reference nifti image')
    parser.add_argument ('--N_old',               type=int, help = 'old resolution');
    parser.add_argument ('--N_new',               type=int, help = 'new resolution');
    parser.add_argument ('--name_old',            type=str, help = 'resample, old name');
    parser.add_argument ('--name_new',            type=str, help = 'resample, new name');
    parser.add_argument ('-extract',              action='store_true', help = 'extracts rho and k from info.dat file of previous level');
    parser.add_argument ('--rho_fac',             type = float, default = 1,   help = 'factor to multiply rho in between leveles');
    parser.add_argument ('-concomp_data',         action='store_true', help = 'computes connected components of input data, along with center of mass and relative mass of component');
    parser.add_argument ('-select_gaussians',     action='store_true', help = 'selects maximal (200 + #nc * 10) Gaussians according to ranking in c(0) of previous level, and relative mass of components.');
    parser.add_argument ('--sigma',               type = float, default = 1,   help = 'sigma = factor * hx on level');


    args = parser.parse_args();

    if args.rdir == None:
        args.rdir = "obs-{0:1.1f}".format(args.obs_lambda);
    elif "cm-data" in args.rdir:
        args.rdir = "cm-data-obs-{0:1.1f}".format(args.obs_lambda);
    elif "obs" in args.rdir:
        args.rdir = "obs-{0:1.1f}".format(args.obs_lambda);

    if args.compute_observation_mask:
        print("computing observation mask OBS = TC + lambda*(1-WT) using lambda=", args.obs_lambda)
        if args.suffix == None:
            computeObservationMask(args.output_path, args.obs_lambda);
        else:
            computeObservationMask(args.output_path, args.obs_lambda, args.suffix);

    if args.resample:
        print("resampling", args.name_old, "from", args.N_old, "to", args.N_new);
        resample(args.output_path, args.name_old, args.name_new, args.N_old, args.N_new);

    if args.convert_netcdf_to_nii:
        print("converting ", args.name_old, "to", args.name_new);
        convert_netcdf_to_nii(args.name_old, args.name_new, affine=None, ref_image=args.reference_image)

    if args.extract:
        print("extracting rho and k from ", args.output_path);
        extractRhoK(args.output_path, args.rho_fac);

    if args.concomp_data:
        res_path  = os.path.join(args.input_path, args.rdir);
        init_path = os.path.join(args.input_path, 'init');
        level = int(res_path.split("nx")[-1].split("/")[0]);
        print("computing connected components of data ");

        # compute connected components
        labeled, comps_data, ncomps_data, xcm_data_px, xcm_data, relmass = connectedComponentsData(init_path, level);

        # write labels
        fio.createNetCDF(os.path.join(args.output_path, 'data_comps_nx'+str(level)+'.nc') , np.shape(labeled), np.swapaxes(labeled,0,2));

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
