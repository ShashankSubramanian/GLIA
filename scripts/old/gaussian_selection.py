import os
import sys
from os import listdir
import numpy as np
import statistics as st
import ntpath
import argparse
import shutil
import csv
import pandas as pd
# import re
# import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
# from matplotlib.colors import ListedColormap
from scipy.spatial import distance
# from tabulate import tabulate
import sqlite3
import heapq
import math
import time
import scipy
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import nibabel as nib
import nibabel.processing
import file_io as fio


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
def dist(p, q):
    "Return the Euclidean distance between points p and q."
    return np.linalg.norm(np.asarray(p)-np.asarray(q));


###
### ------------------------------------------------------------------------ ###
def readDataSolution(path, pvec=None, cmvec=None):
    """
    @brief: reads p vector and corresponding centers of Gaussian support
    """
    if pvec == None:
        pvec = 'p-rec-scaled.txt'
    if cmvec == None:
        cmvec = 'phi-mesh-scaled.dat'

    simga = 0;
    level = int(path.split("nx")[-1].split("/")[0].split("-")[0]);
    hx    = 1/float(level);
    phix, phiy, phiz = np.loadtxt(os.path.join(path, cmvec), comments = ["]", "#"], delimiter=',', skiprows=2, unpack=True);
    # phi = np.loadtxt(os.path.join(path, cmvec), comments = ["]", "#"], delimiter=',', skiprows=2, unpack=False);
    p_vec            = np.loadtxt(os.path.join(path, pvec),  comments = ["]", "#"],                skiprows=1);
    if os.path.exists(os.path.join(path, cmvec)):
        with open(os.path.join(path, cmvec), 'r') as f:
            sigma = float(f.readlines()[0].split("sigma =")[-1].split(",")[0])
    print(" .. reading p vector and Gaussian centers of length ",p_vec.shape[0]," from level ", level, " with sigma =", sigma);

    phi = []
    for x,y,z in zip(phix,phiy,phiz):
        phi.append(tuple([x,y,z]));
    # return p_vec, phi, phix, phiy, phiz, sigma, hx, level
    return p_vec, phi, sigma, hx, level

###
### ------------------------------------------------------------------------ ###
def connectedComponentsP(pvec, phi, sigma, hx):
    """
    @brief: computes connected components in Gaussian center point cloud
    """
    print("phi set:\n", phi);
    print("p vector:\n", pvec);

    G = nx.Graph();
    H = nx.Graph();

    G.add_nodes_from(phi);
    H.add_nodes_from(phi);

    IDs = {}
    for p, i in zip(G.nodes(), range(len(G.nodes()))):
        IDs[str(p)] = i;
        for q in G.nodes():
            if p != q:
                d = dist(p,q)/float(sigma)
                H.add_edge(p,q,weight=d);
                if d <= 10:
                    G.add_edge(p,q,weight=d);

    # compute connected components
    adjMAT = nx.adjacency_matrix(G, weight='weight');
    graph = csr_matrix(adjMAT)
    # print(graph)
    ncomps_sol, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    # partition nodes into components
    comps  = {}
    p_comp = {}
    comp   = []
    lprev = labels[0];
    for i in range(ncomps_sol):
        comps[i]  = []
        p_comp[i] = []
    for n, l, p in zip(G.nodes(), labels, pvec):
        comps[l].append(n);
        p_comp[l].append(p);

    # compute weighted center fo mass
    xcm_sol = {}
    xcm_tot_sol = np.array([0,0,0]);
    for i in range(len(comps)):
        x = np.array([0,0,0]);
        ptot = 0;
        for n in comps[i]:
            id = IDs[str(n)];
            x = np.add(x, pvec[id] * np.array([n[0], n[1], n[2]]));
            ptot += pvec[id];
        xcm_sol[i] = x/float(ptot);
        xcm_tot_sol = np.add(xcm_tot_sol, x);
    xcm_tot_sol = xcm_tot_sol/float(np.sum(pvec));

    print("\nlabels:\n",labels)
    print("IDs:\n", IDs)
    print("components:\n", comps)
    print("center of mass:\n", xcm_sol)
    print(bcolors.OKBLUE, " -> number of components found: ", ncomps_sol, bcolors.ENDC)

    return ncomps_sol, comps, p_comp, xcm_sol, xcm_tot_sol


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
    xcm_data_px = {}
    xcm_data    = {}
    count       = {}
    sums        = {}
    relmass     = {}

    for img in IMG:
        total_mass = 0
        labeled, ncomponents = scipy.ndimage.measurements.label(img, structure);
        print(bcolors.OKBLUE, "   - number of components found: ", ncomponents, bcolors.ENDC)

        for i in range(ncomponents):
            comps[i] = (labeled == i+1)
            a, b = scipy.ndimage.measurements._stats(comps[i])
            total_mass += b
        print("total mass:", total_mass)

        for i in range(ncomponents):
            xcm_data_px[i] = scipy.ndimage.measurements.center_of_mass(comps[i])
            count[i], sums[i]  = scipy.ndimage.measurements._stats(comps[i])
            relmass[i] = sums[i]/float(total_mass);

            print("   [comp #%d]" %i)
            print("    - center of mass:", xcm_data_px[i])
            xcm_data[i] = tuple([2 * math.pi * xcm_data_px[i][0] / float(level), 2 * math.pi * xcm_data_px[i][1] / float(level), 2 * math.pi * xcm_data_px[i][2] / float(level)]);
            print(bcolors.OKGREEN, "                    ", xcm_data[i], bcolors.ENDC)
            print("    - sum, mass:     ", sums[i], sums[i]/float(total_mass))

    return labeled, comps, ncomponents, xcm_data_px, xcm_data, relmass;



###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    pd.options.display.float_format = '{1.2e}%'.format
    parser = argparse.ArgumentParser(description='process BRATS results')
    parser.add_argument ('-x',  '--dir', type = str, help = 'path to the results folder');
    parser.add_argument ('--obs_lambda',     type = float, default = 1,   help = 'parameter to control observation operator OBS = TC + lambda (1-WT)');
    parser.add_argument ('--level',          type = int,   default = None,   help = 'level to evaluate for');
    args = parser.parse_args();


    if args.obs_lambda == None:
        args,obs_lambda = 1;
    levels = []
    if args.level == None:
        levels = [64, 128, 256]
    else:
        levels = [args.level]


    VIS  = False;
    VIS2 = False;
    pvec  = {}
    phi   = {}
    sigma = {}
    hx    = {}
    n     = {}
    ncomps_sol  = {}
    comps_sol   = {}
    p_comp      = {}
    xcm_sol     = {}
    xcm_tot_sol = {}
    labeled     = {}
    comps_data  = {}
    ncomps_data = {}
    xcm_data_px = {}
    xcm_data    = {}
    relmass     = {}

    # get bratsID
    for x in args.dir.split('/'):
        bratsID = x if x.startswith('Brats') else '';
    bratsID = bratsID.split("_1")[0] + '_1'

    # loop over all levels
    for l in levels:
        # define paths
        lvl_prefix = os.path.join(os.path.join(args.dir, 'tumor_inversion'), 'nx' + str(l));
        res_path   = os.path.join(lvl_prefix, "obs-{0:1.1f}".format(args.obs_lambda));
        init_path  = os.path.join(lvl_prefix, 'init');
        vis_path   = os.path.join(args.dir, 'vis');
        out_path   = os.path.join(args.dir,   'input');
        print("\n ### LEVEL %d ### \n" % l);
        level = int(res_path.split("nx")[-1].split("/")[0].split("-")[0]);
        if level != l:
            print(color.FAIL + "Error %d != %d " + color.ENDC % (level, l));
        print("\n (1) computing connected components of data\n");

        # compute connected components
        labeled[l], comps_data[l], ncomps_data[l], xcm_data_px[l], xcm_data[l], relmass[l] = connectedComponentsData(init_path, level);

        # write labels
        fio.createNetCDF(os.path.join(out_path, 'data_comps_nx'+str(level)+'.nc') , np.shape(labeled[l]), np.swapaxes(labeled[l],0,2));

        # if args.sol:
        #     print("\n (2) computing connected components of solution\n");
        #     # pvec, phi, phix, phiy, phiz, sigma, hx, nx = readDataSolution(args.dir);
        #     pvec[l], phi[l], sigma[l], hx[l], n[l] = readDataSolution(res_path);
        #     # distAlltoAll(phi, sigma);
        #     ncomps_sol[l], comps_sol[l], p_comp[l], xcm_sol[l], xcm_tot_sol[l] = connectedComponentsP(pvec[l], phi[l], sigma[l], hx[l]);


        np_shared = 200;
        hx = 2*math.pi/float(l);
        data_c0 = fio.readNetCDF(os.path.join(res_path, "c0Recon.nc"));
        data_c0 = np.swapaxes(data_c0,0,2);
        dims = data_c0.shape;
        dtype = [('x', np.int), ('y', np.int), ('z', np.int), ('value', np.float)]
        feasible = np.zeros_like(data_c0);
        dc0_centers = np.zeros((int(dims[0]/2), int(dims[1]/2), int(dims[2]/2)), dtype=dtype);

        print("max{c(0)}: %1.2e" % np.amax(data_c0.flatten()))
        print("min{c(0)}: %1.2e" % np.amin(data_c0.flatten()))

        d0_centers_comps = {}
        np_comp = {}
        CENTERS = []
        LABELS  = []
        C0      = []
        for i in range(ncomps_data[l]):
            d0_centers_comps[i] = np.zeros((int(dims[0]/2), int(dims[1]/2), int(dims[2]/2)), dtype=dtype);
            np_comp[i] = int(relmass[l][i] * np_shared);

        for x in range(0,dims[0],2):
            for y in range(0,dims[1],2):
                for z in range(0,dims[2],2):
                    C0.append(tuple([data_c0[x,y,z],x,y,z]));
                    dc0_centers[int(x/2),int(y/2),int(z/2)]['x'] = x;
                    dc0_centers[int(x/2),int(y/2),int(z/2)]['y'] = y;
                    dc0_centers[int(x/2),int(y/2),int(z/2)]['z'] = z;
                    dc0_centers[int(x/2),int(y/2),int(z/2)]['value'] = data_c0[x,y,z];
                    feasible[x,y,z] = 1;
                    for i in range(ncomps_data[l]):
                        if labeled[l][x,y,z] == i+1:
                            LABELS.append(tuple([i+1,x,y,z]));
                            d0_centers_comps[i][int(x/2),int(y/2),int(z/2)]['x'] = x;
                            d0_centers_comps[i][int(x/2),int(y/2),int(z/2)]['y'] = y;
                            d0_centers_comps[i][int(x/2),int(y/2),int(z/2)]['z'] = z;
                            d0_centers_comps[i][int(x/2),int(y/2),int(z/2)]['value'] = data_c0[x,y,z];

        if VIS:
            fig = plt.figure()
            ax  = fig.add_subplot(2, 2, 1, projection='3d');
            ax2 = fig.add_subplot(2, 2, 2, projection='3d');
        heap  = {}
        failures = []
        for i, cc in zip(range(ncomps_data[l]), ['r','g','b']):
            heap[i] = [];
            centers = [];
            for x in range(d0_centers_comps[i].shape[0]):
                for y in range(d0_centers_comps[i].shape[1]):
                    for z in range(d0_centers_comps[i].shape[2]):
                        # print(d0_centers_comps[i][x][z][z]['value'])
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
                print(elem);
                if feasible[elem[1], elem[2], elem[3]]:
                    centers.append(tuple([elem[1], elem[2], elem[3], elem[4]]))
                    feasible[elem[1], elem[2], elem[3]] = 0;
                else:
                    print(bcolors.FAIL + "Error: attempting to select a Gaussian with center already selected." + bcolors.ENDC);
                    failures.append(tuple([elem[1], elem[2], elem[3]]));
            print("selected centers of component #%d:\n" % i, centers);
            CENTERS.extend(centers);
            if VIS:
                ax.scatter(*zip(*centers), c=cc)
        if (len(failures) > 0):
            ax.scatter(*zip(*failures), c='magenta', marker='s');
            print(bcolors.WARNING + "Processing level %d done. Gaussian selection finished with errors." % l + bcolors.ENDC);
        else:
            print(bcolors.OKGREEN + "Processing level %d done. No errors occured." % l + bcolors.ENDC);

        # write gaussian centers to file phi-selection.txt
        phi_cm_selection_file = open(os.path.join(init_path,'phi-support-c0.txt'),'w');
        phi_cm_selection_file.write(" sigma = %1.8f, spacing = %1.8f\n" % (hx, 2*hx));
        phi_cm_selection_file.write(" centers = [\n")
        for cm in CENTERS:
            phi_cm_selection_file.write(" %1.8f, %1.8f, %1.8f, %d\n" % (cm[2]*hx, cm[1]*hx, cm[0]*hx, cm[3]+1));
        phi_cm_selection_file.write("];")



        if VIS:
            X   = []
            Y   = []
            Z   = []
            VAL = []
            for x in LABELS:
                X.append(x[1])
                Y.append(x[2])
                Z.append(x[3])
                VAL.append(x[0])
            ax2.scatter(X,Y,Z,c=VAL);

            if VIS2:
                ax3 = fig.add_subplot(2, 2, 3, projection='3d');
                X   = []
                Y   = []
                Z   = []
                VAL = []
                for x in C0:
                    X.append(x[1])
                    Y.append(x[2])
                    Z.append(x[3])
                    VAL.append(x[0])

                alphas = Normalize(0, 0.01, clip=True)(VAL)
                # alphas = np.clip(alphas, .4, 1)  # alpha value clipped at the bottom at .4
                max = np.amax(VAL)
                min = np.amin(VAL)
                VAL_norm = Normalize(min, max)(VAL);
                cmap =plt.cm.cool
                cmap_ = cmap(VAL_norm);
                cmap_[..., -1] = alphas;
                ax3.scatter(X,Y,Z, color=cmap_, s=1);
                ax3.set_xlim(0,dims[0])
                ax3.set_xlim(0,dims[1])
                ax3.set_xlim(0,dims[2])

            # print(dims)
            ax.set_xlim(0,dims[0])
            ax.set_xlim(0,dims[1])
            ax.set_xlim(0,dims[2])
            ax2.set_xlim(0,dims[0])
            ax2.set_xlim(0,dims[1])
            ax2.set_xlim(0,dims[2])
            plt.show()
