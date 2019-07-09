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
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
# from matplotlib.colors import ListedColormap
from scipy.spatial import distance
# from tabulate import tabulate
import sqlite3
import math
import time
import scipy
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import nibabel as nib
import nibabel.processing
import file_io as fio

# cl1 = '#9199BE'
# cl2 = '#54678F'
# cl3 = '#2E303E'
cl3 = '#0099E5'
cl1 = '#FF4C4C'
cl2 = '#34BF49'

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
        cmvec = 'phi-mesh-scaled.txt'

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
def weightedCenterPiForDataComponents(pvec, phi, hx, data_components, n_comps):
    """
    @brief: computes the weigthedcenter of mass for p_i's that fall into one
            connected component of the data
    """
    IDs = {}
    phi_labeled_dcomp = {}
    pi_labeled_dcomp  = {}
    wcm_labeled_dcomp = {}
    for i in range(n_comps+1):
        phi_labeled_dcomp[i] = [];
        pi_labeled_dcomp[i]  = [];
    s = 1./(hx * 2*math.pi); # convert to slice numbers
    for px, pi, i in zip(phi, pvec, range(len(pvec))):
        IDs[str(px)] = i;
        x = int(round(px[2]*s)); # flip
        y = int(round(px[1]*s));
        z = int(round(px[0]*s));
        phi_labeled_dcomp[data_components[x,y,z]].append(px);
        pi_labeled_dcomp[data_components[x,y,z]].append(pi);

    # for i in range(n_comps):
        # print("phi centers of component #%d: \n" % i, phi_labeled_dcomp[i]);
        # print("pi's of component #%d: \n" % i, phi_labeled_dcomp[i]);
    if len(phi_labeled_dcomp[0]) > 0 :
        print(bcolors.FAIL + "some phi centers do not fall in any component of the data:\n", phi_labeled_dcomp, "\n", pi_labeled_dcomp, bcolors.ENDC);

    # compute weighted center fo mass
    for i in range(n_comps+1):
        x = np.array([0,0,0]);
        ptot = 0;
        for px in phi_labeled_dcomp[i]:
            id = IDs[str(px)];
            x = np.add(x, pvec[id] * np.array([px[2], px[1], px[0]])); # flip
            ptot += pvec[id];
        wcm_labeled_dcomp[i] = x/float(ptot) if ptot > 0 else np.array([0,0,0]);

    return phi_labeled_dcomp, pi_labeled_dcomp, wcm_labeled_dcomp; # wcm_labeled has correct x, y, z order (same as xcm DATA)


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
def thresh(slice, cmap, thresh=0.3, v_max=None, v_min=None):
    # clip slice to interval [0,1], generate alpha values: any value > thres will have zero transparency
    slice_clipped = np.clip(slice, 0, 1)
    alphas = Normalize(0, thresh, clip=True)(slice_clipped)
    # alphas = np.clip(alphas, .4, 1)  # alpha value clipped at the bottom at .4
    max = np.amax(slice_clipped) if v_max == None else v_max;
    min = np.amin(slice_clipped) if v_min == None else v_min;
    slice_normalized = Normalize(min, max)(slice_clipped);
    cmap_ = cmap(slice_normalized)
    cmap_[..., -1] = alphas
    return cmap_;

###
### ------------------------------------------------------------------------ ###
def cont(slice, cmap, thresh=0.3, v_max=None, v_min=None):
    slice_clipped = np.clip(slice, 0, 1)
    max = np.amax(slice_clipped) if v_max == None else v_max;
    min = np.amin(slice_clipped) if v_min == None else v_min;
    norm = mpl.cm.colors.Normalize(min, max);
    slice_normalized = Normalize(min, max)(slice_clipped);
    return slice_normalized, norm


###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    pd.options.display.float_format = '{1.2e}%'.format
    parser = argparse.ArgumentParser(description='process BRATS results')
    parser.add_argument ('-x',  '--dir', type = str, help = 'path to the results folder');
    parser.add_argument ('-data', action='store_true', help = 'compute connected components for target data');
    parser.add_argument ('-sol',    action='store_true', help = 'compute connected components for solution vector');
    parser.add_argument ('--obs_lambda',     type = float, default = 1,   help = 'parameter to control observation operator OBS = TC + lambda (1-WT)');
    parser.add_argument ('--level',          type = int,   default = None,   help = 'level to evaluate for');
    args = parser.parse_args();

    VIS       = True;
    BIN_COMPS = True;

    if args.obs_lambda == None:
        args,obs_lambda = 1;
    levels = []
    if args.level == None:
        levels = [64, 128, 256]
    else:
        levels = [args.level]

    markers = ['o', 'o', 'o']
    # colors  = ['red', 'green', 'blue']
    colors  = [cl1,cl2,cl3]
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
    phi_labeled_dcomp  = {}
    pi_labeled_dcomp   = {}
    wcm_labeled_dcomp  = {}
    dist_wcmSOL_cmDATA = {}

    Xx = []
    Yy = []
    Zz = []
    # open output file
    concomp_file = open(os.path.join(args.dir,'components.txt'),'w');
    # get bratsID
    for x in args.dir.split('/'):
        bratsID = x if x.startswith('Brats') else '';
    bratsID = bratsID.split("_1")[0] + '_1'

    if VIS:
        # fig_l = plt.figure(figsize=(8,8));
        # ax_l   = fig_l.add_subplot(111, projection='3d', adjustable='box');
        fig_l2d, ax_l2d = plt.subplots(1,3, figsize=(12,4));

    # loop over all levels
    for l, m, cc in zip(levels, markers, colors):
        # define paths
        lvl_prefix = os.path.join(os.path.join(args.dir, 'tumor_inversion'), 'nx' + str(l));
        res_path   = os.path.join(lvl_prefix, "obs-{0:1.1f}".format(args.obs_lambda));
        init_path  = os.path.join(lvl_prefix, 'init');
        vis_path   = os.path.join(args.dir, 'vis');
        out_path   = os.path.join(args.dir,   'input');
        print("\n ### LEVEL %d ### \n" % l);
        if args.data:
            level = int(res_path.split("nx")[-1].split("/")[0].split("-")[0]);
            if level != l:
                print(color.FAIL + "Error %d != %d " + color.ENDC % (level, l));
            print("\n (1) computing connected components of data\n");

            # compute connected components
            labeled[l], comps_data[l], ncomps_data[l], xcm_data_px[l], xcm_data[l], relmass[l] = connectedComponentsData(init_path, level);

            # write labels
            fio.createNetCDF(os.path.join(out_path, 'data_comps_nx'+str(level)+'.nc') , np.shape(labeled[l]), np.swapaxes(labeled[l],0,2));

        if args.sol:
            print("\n (2) computing connected components of solution\n");
            # pvec, phi, phix, phiy, phiz, sigma, hx, nx = readDataSolution(args.dir);
            pvec[l], phi[l], sigma[l], hx[l], n[l] = readDataSolution(res_path);
            # distAlltoAll(phi, sigma);
            ncomps_sol[l], comps_sol[l], p_comp[l], xcm_sol[l], xcm_tot_sol[l] = connectedComponentsP(pvec[l], phi[l], sigma[l], hx[l]);
            # cluster phi centers and p_i's of solution according to comp(DATA) and compute weighted center of mass
            phi_labeled_dcomp[l], pi_labeled_dcomp[l], wcm_labeled_dcomp[l] =  weightedCenterPiForDataComponents(pvec[l], phi[l], hx[l], labeled[l], ncomps_data[l]);

            dist_wcmSOL_cmDATA[l] = {};
            for i in range(ncomps_data[l]):
                if np.array_equal(np.array([0,0,0]), wcm_labeled_dcomp[l][i+1]):
                    dist_wcmSOL_cmDATA[l][i] = float('nan');
                else:
                    dist_wcmSOL_cmDATA[l][i] = dist(xcm_data[l][i], wcm_labeled_dcomp[l][i+1]); # euclidean dist; wc_pi_dcomp has comp #0 = bg, so adding 1

        if VIS:
            fsize = '4';
            path_256 = os.path.join(os.path.join(args.dir, 'tumor_inversion'), 'nx256');
            path_256 = os.path.join(path_256, "obs-{0:1.1f}".format(args.obs_lambda));

            template_fname = "template_nx"+str(l)+".nii.gz" if l < 256 else bratsID + '_seg_tu.nii.gz'
            template_img = nib.load(os.path.join(path_256, template_fname));
            template     = template_img.get_fdata();
            c0recon_img  = nib.load(os.path.join(res_path, "c0Recon.nii.gz"));
            c0recon      = c0recon_img.get_fdata();
            c1recon_img  = nib.load(os.path.join(res_path, "cRecon.nii.gz"));
            c1recon      = c1recon_img.get_fdata();
            data_img     = nib.load(os.path.join(res_path, "data.nii.gz"));
            data         = data_img.get_fdata();

            hi_c0 = np.amax(np.clip(c0recon, 0, 1));
            lo_c0 = np.amin(np.clip(c0recon, 0, 1));

            hi_c1 = np.amax(np.clip(c1recon, 0, 1));
            lo_c1 = np.amin(np.clip(c1recon, 0, 1));

            data_masked = np.ma.masked_where(data <= 1E-1, data)
            c0_masked   = np.ma.masked_where(c0recon <= 1E-5, c0recon)
            c1_masked   = np.ma.masked_where(c1recon < 1E-2, c1recon)
            c1_masked_on_c0 = np.ma.masked_where(c0recon > 1E-1, c1_masked)

            # find dimensions of chart
            max_nc = 0;
            P_perZ = {};
            X_perZ = {};
            Z_perC = {};
            if BIN_COMPS:
                for k in range(ncomps_sol[l]):
                    P_perZ[k] = {};
                    X_perZ[k] = {};
                    Z         = [];
                    for r in range(len(p_comp[l][k])):
                        if p_comp[l][k][r] < 1e-8:
                            continue;
                        z = int(round(comps_sol[l][k][r][0]/(2*math.pi)*template.shape[2]));
                        if z not in P_perZ[k]:
                            P_perZ[k][z] = [p_comp[l][k][r]];
                            X_perZ[k][z] = [tuple([comps_sol[l][k][r][2],comps_sol[l][k][r][1],comps_sol[l][k][r][0]])];
                        else:
                            P_perZ[k][z].append(p_comp[l][k][r])
                            X_perZ[k][z].append(tuple([comps_sol[l][k][r][2],comps_sol[l][k][r][1],comps_sol[l][k][r][0]]));
                        if z not in Z:
                            Z.append(z);
                    Z_perC[k] = Z
                    max_nc = len(Z_perC[k]) if len(Z_perC[k]) > max_nc else max_nc;
            else:
                max_nc = max(len(comps_sol[l][r]) for r in range(ncomps_sol[l]));

            for vis_c1 in [True, False]:
                hi = None
                lo = None
                ac1 = 0.5;
                ac0 = 0.5 if vis_c1 else 0.5;
                d_thresh  = 0.3;  # values d(x)  > thres have zero transparency
                c0_thresh = 0.3;  # values c0(x) > thres have zero transparency
                c1_thresh = 0.4;  # values c1(x) > thres have zero transparency
                tpos = 0.92;
                cmap_c0 = plt.cm.Reds;
                cmap_c1 = plt.cm.cool;
                cmap_d  = plt.cm.winter;
                lls = [0.01, 0.1, 0.5, 0.7, 0.9];
                cmap_cont = plt.cm.rainbow


                ###
                ### CHART A #################################
                fig, axis = plt.subplots(2+ncomps_sol[l], max( 1+ ncomps_sol[l], 1 + ncomps_data[l], max_nc));
                for ax in axis.flatten():  # remove ticks and labels
                    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False);

                # weigted center of mass of components(SOL)
                ax = axis[0];
                z = int(xcm_sol[l][0][0]/(2*math.pi)*template.shape[2])
                ax[0].imshow(template[:,:,z].T, cmap='gray', interpolation='none');
                ax[0].imshow(thresh(data[:,:,z].T, cmap_d, thresh=d_thresh), interpolation='none', alpha=0.6);
                ax[0].set_ylabel("$x_{cm}$ of components(SOL)", fontsize=fsize)
                ax[0].set_title("axial slice %d" % z, size=fsize, y=tpos)
                for k in range(len(xcm_sol[l])):
                    z = int(round(xcm_sol[l][k][0]/(2*math.pi)*template.shape[2]))
                    y = int(round(xcm_sol[l][k][1]/(2*math.pi)*template.shape[1]))
                    x = int(round(xcm_sol[l][k][2]/(2*math.pi)*template.shape[0]))
                    ax[k+1].imshow(template[:,:,z].T, cmap='gray', interpolation='none');
                    if vis_c1:
                        ax[k+1].imshow(thresh(c1recon[:,:,z].T, cmap_c1, thresh=c1_thresh, v_max=hi_c1, v_min=lo_c1), interpolation='none', alpha=1);
                    # ax[k+1].imshow(thresh(c0recon[:,:,z].T, cmap_c0, thresh=c0_thresh, v_max=hi, v_min=lo), interpolation='none', alpha=1 );
                    slice, norm = cont(c0recon[:,:,z].T, cmap_c0, v_max=hi_c0, v_min=lo_c0);
                    ax[k+1].contour(slice,  levels=lls,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(lls)-1), linestyles=['-'] ,linewidths=[0.2], norm=norm);
                    # ax[r].contourf(slice, levels=lls,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(lls)-1), norm=norm);
                    ax[k+1].set_title("x=(%1.1f, %1.1f, %1.1f)\naxial slice %d" % (xcm_sol[l][k][2]/(2*math.pi)*template.shape[0], xcm_sol[l][k][1]/(2*math.pi)*template.shape[1], xcm_sol[l][k][0]/(2*math.pi)*template.shape[2], z), size=fsize, y=tpos)
                for k in range(len(xcm_sol[l]), axis.shape[1]):
                    ax[k].axis('off');

                # center of mass of components(DATA)
                ax = axis[1];
                z = int(round(xcm_data[l][0][2]/(2*math.pi)*template.shape[2]))
                ax[0].imshow(template[:,:,z].T, cmap='gray', interpolation='none');
                ax[0].imshow(thresh(data[:,:,z].T, cmap_d, d_thresh, v_max=hi, v_min=lo), interpolation='none', alpha=0.6);
                ax[0].set_ylabel("$x_{cm}$ of components(DATA)", fontsize=fsize)
                ax[0].set_title("axial slice %d\n" % z, size=fsize, y=tpos)
                for k in range(len(xcm_data[l])):
                    z = int(round(xcm_data[l][k][2]/(2*math.pi)*template.shape[2]))
                    y = int(round(xcm_data[l][k][1]/(2*math.pi)*template.shape[1]))
                    x = int(round(xcm_data[l][k][0]/(2*math.pi)*template.shape[0]))
                    ax[k+1].imshow(template[:,:,z].T, cmap='gray', interpolation='none');
                    ax[k+1].imshow(thresh(data[:,:,z].T, cmap_d, d_thresh, v_max=hi, v_min=lo), interpolation='none', alpha=0.6);
                    if vis_c1:
                        ax[k+1].imshow(thresh(c1recon[:,:,z].T, cmap_c1, thresh=c1_thresh, v_max=hi_c1, v_min=lo_c1), interpolation='none', alpha=1);
                    # ax[k+1].imshow(thresh(c0recon[:,:,z].T, cmap_c0, thresh=c0_thresh, v_max=hi, v_min=lo), interpolation='none', alpha=1);
                    slice, norm = cont(c0recon[:,:,z].T, cmap_c0, v_max=hi_c0, v_min=lo_c0);
                    ax[k+1].contour(slice,  levels=lls,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(lls)-1), linestyles=['-'] ,linewidths=[0.2], norm=norm);
                    # ax[r].contourf(slice, levels=lls,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(lls)-1), norm=norm);
                    # ax[k+1].scatter(x, y, marker='x', c='blue', s=3)
                    ax[k+1].set_title("x=(%1.1f, %1.1f, %1.1f)\naxial slice %d" % (xcm_data[l][k][0]/(2*math.pi)*template.shape[0], xcm_data[l][k][1]/(2*math.pi)*template.shape[1], xcm_data[l][k][2]/(2*math.pi)*template.shape[2], z), size=fsize, y=tpos)
                for w in range(len(xcm_data[l]), axis.shape[1]):
                    ax[w].axis('off');

                # locations of p activations in SOL
                for k in range(ncomps_sol[l]):
                    ax = axis[2+k]
                    if BIN_COMPS:
                        for z, r in zip(Z_perC[k], range(len(Z_perC[k]))):
                            ax[r].imshow(template[:,:,z].T,  cmap='gray', interpolation='none');
                            ax[0].set_ylabel("$x_i$ of component(SOL) #" + str(k), fontsize=fsize)
                            if vis_c1:
                                ax[r].imshow(thresh(c1recon[:,:,z].T, cmap_c1, thresh=c1_thresh, v_max=hi_c1, v_min=lo_c1), interpolation='none', alpha=1);
                            # ax[r].imshow(thresh(c0recon[:,:,z].T, cmap_c0, thresh=c0_thresh, v_max=hi, v_min=lo), interpolation='none', alpha=1);
                            slice, norm = cont(c0recon[:,:,z].T, cmap_c0, v_max=hi_c0, v_min=lo_c0);
                            ax[r].contour(slice,  levels=lls,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(lls)-1), linestyles=['-'] ,linewidths=[0.2], norm=norm);
                            # ax[r].contourf(slice, levels=lls,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(lls)-1), norm=norm);
                            title = ''
                            for p in P_perZ[k][z]:
                                title += "p_i=" + "{0:1.2e}".format(p) + "\n"
                            index_max = np.argmax(np.array(P_perZ[k][z]));
                            ax[r].set_title(title + "x=(%1.1f, %1.1f, %1.1f)\naxial slice %d" % (X_perZ[k][z][index_max][0]/(2*math.pi)*template.shape[0], X_perZ[k][z][index_max][1]/(2*math.pi)*template.shape[1], X_perZ[k][z][index_max][2]/(2*math.pi)*template.shape[2],z), size=fsize, y=tpos);
                        for w in range(len(Z_perC[k]), axis.shape[1]):
                            ax[w].axis('off');

                    else:
                        for r in range(len(p_comp[l][k])):
                            if p_comp[l][k][r] < 1e-8:
                                continue;
                            z = int(comps_sol[l][k][r][0]/(2*math.pi)*template.shape[2])
                            ax[r].imshow(template[:,:,z].T,  cmap='gray', interpolation='none');
                            if vis_c1:
                                ax[r].imshow(thresh(c1recon[:,:,z].T, cmap_c1, thresh=c1_thresh, v_max=hi, v_min=lo), interpolation='none', alpha=1);
                            # ax[r].imshow(thresh(c0recon[:,:,z].T, cmap_c0, thresh=c0_thresh, v_max=hi, v_min=lo), interpolation='none', alpha=1);
                            slice, norm = cont(c0recon[:,:,z].T, cmap_c0, v_max=hi_c0, v_min=lo_c0);
                            ax[r].contour(slice,  levels=lls,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(lls)-1), linestyles=['-'] ,linewidths=[0.2], norm=norm);
                            # ax[r].contourf(slice, levels=lls,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(lls)-1), norm=norm);
                            ax[r].set_title("$p_i$=%1.2e\nx=(%1.1f, %1.1f, %1.1f)\naxial slice %d" % (p_comp[l][k][r],comps_sol[l][k][r][2]/(2*math.pi)*template.shape[0], comps_sol[l][k][r][1]/(2*math.pi)*template.shape[1], comps_sol[l][k][r][0]/(2*math.pi)*template.shape[2],z), size=fsize, y=tpos);


                ###
                ### CHART B #################################
                fig2, axis2 = plt.subplots(3, ncomps_data[l]);
                for ax in axis2.flatten(): # remove ticks and labels
                    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False);

                # center of mass of components(DATA)
                idx = tuple([0,0]) if isinstance(axis2[0], (np.ndarray, np.generic)) else 0
                axis2[idx].set_ylabel("DATA", fontsize=fsize)
                idx = tuple([1,0]) if isinstance(axis2[0], (np.ndarray, np.generic)) else 1
                axis2[idx].set_ylabel("$x_{cm}$ of components(DATA)", fontsize=fsize)
                idx = tuple([2,0]) if isinstance(axis2[0], (np.ndarray, np.generic)) else 2
                axis2[idx].set_ylabel("weighted $x_{cm}$ of SOL", fontsize=fsize)
                for k in range(len(xcm_data[l])):
                    z = int(round(xcm_data[l][k][2]/(2*math.pi)*template.shape[2]))
                    y = int(round(xcm_data[l][k][1]/(2*math.pi)*template.shape[1]))
                    x = int(round(xcm_data[l][k][0]/(2*math.pi)*template.shape[0]))

                    # row 1, data
                    idx = tuple([0,k]) if isinstance(axis2[0], (np.ndarray, np.generic)) else 0
                    axis2[idx].imshow(template[:,:,z].T, cmap='gray', interpolation='none');
                    axis2[idx].imshow(thresh(data[:,:,z].T, cmap_d, thresh=d_thresh, v_max=hi, v_min=lo), interpolation='none', alpha=0.6);
                    axis2[idx].set_title("axial slice %d\n" % z, size=fsize, y=tpos)

                    # row 2, center of mass of DATA components
                    idx = tuple([1,k]) if isinstance(axis2[0], (np.ndarray, np.generic)) else 1
                    axis2[idx].imshow(template[:,:,z].T, cmap='gray', interpolation='none');
                    axis2[idx].imshow(thresh(data[:,:,z].T, cmap_d, d_thresh, v_max=hi, v_min=lo), interpolation='none', alpha=0.6);
                    if vis_c1:
                        axis2[idx].imshow(thresh(c1recon[:,:,z].T, cmap_c1, thresh=c1_thresh, v_max=hi_c1, v_min=lo_c1), interpolation='none', alpha=1);
                    slice, norm = cont(c0recon[:,:,z].T, cmap_c0, v_max=hi_c0, v_min=lo_c0);
                    axis2[idx].contour(slice,  levels=lls,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(lls)-1), linestyles=['-'] ,linewidths=[0.2], norm=norm);
                    axis2[idx].set_title("x=(%1.1f, %1.1f, %1.1f)\naxial slice %d\n" % (xcm_data[l][k][0]/(2*math.pi)*template.shape[0], xcm_data[l][k][1]/(2*math.pi)*template.shape[1], xcm_data[l][k][2]/(2*math.pi)*template.shape[2], z), size=fsize, y=tpos)

                    # row 3, weighted center of mass of solution, labeled by data components
                    z = int(round(wcm_labeled_dcomp[l][k+1][2]/(2*math.pi)*template.shape[2]))
                    y = int(round(wcm_labeled_dcomp[l][k+1][1]/(2*math.pi)*template.shape[1]))
                    x = int(round(wcm_labeled_dcomp[l][k+1][0]/(2*math.pi)*template.shape[0]))
                    idx = tuple([2,k]) if isinstance(axis2[0], (np.ndarray, np.generic)) else 2
                    axis2[idx].imshow(template[:,:,z].T, cmap='gray', interpolation='none');
                    axis2[idx].imshow(thresh(data[:,:,z].T, cmap_d, d_thresh, v_max=hi, v_min=lo), interpolation='none', alpha=0.6);
                    if vis_c1:
                        axis2[idx].imshow(thresh(c1recon[:,:,z].T, cmap_c1, thresh=c1_thresh, v_max=hi_c1, v_min=lo_c1), interpolation='none', alpha=1);
                    slice, norm = cont(c0recon[:,:,z].T, cmap_c0, v_max=hi_c0, v_min=lo_c0);
                    axis2[idx].contour(slice,  levels=lls,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(lls)-1), linestyles=['-'] ,linewidths=[0.2], norm=norm);
                    title = ''
                    for p in pi_labeled_dcomp[l][k+1]:
                        title += "p_i=" + "{0:1.2e}".format(p) + "\n"
                    axis2[idx].text(1.02, 0.5, title, fontsize=fsize, transform=axis2[idx].transAxes)
                    axis2[idx].set_title("x=(%1.1f, %1.1f, %1.1f)\naxial slice %d\n" % (wcm_labeled_dcomp[l][k+1][0]/(2*math.pi)*template.shape[0], wcm_labeled_dcomp[l][k+1][1]/(2*math.pi)*template.shape[1], wcm_labeled_dcomp[l][k+1][2]/(2*math.pi)*template.shape[2], z), size=fsize, y=tpos)


                # ###
                # ### CHART C #################################
                # fig3, axis3 = plt.subplots(2, ncomps_data[l]);
                # for ax in axis3.flatten(): # remove ticks and labels
                #     ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False);
                #
                # # center of mass of components(DATA)
                # idx = tuple([0,0]) if isinstance(axis2[0], (np.ndarray, np.generic)) else 0
                # axis3[idx].set_ylabel("DATA", fontsize=fsize)
                # idx = tuple([1,0]) if isinstance(axis2[0], (np.ndarray, np.generic)) else 1
                # axis3[idx].set_ylabel("$x_{cm}$ of components(DATA)", fontsize=fsize)
                # for k in range(len(xcm_data[l])):
                #     z = int(round(xcm_data[l][k][2]/(2*math.pi)*template.shape[2]))
                #     y = int(round(xcm_data[l][k][1]/(2*math.pi)*template.shape[1]))
                #     x = int(round(xcm_data[l][k][0]/(2*math.pi)*template.shape[0]))
                #
                #     # row 1, data
                #     idx = tuple([0,k]) if isinstance(axis2[0], (np.ndarray, np.generic)) else 0
                #     axis3[idx].imshow(template[:,:,z].T, cmap='gray', interpolation='none');
                #     axis3[idx].imshow(thresh(data[:,:,z].T, cmap_d, thresh=d_thresh, v_max=hi, v_min=lo), interpolation='none', alpha=0.6);
                #     axis3[idx].set_title("axial slice %d\n" % z, size=fsize, y=tpos)
                #
                #     # row 2, center of mass of DATA components
                #     idx = tuple([1,k]) if isinstance(axis2[0], (np.ndarray, np.generic)) else 1
                #     axis3[idx].imshow(template[:,:,z].T, cmap='gray', interpolation='none');
                #     axis3[idx].imshow(thresh(data[:,:,z].T, cmap_d, d_thresh, v_max=hi, v_min=lo), interpolation='none', alpha=0.6);
                #     if vis_c1:
                #         axis3[idx].imshow(thresh(c1recon[:,:,z].T, cmap_c1, thresh=c1_thresh, v_max=hi_c1, v_min=lo_c1), interpolation='none', alpha=1);
                #     slice, norm = cont(c0recon[:,:,z].T, cmap_c0, v_max=hi_c0, v_min=lo_c0);
                #     axis3[idx].contour(slice,  levels=lls,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(lls)-1), linestyles=['-'] ,linewidths=[0.2], norm=norm);
                #     axis3[idx].set_title("x=(%1.1f, %1.1f, %1.1f)\naxial slice %d\n" % (xcm_data[l][k][0]/(2*math.pi)*template.shape[0], xcm_data[l][k][1]/(2*math.pi)*template.shape[1], xcm_data[l][k][2]/(2*math.pi)*template.shape[2], z), size=fsize, y=tpos)
                #
                #
                # # save fig to file
                # if not os.path.exists(vis_path):
                #     os.makedirs(vis_path)
                # fname = 'chart-A-c1_nx'+str(l) if vis_c1 else 'chart-A-c0_nx'+str(l);
                # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.4);
                # fig.savefig(os.path.join(vis_path, fname + '.pdf'), format='pdf', dpi=1200);
                # fname = 'chart-B-c1_nx'+str(l) if vis_c1 else 'chart-B-c0_nx'+str(l);
                # fig2.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3);
                # fig2.savefig(os.path.join(vis_path, fname + '.pdf'), format='pdf', dpi=1200);
                # fname = 'chart-C-c1_nx'+str(l) if vis_c1 else 'chart-C-c0_nx'+str(l);
                # fig2.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3);
                # fig2.savefig(os.path.join(vis_path, fname + '.pdf'), format='pdf', dpi=1200);


            # visualize evolution of solution over levels
            p_sum = pvec[l].sum();
            mag = 200;
            s = 256./(2*math.pi);
            for k in range(ncomps_sol[l]):
                xs = []
                ys = []
                zs = []
                for r in range(len(comps_sol[l][k])):
                    if p_comp[l][k][r] < 1e-8:
                        continue;
                    xs.append(comps_sol[l][k][r][2]*s) # x
                    ys.append(comps_sol[l][k][r][1]*s) # y
                    zs.append(comps_sol[l][k][r][0]*s) # z
                # ax_l.scatter(xs, ys, zs, c=cc, marker=m, s=((np.array(p_comp[l][k])/p_sum * mag)))
                ax_l2d[0].scatter(xs, ys, c=cc, marker=m, s=((np.array(p_comp[l][k])/p_sum * mag)))
                ax_l2d[1].scatter(xs, zs, c=cc, marker=m, s=((np.array(p_comp[l][k])/p_sum * mag)))
                ax_l2d[2].scatter(ys, zs, c=cc, marker=m, s=((np.array(p_comp[l][k])/p_sum * mag)))
                Xx.extend(xs);
                Yy.extend(ys);
                Zz.extend(zs);
            for k in range(ncomps_data[l]):
                label = '$(l= %d)' % (math.log(level,2)-5);
                # ax_l.scatter(xcm_data[l][k][0]*s, xcm_data[l][k][1]*s, xcm_data[l][k][2]*s, c='magenta', marker='s');
                # ax_l.text(xcm_data[l][k][0]*s-0.02, xcm_data[l][k][1]*s+0.002, xcm_data[l][k][2]*s+0.01, label, (1,1,0), size='6');
                ax_l2d[0].scatter(xcm_data[l][k][0]*s, xcm_data[l][k][1]*s, c='k', marker='x', s=12);
                ax_l2d[0].scatter(wcm_labeled_dcomp[l][k+1][0]*s, wcm_labeled_dcomp[l][k+1][1]*s, c=cc, marker='x', s=12);
                ax_l2d[1].scatter(xcm_data[l][k][0]*s, xcm_data[l][k][2]*s, c='k', marker='x', s=12);
                ax_l2d[1].scatter(wcm_labeled_dcomp[l][k+1][0]*s, wcm_labeled_dcomp[l][k+1][2]*s, c=cc, marker='x', s=12);
                ax_l2d[2].scatter(xcm_data[l][k][1]*s, xcm_data[l][k][2]*s, c='k', marker='x', s=12);
                ax_l2d[2].scatter(wcm_labeled_dcomp[l][k+1][1]*s, wcm_labeled_dcomp[l][k+1][2]*s, c=cc, marker='x', s=12);
                Xx.append(xcm_data[l][k][0]*s);
                Yy.append(xcm_data[l][k][1]*s);
                Zz.append(xcm_data[l][k][2]*s);

        # write to file
        concomp_file.write("## level %d ##\n" % l);
        ## DATA ##
        w = "";
        for i in range(ncomps_data[l]):
            w += "{0:1.2e}".format(relmass[l][i])
            if i < ncomps_data[l] - 1:
                w += ', '
        concomp_file.write("DATA:   #comp: %d, weights: [%s] \n" % (ncomps_data[l], w));
        concomp_file.write("        x_cm:  ");
        for i in range(ncomps_data[l]):
            # .nc file has flipped axes, [z, y, x].nc
            concomp_file.write("2pi(%0.3f, %0.3f, %0.3f);  " % (xcm_data[l][i][0]/(2*math.pi), xcm_data[l][i][1]/(2*math.pi), xcm_data[l][i][2]/(2*math.pi)));
        concomp_file.write("\n");
        concomp_file.write("        x_cm:  ");
        for i in range(ncomps_data[l]):
            # .nc file has flipped axes, [z, y, x].nc
            concomp_file.write("(%1.1f, %1.1f, %1.1f)px;  " % (xcm_data[l][i][0] * template.shape[0] / (2*math.pi), xcm_data[l][i][1]* template.shape[1] / (2*math.pi), xcm_data[l][i][2] * template.shape[2] / (2*math.pi) ));
        concomp_file.write("\n");

        ## SOL (LABELED) ##
        w  = "";
        w2 = "";
        for i in range(ncomps_data[l]):
            w  += "{0:1.2e}".format(dist_wcmSOL_cmDATA[l][i])
            w2 += "{0:1.1f}".format((dist_wcmSOL_cmDATA[l][i] * float(l) / (2*math.pi) ));
            if i < ncomps_data[l] - 1:
                w  += ', '
                w2 += ', '
        concomp_file.write("SOL(L): #comp: %d, distances: [%s] = [%s]px \n" % (ncomps_data[l], w, w2));
        concomp_file.write("        x_cm:  ");
        for i in range(ncomps_data[l]):
            concomp_file.write("2pi(%0.3f, %0.3f, %0.3f);  " % (wcm_labeled_dcomp[l][i+1][0]/(2*math.pi), wcm_labeled_dcomp[l][i+1][1]/(2*math.pi), wcm_labeled_dcomp[l][i+1][2]/(2*math.pi)));
        concomp_file.write("\n");
        concomp_file.write("        x_cm:  ");
        for i in range(ncomps_data[l]):
            # .nc file has flipped axes, [z, y, x].nc
            concomp_file.write("(%1.1f, %1.1f, %1.1f)px;  " % (wcm_labeled_dcomp[l][i+1][0] * template.shape[0] / (2*math.pi), wcm_labeled_dcomp[l][i+1][1]* template.shape[1] / (2*math.pi), wcm_labeled_dcomp[l][i+1][2] * template.shape[2] / (2*math.pi) ));
        concomp_file.write("\n");
        for i in range(ncomps_data[l]):
            s = ""
            for j in range(len(pi_labeled_dcomp[l][i+1])):
                s += "{p_i=" + "{0:1.2e}".format(pi_labeled_dcomp[l][i+1][j]) + ", x_i=2pi(" + "{:1.3f}".format(phi_labeled_dcomp[l][i+1][j][2]/(2*math.pi)) + ',' + "{:1.3f}".format(phi_labeled_dcomp[l][i+1][j][1]/(2*math.pi)) + ',' + "{:1.3f}".format(phi_labeled_dcomp[l][i+1][j][0]/(2*math.pi)) +')}'
                if j < len(pi_labeled_dcomp[l][i+1]) - 1:
                    s += ';  '
            concomp_file.write("        #%d: %s \n" % (i,s));

        ## SOL ##
        concomp_file.write("SOL:    #comp: %d \n" % (ncomps_sol[l]));
        concomp_file.write("        x_cm:  ");
        for i in range(ncomps_sol[l]):
            concomp_file.write("2pi(%0.3f, %0.3f, %0.3f);  " % (xcm_sol[l][i][2]/(2*math.pi), xcm_sol[l][i][1]/(2*math.pi), xcm_sol[l][i][0]/(2*math.pi)));
        concomp_file.write("\n");
        concomp_file.write("        x_cm:  ");
        for i in range(ncomps_sol[l]):
            concomp_file.write("(%1.1f, %1.1f, %1.1f)px;  " % (xcm_sol[l][i][2]* template.shape[0] / (2*math.pi), xcm_sol[l][i][1]* template.shape[1] / (2*math.pi), xcm_sol[l][i][0]* template.shape[2] / (2*math.pi)));
        concomp_file.write("\n");
        for i in range(ncomps_sol[l]):
            s = ""
            for j in range(len(p_comp[l][i])):
                if p_comp[l][i][j] < 1e-8:
                    continue;
                s += "{p_i=" + "{0:1.2e}".format(p_comp[l][i][j]) + ", x_i=2pi(" + "{:1.3f}".format(comps_sol[l][i][j][2]/(2*math.pi)) + ',' + "{:1.3f}".format(comps_sol[l][i][j][1]/(2*math.pi)) + ',' + "{:1.3f}".format(comps_sol[l][i][j][0]/(2*math.pi)) +')}'
                if j < len(p_comp[l][i]) - 1:
                    s += ';  '
            concomp_file.write("        #%d: %s \n" % (i,s));
        concomp_file.write("\n");
    concomp_file.close();


    # write cm(DATA) to phi-cm-data.tx and p-cm-data.txt file
    if l == 256:
        path_256 = os.path.join(os.path.join(args.dir, 'tumor_inversion'), 'nx256');
        path_256 = os.path.join(path_256, "obs-{0:1.1f}".format(args.obs_lambda));

        phi_cm_data_file = open(os.path.join(path_256,'phi-cm-data.txt'),'w');
        p_cm_data_file = open(os.path.join(path_256,'p-cm-data.txt'),'w');
        phi_cm_data_file.write(" sigma = %1.8f, spacing = %1.8f\n" % (2*math.pi/256, 4*math.pi/256));
        phi_cm_data_file.write(" centers = [\n")
        p_cm_data_file.write("p = [\n")
        for i in range(ncomps_data[l]):
            phi_cm_data_file.write(" %1.8f, %1.8f, %1.8f\n" % (xcm_data[l][i][2], xcm_data[l][i][1], xcm_data[l][i][0]));
            p_cm_data_file.write(" %1.8f\n" % 1.);
        phi_cm_data_file.write("];")
        p_cm_data_file.write("];")

    if VIS:
        Xx = np.asarray(Xx);
        Yy = np.asarray(Yy);
        Zz = np.asarray(Zz);
        max_range = np.amax(np.array([np.amax(Xx)-np.amin(Xx), np.amax(Yy)-np.amin(Yy), np.amax(Zz)-np.amin(Zz)]))
        mid_x = (np.amax(Xx)+np.amin(Xx)) * 0.5
        mid_y = (np.amax(Yy)+np.amin(Yy)) * 0.5
        mid_z = (np.amax(Zz)+np.amin(Zz)) * 0.5
        # ax_l.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        # ax_l.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        # ax_l.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        # ax_l.set_xlabel('X')
        # ax_l.set_ylabel('Y')
        # ax_l.set_zlabel('Z')

        ax_l2d[0].set_xlabel('X')
        ax_l2d[0].set_ylabel('Y')
        # ax_l2d[0].set_xlim([0,256])
        # ax_l2d[0].set_ylim([0,256])
        ax_l2d[0].set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax_l2d[0].set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax_l2d[0].set_ylim(ax_l2d[0].get_ylim()[::-1])        # invert the axis
        ax_l2d[0].xaxis.tick_top()                     # and move the X-Axis
        # ax.yaxis.set_ticks(np.arange(0, 16, 1)) # set y-ticks
        # ax.yaxis.tick_left()                    # remove right y-Ticks


        ax_l2d[1].set_xlabel('X')
        ax_l2d[1].set_ylabel('Z')
        # ax_l2d[1].set_xlim([0,256])
        # ax_l2d[1].set_ylim([0,256])
        ax_l2d[1].set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax_l2d[1].set_ylim(mid_z - max_range/2, mid_z + max_range/2)
        ax_l2d[2].set_xlabel('Y')
        ax_l2d[2].set_ylabel('Z')
        # ax_l2d[2].set_xlim([0,256])
        # ax_l2d[2].set_ylim([0,256])
        ax_l2d[2].set_xlim(mid_y - max_range/2, mid_y + max_range/2)
        ax_l2d[2].set_ylim(mid_z - max_range/2, mid_z + max_range/2)
        ax_l2d[0].set_aspect('equal', adjustable='box')
        ax_l2d[1].set_aspect('equal', adjustable='box')
        ax_l2d[2].set_aspect('equal', adjustable='box')

        sns.despine(offset=0, trim=True)

        fname = 'components_plot'
        # fig_l.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.4);
        # fig_l.savefig(os.path.join(vis_path, fname + '.pdf'), format='pdf', dpi=1200);
        fig_l2d.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.2, hspace=0.5);
        fig_l2d.savefig(os.path.join(vis_path, fname + '.pdf'), format='pdf', dpi=1200);
        # ax_l.set_xlim([0,256])
        # ax_l.set_ylim([0,256])
        # ax_l.set_zlim([0,256])
        # plt.show()


    # if vis:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(121, projection='3d')
    #     ax2 = fig.add_subplot(122, projection='3d')
    #     mag = 100;
    #     ptot = pvec.sum();
    #     for i, c in zip(range(len(comps_sol[l])), ['r', 'g', 'b', 'violet', 'black', 'palegreen', 'darkmagenta']):
    #         xs = []
    #         ys = []
    #         zs = []
    #         for n in comps_sol[l][i]:
    #             xs.append(n[0])
    #             ys.append(n[1])
    #             zs.append(n[2])
    #         ax.scatter(xs, ys, zs, c=c, marker='o', s=((np.array(p_comp[l][i])*mag)+1))
    #         ax.scatter(xcm_sol[l][i][0], xcm_sol[l][i][1], xcm_sol[l][i][2], c=c, marker='x')
    #         # ax.scatter(xcm_tot_sol[l][0], xcm_tot_sol[l][1], xcm_tot_sol[l][2], c='b', marker='d')
    #         ax2.scatter(xs, ys, zs, c=c, marker='o', s=((np.array(p_comp[l][i])*mag)+1))
    #         for x, y, z, p in zip(xs, ys, zs, p_comp[l][i]):
    #             # label = '(%d, %d, %d), p=%1.2f' % (x, y, z, p)
    #             label1 = 'p=%1.2e' % (p)
    #             label2 = 'p=%1.2e' % (p/ptot)
    #             ax2.text(x+0.01, y+0.01, z+0.01, label1, (1,1,0), size='6')
    #             ax2.text(x+0.01, y+0.01, z-0.01, label2, (1,1,0), size='6')
    #         ax2.scatter(xcm_sol[l][i][0], xcm_sol[l][i][1], xcm_sol[l][i][2], c=c, marker='x')
    #         ax2.scatter(xcm_tot_sol[l][0], xcm_tot_sol[l][1], xcm_tot_sol[l][2], c='b', marker='d')
    #     for i in range(ncomps_data)[l]:
    #         print("x_cm_data:",xcm_data[l][i])
    #         z = xcm_data[l][i][0]
    #         y = xcm_data[l][i][1]
    #         x = xcm_data[l][i][2]
    #         ax.scatter(x, y, z, c='magenta', marker='s')
    #         # ax.text(x+0.01, y+0.01, z+0.01, "x_cm(TC)", (1,1,0), size='6')
    #         ax2.scatter(x, y, z, c='magenta', marker='s')
    #         # ax2.text(x+0.01, y+0.01,z+0.01, "x_cm(TC)", (1,1,0), size='6')
    #
    #     ax.set_xlim([0,2*math.pi])
    #     ax.set_ylim([0,2*math.pi])
    #     ax.set_zlim([0,2*math.pi])
    #     plt.show()
