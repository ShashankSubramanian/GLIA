import matplotlib as mpl
mpl.use('Agg')
import os
import collections
from os import listdir
import numpy as np
import pandas as pd
import imageTools as imgtools
import nibabel as nib
import nibabel.processing
import ntpath
import argparse
import shutil
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import networkx as nx                  # graph to compute connected components
import scipy
from scipy.sparse import csr_matrix    # sparse adjacency matrix
from scipy.sparse.csgraph import connected_components # conected components algo
from scipy.spatial import distance     # dice etc
import file_io as fio
import re
from math import ceil
import sys
from colorama import Fore, Back, Style




class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


cl1 = '#FF4C4C'
cl2 = '#34BF49'
cl3 = '#0099E5'

###
### ------------------------------------------------------------------------ ###
def showSlices(slices):
    """
    @short - shows axial, coroanl, sagittal slice of image
    """
    fig,axes = plt.subplots(1,len(slices))
    for i,slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower");
    plt.show()

###
### ------------------------------------------------------------------------ ###
def computeDice(patient_img, atlas_img, patient_labels):
    """
    @short: - computes Dice coefficients between atlas and patient image for
              wm, gm, csf, based on patient labels.
    """
    # bool map atlas wm, gm, csf
    acsf = atlas_img == patient_labels['csf'];
    avt = atlas_img == patient_labels['vt'];
    agm  = atlas_img == patient_labels['gm'];
    awm  = atlas_img == patient_labels['wm'];
    # bool map patient wm, gm, csf
    pcsf = patient_img == patient_labels['csf'];
    pvt = patient_img == patient_labels['vt'];
    pgm  = patient_img == patient_labels['gm'];
    pwm  = patient_img == patient_labels['wm'];
    pen  = patient_img == patient_labels['en'];
    ped  = patient_img == patient_labels['ed'];
    pnec = patient_img == patient_labels['nec'];
    # combine edema, necrotic, enhancing in one mask
    mask = np.logical_or.reduce((ped,pnec,pen));
    mask = 1. - mask.astype(float);
    mask = mask.flatten()
    csf_dice = 1.0 - distance.dice(acsf.flatten(), pcsf.flatten(), mask);
    vt_dice = 1.0 - distance.dice(avt.flatten(), pvt.flatten(), mask);
    gm_dice  = 1.0 - distance.dice(agm.flatten(),  pgm.flatten(),  mask);
    wm_dice  = 1.0 - distance.dice(awm.flatten(),  pwm.flatten(),  mask);
    # print("healthy tissue dice: (csf_dice, vt_dice, gm_dice, wm_dice) =", csf_dice,vt_dice,gm_dice,wm_dice);
    return csf_dice,vt_dice,gm_dice,wm_dice

###
### ------------------------------------------------------------------------ ###
def rescaleImage(img):
    """
    @short: - rescales image to [0,1]
    """
    img_min = np.amin(img)
    img_max = np.amax(img)
    if np.any(img<0):
        img_min = 0
    img = (img - img_min)/(img_max-img_min)
    return img

###
### ------------------------------------------------------------------------ ###
def getLargestComponentSlice(f, imgsize):
    """
    @short: - gets the slice number for the connected component with largest weight
              by reading in the components text file
              (imgsize is a list denoting image resolution, e.g. imgsize = [256, 256, 256])
              only supports 256^3 resolution right now. TODO make it general
    """
    pix= []
    wt = []
    # skip all lines until level 256 solution is reached
    for x in f:
        if "level 256" not in x:
            continue
        else:
            break
    # skip all lines until SOLUTION is reached
    for x in f:
        if "SOL:" not in x:
            continue
        else:
            break

    max_wt = 0
    # skip all lines until the component weights and their location is reached
    for x in f:
        if "#" not in x:
            continue
        else:

            try:
                # get all the component strings (enclosed within {}) in the current line
                matches = re.finditer('\{(.*?)\}', x, re.MULTILINE | re.DOTALL)
                for matchNum,match in enumerate(matches):
                    for groupNum in range(0,len(match.group())):
                        comp_str = match.group(1)
                        # get the weight of the component from the component string
                        wt = float((comp_str.split(",")[0]).split("=")[1])
                        if wt>max_wt:
                            max_wt = wt
                            # get the normalized pixel location from component string
                            pix = re.search('\((.*?)\)', comp_str).group(1)
                            pix = [float(x) for x in pix.split(",")]
            except AttributeError:
                print("No matches found")
                sys.exit()
    # convert back pixel to image resolution range which is 256^3
    slice_num = [ceil(256*x) for x in pix]
    return slice_num

###
### ------------------------------------------------------------------------ ###
def analyzeRegistration(input_dir, atlas_path, patient_labels):
    """
    @short: -
    """

    # default image size
    imgsize = [256, 256, 256]
    # for clipping c0 to compute dice
    threshold = 0.05

    # find out which slice to print
    f = open(os.path.join(input_dir,"components_obs-1.0.txt"),"r")
    slice_num = getLargestComponentSlice(f, imgsize)
    print(">Printing ", slice_num)
    axial_slice = slice_num[-1]
    f.close()

    reg_dir = os.path.join(input_dir, "registration")
    reg_cases = next(os.walk(reg_dir))[1]
    num_cases = len(reg_cases)
    reg_case2 = os.path.join(reg_dir, reg_cases[0])

    tick_param_kwargs = {"axis":"both", "which":"both", "bottom":False, "left":False, "labelbottom":False, "labelleft":False}
    imshow_kwargs = {"cmap":"gray", "aspect":"equal"}
    imshow_kwargs_r = {"cmap":"gray_r", "aspect":"equal"}

    fig, ax = plt.subplots(1+2*num_cases, 4, figsize=(4*4,4*(1+2*num_cases)))
    for a,i in zip(ax.flatten(),np.arange(len(ax.flatten()))):
        a.tick_params(**tick_param_kwargs)
        a.set_yticklabels([])
        a.set_xticklabels([])
        a.set_frame_on(False)
        row = np.unravel_index(i, ax.shape,'C')[0]
        if np.mod(row,2) != 0:
            a.add_patch(plt.Rectangle((0,1),12, 0.01,facecolor='silver',
                              clip_on=False, linewidth = 0))


    # read in atlas t1 and atlas segmentation
    atlas_t1  = nib.load(os.path.join(atlas_path,"jakob_stripped_with_cere_lps_256x256x256.nii.gz")).get_fdata()
    atlas_seg = nib.load(os.path.join(atlas_path,"jakob_segmented_with_cere_lps_256x256x256.nii.gz")).get_fdata()
    ax[0,0].imshow(atlas_t1[:,:,axial_slice].T, **imshow_kwargs)
    ax[0,0].set_title('Atlas T1')
    print(">Printing atlas T1")

    acsf   = nib.load(os.path.join(atlas_path,"atlas_csf.nii.gz")).get_fdata()
    ave    = nib.load(os.path.join(atlas_path,"atlas_ve.nii.gz")).get_fdata()
    agm    = nib.load(os.path.join(atlas_path,"atlas_gm.nii.gz")).get_fdata()
    awm    = nib.load(os.path.join(atlas_path,"atlas_wm.nii.gz")).get_fdata()
    C      = np.linalg.norm([np.linalg.norm(acsf), np.linalg.norm(ave), np.linalg.norm(agm), np.linalg.norm(awm)])
    C_ve   = np.linalg.norm(ave)

    # read patient t1
    patient_t1 = nib.load(os.path.join(reg_case2, 'patient_t1.nii.gz')).get_fdata()
    ax[0,1].imshow(patient_t1[:,:,axial_slice].T, **imshow_kwargs)
    ax[0,1].set_title('Patient T1')
    print(">Printing patient T1")

    # read patient healthy segmentation (obtained via registration)
    patient_seg = nib.load(os.path.join(reg_case2, 'patient_seg.nii.gz')).get_fdata()
    clrs = ['black','red', 'green', 'yellow', 'blue', 'cyan', 'orange', 'purple'];
    ax[0,2].imshow(patient_seg[:,:,axial_slice].T, aspect='equal', cmap=colors.ListedColormap(clrs))
    ax[0,2].set_title('Patient labels')
    print(">Printing patient segmentation")

    # get initial condition in Patient space
    c0_dir = os.path.join(input_dir, "tumor_inversion/nx256/obs-1.0")
    c0 = rescaleImage(nib.load(os.path.join(c0_dir, "c0Recon_256x256x256_aff2jakob.nii.gz")).get_fdata())
    c0_thres = c0 > threshold
    x = slice_num[0]
    y = slice_num[1]
    d = 30
    ax[0,3].imshow(c0_thres[x-d:x+d,y-d:y+d,axial_slice].T, aspect='equal', cmap=colors.ListedColormap(['black','red']))
    ax[0,3].set_title('Patient c0')
    print(">Printing patient c0")

    # loop over registration cases to compute statistics and print relevant images
    r0_ve = 0
    r1_ve = 0
    c0_dice = np.zeros((num_cases+1, num_cases+1))
    c0_residual = np.zeros((num_cases+1, num_cases+1))
    c0_residual[0,0] = 0
    c0_dice[0,0] = 1
    healthy_dice_before = np.zeros((num_cases,4))
    healthy_dice_after = np.zeros((num_cases,4))
    res_all = np.zeros((num_cases,2))
    res_ve = np.zeros_like(res_all)
    res_ve[0,:] = 0

    nx = c0.shape
    c0_arr = np.zeros((nx[0],nx[1],nx[2],num_cases+1))
    c0_arr[:,:,:,0] = c0

    reg_cases = sorted(reg_cases, key=lambda x: x.split('_')[0])
    for case,case_num,i in zip(reg_cases,np.arange(1,1+2*num_cases,2),np.arange(1,num_cases+1)):
        r0 = 0
        r1 = 0
        print(">Processing " + case)
        ax[case_num,0].text(0.5, 0.5, case[0:8], fontsize=25, horizontalalignment='center', verticalalignment='center', transform=ax[case_num,0].transAxes)

        reg_case_dir = os.path.join(reg_dir, case)
        for filename in os.listdir(reg_case_dir):
            # get residuals
            if "t=0" in filename:
                r0 = r0 + nib.load(os.path.join(reg_case_dir,filename)).get_fdata()
            if "t=1" in filename:
                r1 = r1 + nib.load(os.path.join(reg_case_dir,filename)).get_fdata()

        vmin = np.fmin(np.min(r0[:,:,axial_slice]), np.min(r1[:,:,axial_slice]))
        vmax = np.fmax(np.max(r0[:,:,axial_slice]), np.max(r1[:,:,axial_slice]))

        print(">Printing total residual before registration")
        ax[case_num,2].imshow(r0[:,:,axial_slice].T, vmin=vmin, vmax=vmax, **imshow_kwargs_r)
        ax[case_num,2].set_ylabel('Residual-t=0')

        print(">Printing total residual after registration")
        ax[case_num+1,2].imshow(r1[:,:,axial_slice].T, vmin=vmin, vmax=vmax, **imshow_kwargs_r)
        ax[case_num+1,2].set_ylabel("Residual-t=1")

        print(">Printing jacobian")
        jac = nib.load(os.path.join(reg_case_dir,"det-deformation-grad.nii.gz")).get_fdata()
        jacmin = np.min(jac[:,:,axial_slice])
        jacmax = np.max(jac[:,:,axial_slice])
        ax[case_num,3].imshow(jac[:,:,axial_slice].T, aspect='equal', cmap='jet')
        ax[case_num,3].set_ylabel("Jacobian")

        # reset colormaps of all jacobians images
        if i==1:
            jacmin_global = jacmin
            jacmax_global = jacmax
        else:
            jacmin_global = np.fmin(jacmin, jacmin_global)
            jacmax_global = np.fmax(jacmax, jacmax_global)

        for a in ax[1:-1:2,3].flatten():
            for im in a.get_images():
                im.set_clim(vmin=jacmin_global, vmax=jacmax_global)

        print(">Printing c0 in atlas space")
        c0_arr[:,:,:,i] = rescaleImage(nib.load(os.path.join(reg_case_dir,"c0_in_Aspace.nii.gz")).get_fdata())
        for j in range(i+1):
            # residual with patient and other methods
            if j==0:
                c0_residual[j,i] = np.linalg.norm(c0_arr[:,:,:,j]-c0_arr[:,:,:,i])/np.linalg.norm(c0_arr[:,:,:,])
            else:
                c0_residual[j,i] = np.linalg.norm(c0_arr[:,:,:,j]-c0_arr[:,:,:,i])
            # compute dice with patient
            c0_dice[j,i] = 1 - distance.dice((c0_arr[:,:,:,j] > threshold).flatten(), (c0_arr[:,:,:,i] > threshold).flatten())

        # plot c0 in atlas space overlaid with c0 in patient space
        ax[case_num+1,3].imshow((c0_arr[x-d:x+d,y-d:y+d,axial_slice,i]>threshold).T, aspect='equal', cmap=colors.ListedColormap(['black','green']))
        ax[case_num+1,3].imshow((c0_arr[x-d:x+d,y-d:y+d,axial_slice,0]>threshold).T, aspect='equal', alpha=0.5, cmap=colors.ListedColormap(['black','red']))
        ax[case_num+1,3].set_ylabel("c0 in Atlas")

        # get c0_in_Aspace
        if "case-2" not in case:
            # ventricle residuals
            r0_ve = nib.load(os.path.join(reg_case_dir,"residual-t=0-000.nii.gz")).get_fdata()
            r1_ve = nib.load(os.path.join(reg_case_dir,"residual-t=1-000.nii.gz")).get_fdata()
            vmin = np.fmin(np.min(r0_ve[:,:,axial_slice]), np.min(r1_ve[:,:,axial_slice]))
            vmax = np.fmax(np.max(r0_ve[:,:,axial_slice]), np.max(r1_ve[:,:,axial_slice]))

            print(">Printing ventricle residual before registration")
            ax[case_num,1].imshow(r0_ve[:,:,axial_slice].T, vmin=vmin, vmax=vmax, **imshow_kwargs_r)
            ax[case_num,1].set_ylabel('Residual_VE-t=0')

            print(">Printing ventricle residual after registration")
            ax[case_num+1,1].imshow(r1_ve[:,:,axial_slice].T, vmin=vmin, vmax=vmax, **imshow_kwargs_r)
            ax[case_num+1,1].set_ylabel('Residual_VE-t=1')


        # Compute dice of healthy
        patient_seg_in_Aspace = nib.load(os.path.join(reg_case_dir,"patient_seg_in_Aspace.nii.gz")).get_fdata()
        healthy_dice_before[i-1,:] = np.asarray(computeDice(patient_seg, atlas_seg, patient_labels))
        healthy_dice_after[i-1,:] = np.asarray(computeDice(patient_seg_in_Aspace, atlas_seg, patient_labels))

        # compute relative residuals
        pcsf = nib.load(os.path.join(reg_case_dir,"patient_csf.nii.gz")).get_fdata()
        pve = nib.load(os.path.join(reg_case_dir,"patient_ve.nii.gz")).get_fdata()
        pgm = nib.load(os.path.join(reg_case_dir,"patient_gm.nii.gz")).get_fdata()
        pwm = nib.load(os.path.join(reg_case_dir,"patient_edwm.nii.gz")).get_fdata()
        diff0 = np.linalg.norm([np.linalg.norm(acsf-pcsf), np.linalg.norm(ave-pve), np.linalg.norm(agm-pgm), np.linalg.norm(awm-pwm)])
        diff0_ve = np.linalg.norm(ave-pve)

        pcsfA = rescaleImage(nib.load(os.path.join(reg_case_dir,"patient_csf_in_Aspace.nii.gz")).get_fdata())
        pveA = rescaleImage(nib.load(os.path.join(reg_case_dir,"patient_ve_in_Aspace.nii.gz")).get_fdata())
        pgmA = rescaleImage(nib.load(os.path.join(reg_case_dir,"patient_gm_in_Aspace.nii.gz")).get_fdata())
        pwmA = rescaleImage(nib.load(os.path.join(reg_case_dir,"patient_edwm_in_Aspace.nii.gz")).get_fdata())
        diff1 = np.linalg.norm([np.linalg.norm(acsf-pcsfA), np.linalg.norm(ave-pveA), np.linalg.norm(agm-pgmA), np.linalg.norm(awm-pwmA)])
        diff1_ve = np.linalg.norm(ave-pveA)
        res_all[i-1,0] = diff0/C
        res_all[i-1,1] = diff1/C
        res_ve[i-1,0] = diff0_ve/C_ve
        res_ve[i-1,1] = diff1_ve/C_ve



    print(Fore.BLUE + "\n\nDice(c0)\n", c0_dice)
    print(Fore.RED + "\n\nResidual(c0)\n", c0_residual)
    print(Fore.GREEN + "\n\nHealthy Dice\n", healthy_dice_after)

    print(Style.RESET_ALL)

    c0_dice = c0_dice + np.triu(c0_dice).T
    np.fill_diagonal(c0_dice, 1)

    c0_residual = c0_residual + np.triu(c0_residual).T
    np.fill_diagonal(c0_residual, 0)

    columns = ['Patient', 'Method 2', 'Method 3', 'Method 4', 'Method 5', 'Method 6'];
    s = '{0[0]:<15}{0[1]:<15}{0[2]:<15}{0[3]:<15}{0[4]:<15}{0[5]:<15}'.format(columns);

    f = open(os.path.join(input_dir,"registration_stats.txt"), "w")
    for i in range(num_cases):
        case_num = i+2
        f.write("Registration Case " + str(case_num) + "\n---------------------------------------------------------------------------------------\n")
        f.write("Healthy dice before (CSF,VE,GM,WM) = ({0[0]:.2f},{0[1]:.2f},{0[2]:.2f},{0[3]:.2f})\n".format(healthy_dice_before[i,:]))
        f.write("Healthy dice after  (CSF,VE,GM,WM) = ({0[0]:.2f},{0[1]:.2f},{0[2]:.2f},{0[3]:.2f})\n\n".format(healthy_dice_after[i,:]))

        f.write("Total residual before = {0:.4f}\n".format(res_all[i,0]))
        f.write("Total residual after  = {0:.4f}\n\n".format(res_all[i,1]))

        f.write("Ventricle residual before = {0:.4f}\n".format(res_ve[i,0]))
        f.write("Ventricle residual after  = {0:.4f}\n\n".format(res_ve[i,1]))

        f.write("c0_Residual\n")
        f.write(s)
        f.write("\n")
        c = ''
        for j in range(num_cases+1):
            c = c + '{0[' + str(j) + ']:<15.2f}'
        f.write(c.format(c0_residual[:,i+1]))

        f.write("\n\n")
        f.write("c0_dice\n")
        f.write(s)
        f.write("\n")
        c = ''
        for j in range(num_cases+1):
            c = c + '{0[' + str(j) + ']:<15.2f}'
        f.write(c.format(c0_dice[:,i+1]))
        f.write("\n\n\n")

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.1);
    fig.savefig(os.path.join(input_dir,"registration_comparison.pdf"), dpi=1200)



###
### ------------------------------------------------------------------------ ###
def computeTumorStats(features, patient_ref_, t1_recon_seg, t0_recon_seg, c1_recon, c0_recon, c1_pred12, c1_pred15, data, component_mask,  patient_labels, tumor_output_path):
    """
    @short - computes the dice coefficient for the tumor and further statistics
           - assumes t1_recon_seg to have the segmentation definend in inverse.cpp
            bg : 0, gm : 1, wm : 2, csf : 3, tu : 4
    """

    patient_ref = patient_ref_.get_fdata();
    affine      = patient_ref_.affine;
    # bool map patient_ref wm, gm, csf, tu
    pref_csf = patient_ref == patient_labels['csf'];
    if 'vt' in patient_labels:
        pref_csf = np.logical_or(pref_csf, patient_ref == patient_labels['vt'])
    pref_gm  = patient_ref == patient_labels['gm'];
    pref_wm  = patient_ref == patient_labels['wm'];
    pref_en  = patient_ref == patient_labels['en'];
    pref_ed  = patient_ref == patient_labels['ed'];
    pref_nec = patient_ref == patient_labels['nec'];
    # combine labels
    pref_brain = np.logical_or.reduce((pref_wm, pref_gm, pref_csf));
    pref_wt    = np.logical_or.reduce((pref_ed, pref_nec, pref_en));
    pref_tc    = np.logical_or.reduce((pref_nec, pref_en));

    # bool map t1_recon_seg wm, gm, csf, tu
    prec_csf    = (t1_recon_seg == 3);  # csf label: 3
    prec_gm     = (t1_recon_seg == 1);  # gm  label: 1
    prec_wm     = (t1_recon_seg == 2);  # wm  label: 2
    prec_wt     = (t1_recon_seg == 4);  # tu  label: 4
    prec_tc9    = (c1_recon  >  0.9);   # define tc where c(1)  >  0.9
    prec_tc8    = (c1_recon  >  0.8);   # define tc where c(1)  >  0.8
    prec_nec1   = (c1_recon  >= 0.99);  # define nec where c(1) >= 0.99
    if c1_pred15 != None:
        pred15_tc9  = (c1_pred15 >  0.9);   # define tc where c(t=1.5)  >  0.9
    # edema
    prec_ed8    = np.logical_and((c1_recon < 0.8), (c1_recon > 0.02));    # define ed where 0.8 < c(1) < 0.02
    prec_ed7    = np.logical_and((c1_recon < 0.7), (c1_recon > 0.02));    # define ed where 0.7 < c(1) < 0.02
    prec_ed6    = np.logical_and((c1_recon < 0.6), (c1_recon > 0.02));    # define ed where 0.6 < c(1) < 0.02

    # edema and tumor core masked
    ed_mask     = pref_ed.astype(int);
    tc_mask     = pref_tc.astype(int);
    wt_mask     = pref_wt.astype(int);
    c1_ed       = np.multiply(c1_recon, ed_mask);    # get c1 masked by ED
    c1_tc       = np.multiply(c1_recon, tc_mask);    # get c1 masked by TC
    c1_b_no_wt  = np.multiply(c1_recon, 1-wt_mask);  # get c1 masked by B/WT
    fio.writeNII(c1_ed,      os.path.join(tumor_output_path, "c1(ED).nii.gz"), affine);
    fio.writeNII(c1_tc,      os.path.join(tumor_output_path, "c1(TC).nii.gz"), affine);
    fio.writeNII(c1_b_no_wt, os.path.join(tumor_output_path, "c1(B\WT).nii.gz"), affine);

    # bool map t0_recon_seg wm, gm, csf, tu
    prec0_csf = (t0_recon_seg == 3);
    prec0_gm  = (t0_recon_seg == 1);
    prec0_wm  = (t0_recon_seg == 2);
    prec0_c0  = (t0_recon_seg == 4);

    # -- a) compute dice --
    features['csf_dice']  = 1.0 - distance.dice(pref_csf.flatten(), prec_csf.flatten());
    features['gm_dice']   = 1.0 - distance.dice(pref_gm.flatten(),  prec_gm.flatten());
    features['wm_dice']   = 1.0 - distance.dice(pref_wm.flatten(),  prec_wm.flatten());
    features['wt_dice']   = 1.0 - distance.dice(pref_wt.flatten(),  prec_wt.flatten());
    features['nec_dice']  = 1.0 - distance.dice(pref_nec.flatten(), prec_wt.flatten());
    features['tc_dice']   = 1.0 - distance.dice(pref_tc.flatten(),  prec_wt.flatten());
    features['tc_c(1)[x>0.9]_dice']  = 1.0 - distance.dice(pref_tc.flatten(),  prec_tc9.flatten());
    features['tc_c(1)[x>0.8]_dice']  = 1.0 - distance.dice(pref_tc.flatten(),  prec_tc8.flatten());
    features['nec_c(1)[x=1.0]_dice'] = 1.0 - distance.dice(pref_nec.flatten(), prec_nec1.flatten());

    features['ed_c(1)[0.02<x<0.8]_dice']  = 1.0 - distance.dice(pref_ed.flatten(),  prec_ed8.flatten());
    features['ed_c(1)[0.02<x<0.7]_dice']  = 1.0 - distance.dice(pref_ed.flatten(),  prec_ed7.flatten());
    features['ed_c(1)[0.02<x<0.6]_dice']  = 1.0 - distance.dice(pref_ed.flatten(),  prec_ed6.flatten());

    # -- b) max and min c(1) value in edema and tumor core mask --
    features['max{c(1)|_ED}'] = np.amax(c1_ed);
    features['min{c(1)|_ED}'] = np.amin(c1_ed);
    features['max{c(1)|_TC}'] = np.amax(c1_tc);
    features['min{c(1)|_TC}'] = np.amin(c1_tc);
    features['int{c(1)}']     = 2*math.pi/256 * np.sum(c1_recon.flatten());
    features['int{c(1)|_ED}'] = 2*math.pi/256 * np.sum(c1_ed.flatten());
    features['int{c(1)|_TC}'] = 2*math.pi/256 * np.sum(c1_tc.flatten());
    features['int{c(1)|_B-WT}'] = 2*math.pi/256 * np.sum(c1_b_no_wt.flatten());

    # -- c) rel. volume --
    data_nonsmooth = pref_tc.astype(float);
    brain = float(np.sum(pref_brain.flatten()));
    features['vol(WT)_a']   = np.sum(pref_wt.flatten());
    features['vol(ED)_a']   = np.sum(pref_ed.flatten());
    features['vol(EN)_a']   = np.sum(pref_en.flatten());
    features['vol(NEC)_a']  = np.sum(pref_nec.flatten());
    features['vol(TC)_a']   = np.sum(pref_tc.flatten());
    features['vol(c(0))_a'] = np.sum(prec0_c0.flatten());
    features['vol(WT)_r']   = features['vol(WT)_a'] / brain;
    features['vol(ED)_r']   = features['vol(ED)_a'] / brain;
    features['vol(EN)_r']   = features['vol(EN)_a'] / brain;
    features['vol(NEC)_r']  = features['vol(NEC)_a'] / brain;
    features['vol(TC)_r']   = features['vol(TC)_a'] / brain;
    features['vol(c(0))_r'] = features['vol(c(0))_a'] / brain;
    features['vol(c(1)[x>0.9])_a']   = np.sum(prec_tc9.flatten());
    features['vol(c(1)[x>0.9])_r']   = features['vol(c(1)[x>0.9])_a'] / brain;
    features['vol(c(1.5)[x>0.9])_r'] = -1;
    if c1_pred15 != None:
        features['vol(c(1.5)[x>0.9])_r']   = np.sum(pred15_tc9.flatten()) / brain;

    x_cm_tc  = {}
    x_cm_nec = {}
    x_cm_c0  = {}
    nnc = 0
    if component_mask:
        for nc in range(min(3,len(component_mask))):
            # convert to brats dimensions
            mask = imgtools.resizeImage(component_mask[nc], tuple(patient_ref.shape), 0);
            tc_in_comp  = np.multiply(pref_tc.astype(int),    mask.astype(int));
            ed_in_comp  = np.multiply(pref_ed.astype(int),    mask.astype(int));
            nec_in_comp = np.multiply(pref_nec.astype(int),   mask.astype(int));
            en_in_comp  = np.multiply(pref_en.astype(int),    mask.astype(int));
            c0_in_comp  = np.multiply(c0_recon.astype(float), mask.astype(int));

            # compute center of mass of necrotic tumor in component #nc
            x_cm = scipy.ndimage.measurements.center_of_mass(tc_in_comp)
            x_cm_tc[nc] = tuple([2 * math.pi * x_cm[0] / patient_ref.shape[0], 2 * math.pi * x_cm[1] / patient_ref.shape[1], 2 * math.pi * x_cm[2] / patient_ref.shape[2]]);
            features['cm(TC|_#c) (#c='+str(nc)+')']   =  "(%1.1f, %1.1f, %1.1f)px" % (x_cm[0], x_cm[1], x_cm[2]) if nc < 3 else "n/a";
            x_cm = scipy.ndimage.measurements.center_of_mass(nec_in_comp)
            x_cm_nec[nc] = tuple([2 * math.pi * x_cm[0] / patient_ref.shape[0], 2 * math.pi * x_cm[1] / patient_ref.shape[1], 2 * math.pi * x_cm[2] / patient_ref.shape[2]]);
            features['cm(NEC|_#c) (#c='+str(nc)+')']   =  "(%1.1f, %1.1f, %1.1f)px" % (x_cm[0], x_cm[1], x_cm[2]) if nc < 3 else "n/a";
            x_cm = scipy.ndimage.measurements.center_of_mass(c0_in_comp)
            x_cm_c0[nc] = tuple([2 * math.pi * x_cm[0] / patient_ref.shape[0], 2 * math.pi * x_cm[1] / patient_ref.shape[1], 2 * math.pi * x_cm[2] / patient_ref.shape[2]]);
            features['cm(c(0)|_#c) (#c='+str(nc)+')']   =  "(%1.1f, %1.1f, %1.1f)px" % (x_cm[0], x_cm[1], x_cm[2]) if nc < 3 else "n/a";

            features['vol(TC|_#c)_a(#c='+str(nc)+')']  =  np.sum(tc_in_comp.flatten())                       if nc < 3 else -1;
            features['vol(TC|_#c)_r(#c='+str(nc)+')']  =  np.sum(tc_in_comp.flatten())/features['vol(TC)_a'] if nc < 3 else -1;
            features['vol(ED|_#c)_a(#c='+str(nc)+')']  =  np.sum(ed_in_comp.flatten())                       if nc < 3 else -1;
            features['vol(ED|_#c)_r(#c='+str(nc)+')']  =  np.sum(ed_in_comp.flatten())/features['vol(ED)_a'] if nc < 3 else -1;
            features['vol(EN|_#c)_a(#c='+str(nc)+')']  =  np.sum(ed_in_comp.flatten())                       if nc < 3 else -1;
            features['vol(EN|_#c)_r(#c='+str(nc)+')']  =  np.sum(ed_in_comp.flatten())/features['vol(TC)_a'] if nc < 3 else -1;
            features['vol(NEC|_#c)_a(#c='+str(nc)+')'] =  np.sum(ed_in_comp.flatten())                       if nc < 3 else -1;
            features['vol(NEC|_#c)_r(#c='+str(nc)+')'] =  np.sum(ed_in_comp.flatten())/features['vol(TC)_a'] if nc < 3 else -1;

            features['l2[c(0)|_#c]_a(#c='+str(nc)+')'] =  distance.euclidean(np.zeros_like(c0_in_comp.flatten()), c0_in_comp.flatten()) if nc < 3 else -1;
            features['l2[c(0)|_#c]_r(#c='+str(nc)+')'] =  features['l2[c(0)|_#c]_a(#c='+str(nc)+')'] / distance.euclidean(np.zeros_like(c0_recon.flatten()), c0_recon.astype(float).flatten()) if nc < 3 else -1;

            nnc += 1
        if nnc < 2: # less than 3 components, fill with -1 dummy vals.
            for i in range (nnc, 3):
                features['cm(NEC|_#c) (#c='+str(nc)+')']   = "n/a";
                features['cm(TC|_#c) (#c='+str(nc)+')']    = "n/a";
                features['cm(c(0)|_#c) (#c='+str(nc)+')']  = "n/a";
                features['vol(TC|_#c)_a(#c='+str(nc)+')']  = -1;
                features['vol(TC|_#c)_r(#c='+str(nc)+')']  = -1;
                features['vol(ED|_#c)_a(#c='+str(nc)+')']  = -1;
                features['vol(ED|_#c)_r(#c='+str(nc)+')']  = -1;
                features['vol(EN|_#c)_a(#c='+str(nc)+')']  = -1;
                features['vol(EN|_#c)_r(#c='+str(nc)+')']  = -1;
                features['vol(NEC|_#c)_a(#c='+str(nc)+')'] = -1;
                features['vol(NEC|_#c)_r(#c='+str(nc)+')'] = -1;
                features['l2[c(0)|_#c]_a(#c='+str(nc)+')'] = -1;
                features['l2[c(0)|_#c]_r(#c='+str(nc)+')'] = -1;


    # -- d) rel. surface --
    # compute area of spheres that have same volume as ROI
    area_sphere_tc  = 4 * math.pi * (3/(4*math.pi) * features['vol(TC)_a']) ** (2./3);
    area_sphere_ed  = 4 * math.pi * (3/(4*math.pi) * features['vol(ED)_a']) ** (2./3);
    area_sphere_nec = 4 * math.pi * (3/(4*math.pi) * features['vol(NEC)_a']) ** (2./3);
    features['area(TC)_a'] = 0;
    features['area(ED)_a'] = 0;
    features['area(NEC)_a'] = 0;
    dx,dy,dz = np.gradient(pref_tc.astype(int), 1./patient_ref.shape[0], 1./patient_ref.shape[1], 1./patient_ref.shape[2]);
    features['area(TC)_a'] = np.sum(np.multiply(pref_tc.astype(int), np.sqrt(np.multiply(dx,dx) + np.multiply(dy,dy) + np.multiply(dz,dz))));
    dx,dy,dz = np.gradient(pref_ed.astype(int), 1./patient_ref.shape[0], 1./patient_ref.shape[1], 1./patient_ref.shape[2]);
    features['area(ED)_a'] = np.sum(np.multiply(pref_ed.astype(int), np.sqrt(np.multiply(dx,dx) + np.multiply(dy,dy) + np.multiply(dz,dz))));
    dx,dy,dz = np.gradient(pref_nec.astype(int), 1./patient_ref.shape[0], 1./patient_ref.shape[1], 1./patient_ref.shape[2]);
    features['area(NEC)_a'] = np.sum(np.multiply(pref_nec.astype(int), np.sqrt(np.multiply(dx,dx) + np.multiply(dy,dy) + np.multiply(dz,dz))));
    features['area(TC)_r']  = features['area(TC)_a']  / area_sphere_tc;
    features['area(ED)_r']  = features['area(ED)_a']  / area_sphere_ed;
    features['area(NEC)_r'] = features['area(NEC)_a'] / area_sphere_tc;

    # -- e) integral fractions --
    features['int{c(0)}/int{c(1)}']     = np.sum(c0_recon.flatten())  / sum(c1_recon.flatten());
    features['int{c(1.5)}/int{c(1.2)}'] = -1;
    features['int{c(1.5)}/int{c(1)}']   = -1;
    features['int{c(1.5)}/int{TC}']     = -1;
    if c1_pred15 != None:
        features['int{c(1.5)}/int{c(1)}'] = np.sum(c1_pred15.flatten()) / sum(c1_recon.flatten());
        features['int{c(1.5)}/int{TC}']   = np.sum(c1_pred15.flatten()) / sum(data.flatten());
    if c1_pred12 != None and c1_pred15 != None:
        features['int{c(1.5)}/int{c(1.2)}'] = np.sum(c1_pred15.flatten()) / sum(c1_pred12.flatten());

    # compute l2-error (everywhere and at observation points)
    diff_virg = c1_recon - data;
    diff_virg_nonsmooth = c1_recon - data_nonsmooth;
    obs_mask = ~pref_ed;
    diff_obs              = np.multiply(c1_recon, obs_mask)  - data;
    diff_obs_nonsmooth    = np.multiply(c1_recon, obs_mask)  - data_nonsmooth;
    features['l2[c(1)-TC]']           = np.linalg.norm(diff_virg.flatten(), 2)           / np.linalg.norm(data.flatten(), 2)
    features['l2[c(1)-TC]_nonsmooth'] = np.linalg.norm(diff_virg_nonsmooth.flatten(), 2) / np.linalg.norm(data_nonsmooth.flatten(), 2)
    features['l2[Oc(1)-TC]']            = np.linalg.norm(diff_obs.flatten(), 2)            / np.linalg.norm(data.flatten(), 2)
    features['l2[Oc(1)-TC]_nonsmooth']  = np.linalg.norm(diff_obs_nonsmooth.flatten(), 2)  / np.linalg.norm(data_nonsmooth.flatten(), 2)

    print("healthy tissue dice: (features['csf_dice'], features['gm_dice'], features['wm_dice'])    =", features['csf_dice'],features['gm_dice'],features['wm_dice']);
    print("tumor dice (max):    (features['wt_dice'], features['tc_dice'], features['nec_dice'])    =", features['wt_dice'],features['tc_dice'],features['nec_dice']);
    print("tumor dice (> x):    (features['tc_c(1)[x>0.9]_dice'], features['tc_c(1)[x>0.8]_dice'], features['nec_c(1)[x=1.0]_dice']) =", features['tc_c(1)[x>0.9]_dice'],features['tc_c(1)[x>0.8]_dice'],features['nec_c(1)[x=1.0]_dice']);
    print("ed dice (x<ed<0.02): (features['ed_c(1)[0.02<x<0.8]_dice'], features['ed_c(1)[0.02<x<0.7]_dice'], features['ed_c(1)[0.02<x<0.6]_dice'])  =", features['ed_c(1)[0.02<x<0.8]_dice'],features['ed_c(1)[0.02<x<0.7]_dice'],features['ed_c(1)[0.02<x<0.6]_dice']);
    print("c(ed,1) (max,min):                                   =", features['max{c(1)|_ED}'], features['min{c(1)|_ED}']);
    print("c(tc,1) (max,min):                                   =", features['max{c(1)|_TC}'], features['min{c(1)|_TC}']);
    print("stats: #tu/#brain    (wt,ed,en,nec,c0)               =", features['vol(WT)_r'], features['vol(ED)_r'], features['vol(EN)_r'], features['vol(NEC)_r'], features['vol(c(0))_r']);
    print("stats: #tc/#brain    (ref_tc,rec_tc9,pred_tc9)       =", features['vol(TC)_r'],features['vol(c(1)[x>0.9])_r'],features['vol(c(1.5)[x>0.9])_r']);
    print("stats: int_ED c(1), int_TC c(t), int_B/WT c(1), int_B c(1) =", features['int{c(1)|_ED}'],features['int{c(1)|_TC}'],features['int{c(1)|_B-WT}'], features['int{c(1)}']);
    print("stats: int c(0)   / int c(1)   =", features['int{c(0)}/int{c(1)}']);
    print("stats: int c(1.5) / int c(1)   =", features['int{c(1.5)}/int{c(1)}']);
    print("stats: int c(1.5) / int c(1.2) =", features['int{c(1.5)}/int{c(1.2)}']);
    print("stats: int c(1.5) / int d      =", features['int{c(1.5)}/int{TC}']);
    print("l2err_c(1):(smooth) (virg,obs) =", features['l2[c(1)-TC]'], features['l2[Oc(1)-TC]']);
    print("l2err_c(1):         (virg,obs) =", features['l2[c(1)-TC]_nonsmooth'], features['l2[Oc(1)-TC]_nonsmooth']);


###
### ------------------------------------------------------------------------ ###
def convertImagesToOriginalSize(input_path, tumor_output_path, reference_image_path, gridcont=False):
    """
    @short - resamples the output images back to the original dimension
             (same dimension as reference image)
    """
    templates = {}
    input_path = input_path;
    reg_output_path = os.path.join(input_path, 'registration/')
    # tumor_output_path = os.path.join(input_path, 'tumor_inversion/');
    ref_img = nib.load(reference_image_path);
    templates['256'] = ref_img;
    affine = ref_img.affine;
    output_size = ref_img.shape;
    # write reference image to patient dir
    filename = ntpath.basename(reference_image_path);
    fio.writeNII(ref_img.get_fdata(), os.path.join(tumor_output_path, filename), affine);
    #resample ref image for grid cont vis
    if gridcont:
        new_affine = np.copy(ref_img.affine)
        print("original affine:\n", new_affine);
        row,col = np.diag_indices(new_affine.shape[0])
        new_affine[row,col] = np.array([-2,-2,2,1]);
        print("nx128 affine:\n", new_affine);
        resampled_template_128 = nib.processing.resample_from_to(ref_img, (np.multiply(0.5,ref_img.shape).astype(int), new_affine))
        templates['128'] = resampled_template_128;
        nib.save(resampled_template_128, os.path.join(tumor_output_path,"template_nx128.nii.gz"));

        new_affine = np.copy(ref_img.affine)
        print("original affine:\n", new_affine);
        row,col = np.diag_indices(new_affine.shape[0])
        new_affine[row,col] = np.array([-4,-4,4,1]);
        print("nx64 affine:\n", new_affine);
        resampled_template_64 = nib.processing.resample_from_to(ref_img, (np.multiply(0.25,ref_img.shape).astype(int), new_affine))
        templates['64'] = resampled_template_64;
        nib.save(resampled_template_64, os.path.join(tumor_output_path,"template_nx64.nii.gz"));


    # get list of nc and nii files in claire output
    niifiles_claire = imgtools.getNIIImageList(reg_output_path);
    ncfiles_claire = imgtools.getNetCDFImageList(reg_output_path);
    # get list of nc and nii files in tumor inversion output
    niifiles_tumor = imgtools.getNIIImageList(tumor_output_path);
    ncfiles_tumor  = imgtools.getNetCDFImageList(tumor_output_path);
    levels = [64,128,256] if gridcont else [256];
    # levels = [256];

    # gather files
    niifiles = niifiles_claire + niifiles_tumor;
    ncfiles  = ncfiles_claire + ncfiles_tumor;
    # out path
    new_output_path = input_path
    if not os.path.exists(new_output_path):
        os.mkdir(new_output_path)

    # loops over netcdf files to resample them and save as nii
    print('post processing claire-netcdf files')
    for f in ncfiles_claire:
        if '128_velocity' not in f:
            print('processing '+f)
            data = fio.readNetCDF(f);
            data = np.swapaxes(data,0,2);
            if 'seg' in f:
                print('segmentation file found, using NN interpolation')
                newdata = imgtools.resizeImage(data, tuple(output_size), 0);
            else:
                newdata = imgtools.resizeImage(data, tuple(output_size), 1);
            filename = ntpath.basename(f);
            filename, fileext = os.path.splitext(filename);
            newfilename = filename + '.nii.gz';
            fio.writeNII(newdata, os.path.join(reg_output_path, newfilename), affine);

    print('post processing tumor-netcdf files')
    # if not gridcont:
    #     for f in ncfiles_tumor:
    #         print('processing '+f)
    #         data = fio.readNetCDF(f);
    #         data = np.swapaxes(data,0,2);
    #         if 'seg' in f:
    #             print('segmentation file found, using NN interpolation')
    #             newdata = imgtools.resizeImage(data, tuple(output_size), 0);
    #         else:
    #             newdata = imgtools.resizeImage(data, tuple(output_size), 1);
    #
    #         filename = ntpath.basename(f);
    #         filename, fileext = os.path.splitext(filename);
    #         newfilename = filename + '.nii.gz';
    #         fio.writeNII(newdata, os.path.join(tumor_output_path, newfilename), affine);
    # else:
    for l in levels:
        print("## processing level",l," ##")
        template = templates[str(l)];
        tu_out_path = os.path.join(input_path, 'tumor_inversion/nx'+str(l)+'/');
        dirs = os.listdir(tu_out_path)
        for dir in dirs:
            #if not "obs-1.0" in dir:
            if not "obs" in dir:
                continue;
            print('converting images in ',dir)
            print('voxel axis: ', nib.aff2axcodes(template.affine))
            print('affine:\n',     template.affine)
            ncfiles_tumor = imgtools.getNetCDFImageList(os.path.join(tu_out_path, dir + '/'));
            for f in ncfiles_tumor:
                print('processing ', f)
                data = fio.readNetCDF(f);
                data = np.swapaxes(data,0,2);
                if 'seg' in f:
                    print('segmentation file found, using NN interpolation')
                    newdata = imgtools.resizeImage(data, tuple(template.shape), 0);
                else:
                    newdata = imgtools.resizeImage(data, tuple(template.shape), 1);

                filename = ntpath.basename(f);
                filename, fileext = os.path.splitext(filename);
                newfilename = filename + '.nii.gz';
                fio.writeNII(newdata, os.path.join(os.path.join(tu_out_path, dir), newfilename), template.affine);


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
        if not os.path.isfile(os.path.join(path, cmvec)):
            cmvec = 'phi-mesh-scaled.dat'

    simga = 0;
    level = int(path.split("nx")[-1].split("/")[0].split("-")[0]);
    hx    = 1/float(level);
    phix, phiy, phiz = np.loadtxt(os.path.join(path, cmvec), comments = ["]", "#"], delimiter=',', skiprows=2, unpack=True);
    p_vec            = np.loadtxt(os.path.join(path, pvec),  comments = ["]", "#"],                skiprows=1);

    # one p actiavation only
    if not isinstance(phix, collections.Iterable):
        print("Only one p activation found")
        p_vec = np.array([p_vec])
        phix = np.array([phix])
        phiy = np.array([phiy])
        phiz = np.array([phiz])
    if os.path.exists(os.path.join(path, cmvec)):
        with open(os.path.join(path, cmvec), 'r') as f:
            sigma = float(f.readlines()[0].split("sigma =")[-1].split(",")[0])
    print(" .. reading p vector and Gaussian centers of length ",p_vec.shape," from level ", level, " with sigma =", sigma);

    phi = []
    for x,y,z in zip(phix,phiy,phiz):
        phi.append(tuple([x,y,z]));
    return p_vec, phi, sigma, hx, level

###
### ------------------------------------------------------------------------ ###
def connectedComponentsP(pvec, phi, sigma, hx):
    """
    @brief: computes connected components in Gaussian center point cloud
    """
    # print("phi set:\n", phi);
    # print("p vector:\n", pvec);

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
def cont(slice, cmap, thresh=0.3, v_max=None, v_min=None, clip01=True):

    slice_clipped = np.clip(slice, 0, 1) if clip01 else slice;
    # alphas = Normalize(0, thresh, clip=True)(slice_clipped)
    max = np.amax(slice_clipped) if v_max == None else v_max;
    min = np.amin(slice_clipped) if v_min == None else v_min;
    norm = mpl.cm.colors.Normalize(min, max);
    slice_normalized = Normalize(min, max)(slice_clipped);
    # cmap_ = cmap(slice_normalized)
    # cmap_[..., -1] = alphas
    # return cmap_, norm
    return slice_normalized, norm


def parseTumorLog(xdir, features):
  pass;

###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process input images')
    parser.add_argument ('-input_path',           type = str,          help = 'path to the results folder');
    parser.add_argument ('-atlas_path',           type = str,          help = 'path to atlas folder')
    parser.add_argument ('-tu_path',              type = str,          help = 'path to tumor solver results');   # REMOVE
    parser.add_argument ('-rdir',                 type = str,          help = 'path to tumor solver results');
    parser.add_argument ('-tu_dir',               type = str,          help = 'results dir name');
    parser.add_argument ('-reference_image_path', type = str,          help = 'path to a reference image for resizing (using the header), also patient segmentation image')
    parser.add_argument ('-patient_labels',       type = str,          help = 'patient labels');
    parser.add_argument ('-convert_images',       action='store_true', help = 'convert all output images back to original dimension');
    parser.add_argument ('-gridcont',             action='store_true', help = 'grid continuation is active');
    parser.add_argument ('-compute_dice_healthy', action='store_true', help = 'compute dice scores');
    parser.add_argument ('-compute_tumor_stats',  action='store_true', help = 'compute dice scores of tumor tissue and other statistics');
    parser.add_argument ('-analyze_concomps',     action='store_true', help = 'analyze connected components');
    parser.add_argument ('-analyze_registration', action='store_true', help = 'analyze registration results');
    parser.add_argument ('--obs_lambda',          type = float, default = 1,   help = 'parameter to control observation operator OBS = TC + lambda (1-WT)');
    parser.add_argument ('-generate_slices',      action='store_true', help = 'generates charts of slices');
    parser.add_argument ('--prediction',          action='store_true', help = 'indicates if to postprocess prediction files');
    args = parser.parse_args();


    if args.rdir == None:
        args.rdir = "obs-{0:1.1f}".format(args.obs_lambda);
    elif "cm-data" in args.rdir:
        args.rdir = "cm-data-obs-{0:1.1f}".format(args.obs_lambda);
    elif "obs" in args.rdir:
        args.rdir = "obs-{0:1.1f}".format(args.obs_lambda);


    FEATURES = {}
    patient_labels = {};
    # paths
    input_path = args.input_path;
    reg_output_path = os.path.join(input_path, 'registration/')
    path_256 = os.path.join(os.path.join(args.input_path, 'tumor_inversion'), 'nx256');
    path_256 = os.path.join(path_256, args.rdir);

    if args.patient_labels is not None:
        for x in args.patient_labels.split(','):
            patient_labels[int(x.split('=')[0])] = x.split('=')[1];
        patient_label_rev = {v:k for k,v in patient_labels.items()};

    if args.analyze_registration:
        if args.patient_labels is None:
            print("Need to provide patient labels, exiting")
            sys.exit()
        analyzeRegistration(args.input_path, args.atlas_path, patient_label_rev)
        sys.exit()

    # get bratsID
    for x in args.reference_image_path.split('/'):
        if x.startswith('Brats'):
            bratsID = x;
            break;
    bratsID = bratsID.split("_")[0] + "_" +  bratsID.split("_")[1] + "_" +  bratsID.split("_")[2] + '_1';

    # convert all images to original dimension
    if args.convert_images:
        convertImagesToOriginalSize(args.input_path, path_256, args.reference_image_path, args.gridcont);

    # compute dice scores
    if args.compute_dice_healthy:
        if args.patient_labels is None:
            print("Need to provide patient labels, exiting")
            sys.exit()
        patient_ref = nib.load(args.reference_image_path)
        patient_ref = patient_ref.get_fdata();
        atlas_img = nib.load(reg_output_path + "atlas_in_Pspace_seg.nii.gz")
        atlas_img = atlas_img.get_fdata()
        FEATURES[256]['csf_dice'],FEATURES[256]['gm_dice'],FEATURES[256]['wm_dice'] = computeDice(patient_ref, atlas_img, patient_label_rev);


    if args.analyze_concomps:
        if args.obs_lambda == None:
            args,obs_lambda = 1;

        levels      = [64, 128, 256] if args.gridcont else [256]
        pvec        = {}
        phi         = {}
        sigma       = {}
        hx          = {}
        n           = {}
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
        Xx          = []
        Yy          = []
        Zz          = []
        markers     = ['o', 'D', 'o'] if args.gridcont else ['o']
        colors      = [cl2,cl1,cl3]   if args.gridcont else [cl3]
        l2err_TC            = {};
        l2normref_TC        = {};
        l2err_percomp_TC    = {};
        l2normref_percomp   = {};
        l2errc0_over_levels = {};
        phi_labeled_dcomp   = {};
        pi_labeled_dcomp    = {};
        wcm_labeled_dcomp   = {};
        dist_wcmSOL_cmDATA  = {};

        ccfile = os.path.join(args.input_path,'components_'+args.rdir+'.txt')
        concomp_file = open(ccfile,'w');
        if args.generate_slices:
            fig_l2d, ax_l2d = plt.subplots(1,3, figsize=(12,4));
        for l, m, cc in zip(levels, markers, colors):

            FEATURES[l] = {};
            # paths
            lvl_prefix = os.path.join(os.path.join(args.input_path, 'tumor_inversion'), 'nx' + str(l));
            res_path   = os.path.join(lvl_prefix, args.rdir);
            init_path  = os.path.join(lvl_prefix, 'init');
            vis_path   = os.path.join(args.input_path, 'vis_'+args.rdir)
            out_path   = os.path.join(args.input_path, "input");

            template_fname = "template_nx"+str(l)+".nii.gz" if l < 256 else bratsID + '_seg_tu.nii.gz'
            template_img = nib.load(os.path.join(path_256, template_fname));
            template     = template_img.get_fdata();
            template_256 = nib.load(os.path.join(path_256, bratsID + '_seg_tu.nii.gz'));
            affine_256   = template_256.affine
            template_256 = template_256.get_fdata();

            # ### concomp DATA ###
            level = int(res_path.split("nx")[-1].split("/")[0].split("-")[0]);
            if level != l:
                print(color.FAIL + "Error %d != %d " + color.ENDC % (level, l));
            print("\n (1) computing connected components of data\n");
            labeled[l], comps_data[l], ncomps_data[l], xcm_data_px[l], xcm_data[l], relmass[l] = connectedComponentsData(init_path, level);
            fio.createNetCDF(os.path.join(out_path, 'data_comps_nx'+str(level)+'.nc') , np.shape(labeled[l]), np.swapaxes(labeled[l],0,2));

            # ### concomp SOL ###
            print("\n (2) computing connected components of solution\n");
            pvec[l], phi[l], sigma[l], hx[l], n[l] = readDataSolution(res_path);
            ncomps_sol[l], comps_sol[l], p_comp[l], xcm_sol[l], xcm_tot_sol[l] = connectedComponentsP(pvec[l], phi[l], sigma[l], hx[l]);

            # ### cluster phi centers and p_i's of solution according to comp(DATA) and compute weighted center of mass ###
            phi_labeled_dcomp[l], pi_labeled_dcomp[l], wcm_labeled_dcomp[l] =  weightedCenterPiForDataComponents(pvec[l], phi[l], hx[l], labeled[l], ncomps_data[l]);

            # ### distance between weighted center of mass of SOL, labeled by data and center of mass of DATA ###
            dist_wcmSOL_cmDATA[l] = {};
            for i in range(ncomps_data[l]):
                if np.array_equal(np.array([0,0,0]), wcm_labeled_dcomp[l][i+1]):
                    dist_wcmSOL_cmDATA[l][i] = float('nan');
                else:
                    dist_wcmSOL_cmDATA[l][i] = dist(xcm_data[l][i], wcm_labeled_dcomp[l][i+1]); # euclidean dist; wc_pi_dcomp has comp #0 = bg, so adding 1

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
            iii = 0
            for i in range(ncomps_data[l]):
                if i < 3: FEATURES[l]["dist[wcm(p|_#c) - cm(TC|_#c)]_(c="+str(i)+")"] = dist_wcmSOL_cmDATA[l][i];
                w  += "{0:1.2e}".format(dist_wcmSOL_cmDATA[l][i])
                w2 += "{0:1.1f}".format((dist_wcmSOL_cmDATA[l][i] * float(l) / (2*math.pi) ));
                iii += 1;
                if i < ncomps_data[l] - 1:
                    w  += ', '
                    w2 += ', '
            if iii < 3: 
                for iiii in range(iii,3): 
                    FEATURES[l]["dist[wcm(p|_#c) - cm(TC|_#c)]_(c="+str(i)+")"] = -1;
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


            # compute l2-error || (c(1) - d)|_TC ||_2 restricted to TC, break up into sub-components
            print("\n (3) computing l2-error in TC (c(1) rescaled to 1)\n");
            l2err_percomp_TC[l]  = {};
            l2normref_percomp[l] = {};
            data_nc   = fio.readNetCDF(os.path.join(res_path,"data.nc"));
            data_nc   = np.swapaxes(data_nc,0,2);
            dims_nc   = data_nc.shape;
            c1rec_nc  = fio.readNetCDF(os.path.join(res_path,"cRecon.nc"));
            c1rec_nc  = np.swapaxes(c1rec_nc,0,2);
            c1min     = np.amin(c1rec_nc.flatten());
            c1max     = np.amax(c1rec_nc.flatten());
            c1recs_nc = c1rec_nc * (1./c1max);
            c1smin    = np.amin(c1recs_nc.flatten());
            c1smax    = np.amax(c1recs_nc.flatten());
            print("     .. min/max{c(1)} %1.2e/%1.2e --> rescaling in [min,1] .. checking %1.2e/%1.2e" % (c1min, c1max, c1smin, c1smax))
            nnc = 0;
            for nc in range(ncomps_data[l]):
                l2err_percomp_TC[l][nc]  =  distance.euclidean(c1recs_nc.flatten(), data_nc.flatten(), comps_data[l][nc].flatten());
                l2normref_percomp[l][nc] =  distance.euclidean(np.zeros_like(data_nc.flatten()), data_nc.flatten(), comps_data[l][nc].flatten());
                if nc < 3:
                    FEATURES[l]['l2[c(1)|_#c - TC|_#c]_a(#c='+str(nc)+')'] =  l2err_percomp_TC[l][nc]
                    FEATURES[l]['l2[TC|_#c]_a(#c='+str(nc)+')'] =  l2err_percomp_TC[l][nc]

            if nnc < 2: # less than 3 components, fill with -1 dummy vals.
                for i in range (nnc, 3):
                    FEATURES[l]['l2[c(1)|_#c - TC|_#c]_a(#c='+str(nc)+')'] = -1;
                    FEATURES[l]['l2[TC|_#c]_a(#c='+str(nc)+')'] = -1;


            mask_TC = labeled[l] > 0;
            l2normref_TC[l] = distance.euclidean(np.zeros_like(data_nc.flatten()), data_nc.flatten());
            l2err_TC[l]     = distance.euclidean(c1recs_nc.flatten(), data_nc.flatten(), mask_TC.flatten());


            # write cm(DATA) to phi-cm-data.tx and p-cm-data.txt file
            if l == 256:
                phi_cm_data_file = open(os.path.join(path_256,'phi-cm-data.txt'),'w');
                p_cm_data_file = open(os.path.join(path_256,'p-cm-data.txt'),'w');
                phi_cm_data_file.write(" sigma = %1.8f, spacing = %1.8f\n" % (2*math.pi/256, 4*math.pi/256));
                phi_cm_data_file.write(" centers = [\n")
                p_cm_data_file.write("p = [\n")
                for i in range(ncomps_data[l]):
                    phi_cm_data_file.write(" %1.8f, %1.8f, %1.8f\n" % (xcm_data[l][i][2], xcm_data[l][i][1], xcm_data[l][i][0]));
                    p_cm_data_file.write(" %1.8f\n" % 0.);
                phi_cm_data_file.write("];")
                p_cm_data_file.write("];")


            # ### generate slice images ###
            if args.generate_slices:
                BIN_COMPS = True;
                CHART_A = False;
                CHART_B = False;
                CHART_C = True;
                fsize = '4';
                c0recon_img  = nib.load(os.path.join(res_path, "c0Recon.nii.gz"));
                c0recon      = c0recon_img.get_fdata();
                c1recon_img  = nib.load(os.path.join(res_path, "cRecon.nii.gz"));
                c1recon      = c1recon_img.get_fdata();
                data_img     = nib.load(os.path.join(res_path, "data.nii.gz"));
                data         = data_img.get_fdata();

                # max/ min values for equal scaling
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
                            z = int(comps_sol[l][k][r][0]/(2*math.pi)*template.shape[2]);
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
                    levels_contour_c0  = [0.01, 0.1, 0.5, 0.7, 0.9];
                    levels_contour_c1  = [0.05, 0.1, 0.5, 0.7, 0.9, 1.0];
                    levels_contourf_c1 = [0.05, 0.1, 0.5, 0.7,];
                    cmap_cont = plt.cm.rainbow

                    # creating vis dirs
                    if not os.path.exists(vis_path):
                        os.makedirs(vis_path)
                    vis_path_slices = os.path.join(vis_path,"slices");
                    if not os.path.exists(vis_path_slices):
                        os.makedirs(vis_path_slices)
                    vis_path_more = os.path.join(vis_path, "more");
                    if CHART_A or CHART_B:
                        if not os.path.exists(vis_path_more):
                            os.makedirs(vis_path_more)
                    ###
                    ### CHART A #################################
                    if CHART_A:
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
                            ax[k+1].contour(slice,  levels=levels_contour_c0,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(levels_contour_c0)-1), linestyles=['-'] ,linewidths=[0.2], norm=norm);
                            # ax[r].contourf(slice, levels=levels_contour_c0,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(levels_contour_c0)-1), norm=norm);
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
                            ax[k+1].contour(slice,  levels=levels_contour_c0,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(levels_contour_c0)-1), linestyles=['-'] ,linewidths=[0.2], norm=norm);
                            # ax[r].contourf(slice, levels=levels_contour_c0,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(levels_contour_c0)-1), norm=norm);
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
                                    ax[r].contour(slice,  levels=levels_contour_c0,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(levels_contour_c0)-1), linestyles=['-'] ,linewidths=[0.2], norm=norm);
                                    # ax[r].contourf(slice, levels=levels_contour_c0,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(levels_contour_c0)-1), norm=norm);
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
                                    ax[r].contour(slice,  levels=levels_contour_c0,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(levels_contour_c0)-1), linestyles=['-'] ,linewidths=[0.2], norm=norm);
                                    # ax[r].contourf(slice, levels=levels_contour_c0,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(levels_contour_c0)-1), norm=norm);
                                    ax[r].set_title("$p_i$=%1.2e\nx=(%1.1f, %1.1f, %1.1f)\naxial slice %d" % (p_comp[l][k][r],comps_sol[l][k][r][2]/(2*math.pi)*template.shape[0], comps_sol[l][k][r][1]/(2*math.pi)*template.shape[1], comps_sol[l][k][r][0]/(2*math.pi)*template.shape[2],z), size=fsize, y=tpos);


                    ###
                    ### CHART B #################################
                    if CHART_B:
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
                            axis2[idx].contour(slice,  levels=levels_contour_c0,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(levels_contour_c0)-1), linestyles=['-'] ,linewidths=[0.2], norm=norm);
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
                            axis2[idx].contour(slice,  levels=levels_contour_c0,  cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(levels_contour_c0)-1), linestyles=['-'] ,linewidths=[0.2], norm=norm);
                            title = ''
                            for p in pi_labeled_dcomp[l][k+1]:
                                title += "p_i=" + "{0:1.2e}".format(p) + "\n"
                            axis2[idx].text(1.02, 0.5, title, fontsize=fsize, transform=axis2[idx].transAxes)
                            axis2[idx].set_title("x=(%1.1f, %1.1f, %1.1f)\naxial slice %d\n" % (wcm_labeled_dcomp[l][k+1][0]/(2*math.pi)*template.shape[0], wcm_labeled_dcomp[l][k+1][1]/(2*math.pi)*template.shape[1], wcm_labeled_dcomp[l][k+1][2]/(2*math.pi)*template.shape[2], z), size=fsize, y=tpos)



                    # ###
                    # ### CHART C #################################
                    if CHART_C:
                        nc = 0
                        for k in range(len(xcm_data[l])):
                            if relmass[l][k] > 1E-3:
                                nc+=1
                        fig3, axis3 = plt.subplots(nc, 3);
                        for ax in axis3.flatten(): # remove ticks and labels
                            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False);

                        fsize=6;
                        tpos = 1.0;
                        # cmap_d = plt.cm.Wistia
                        cmap_d = plt.cm.YlOrBr_r
                        # cmap_c1 = plt.cm.winter;
                        cmap_c1 = mpl.cm.get_cmap(plt.cm.rainbow, len(levels_contour_c1)-1);
                        # cmap_c0c = cmap=mpl.cm.get_cmap(plt.cm.rainbow, len(levels_contour_c0)-1);
                        cmap_c0c = mpl.cm.winter_r
                        # linewidths for contour
                        lwidths = [0.1, 0.1, 0.1, 0.1, 0.5, 0.1]

                        for k in range(len(xcm_data[l])):
                            if relmass[l][k] <= 1E-3:
                                continue; # don't display if rel mass of component is less then 10%
                            z = int(round(xcm_data[l][k][2]/(2*math.pi)*template.shape[2]))
                            y = int(round(xcm_data[l][k][1]/(2*math.pi)*template.shape[1]))
                            x = int(round(xcm_data[l][k][0]/(2*math.pi)*template.shape[0]))

                            fig_ax  = plt.figure(4);
                            fig_cor = plt.figure(5);
                            fig_sag = plt.figure(6);

                            # axial
                            bb = bratsID.split("_")[0] + "_" +  bratsID.split("_")[1] + "_" +  bratsID.split("_")[2] + '_1';
                            idx = tuple([k,0]) if isinstance(axis3[0], (np.ndarray, np.generic)) else 0
                            for aax, s, t in zip([axis3[idx], fig_ax.add_subplot(1, 1, 1)], [fsize, '14'], [bb+' $x_{cm}$ comp #'+str(k+1), bb]):
                                aax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False);
                                aax.imshow(template[:,:,z].T, cmap='gray', interpolation='none', origin='upper');
                                aax.imshow(thresh(data[:,:,z].T, cmap_d, d_thresh, v_max=hi, v_min=lo), interpolation='none', origin='upper', alpha=1);
                                slice, norm = cont(c1recon[:,:,z].T, cmap_c1, v_max=hi_c1, v_min=lo_c1);
                                aax.contourf(slice, levels=levels_contourf_c1,  cmap=cmap_c1, linestyles=['-'], norm=norm, alpha=0.6);
                                # slice, norm = cont(c1recon[:,:,z].T, cmap_c1, v_max=hi_c1, v_min=lo_c1, clip01=True);
                                aax.contour(slice,  levels=levels_contour_c1,  cmap=cmap_c1, linestyles=['-'] ,linewidths=lwidths, norm=norm, alpha=0.9);
                                # aax.imshow(thresh(c1recon[:,:,z].T, cmap_c1, thresh=c1_thresh, v_max=hi_c1, v_min=lo_c1), interpolation='none', alpha=0.5);
                                slice, norm = cont(c0recon[:,:,z].T, cmap_c0, v_max=hi_c0, v_min=lo_c0);
                                aax.contourf(slice, levels=levels_contour_c0, cmap=cmap_c0c, linestyles=['-'] , norm=norm, alpha=0.8);
                                aax.contour(slice,  levels=levels_contour_c0, cmap=cmap_c0c, linestyles=['-'] ,linewidths=[0.2], norm=norm, alpha=1);
                                aax.set_ylabel(t, fontsize=s)
                                aax.set_title("axial slice %d" %  z, size=s, y=tpos)

                            # sagittal
                            idx = tuple([k,1]) if isinstance(axis3[0], (np.ndarray, np.generic)) else 1
                            for aax, s, t in zip([axis3[idx], fig_cor.add_subplot(1, 1, 1)], [fsize, '14'], ['', bb]):
                                aax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False);
                                aax.imshow(template[x,:,:].T, cmap='gray', interpolation='none', origin='lower');
                                aax.imshow(thresh(data[x,:,:].T, cmap_d, d_thresh, v_max=hi, v_min=lo), interpolation='none', origin='lower', alpha=1);
                                slice, norm = cont(c1recon[x,:,:].T, cmap_c1, v_max=hi_c1, v_min=lo_c1);
                                aax.contourf(slice, levels=levels_contourf_c1,  cmap=cmap_c1, linestyles=['-'] , norm=norm, alpha=0.6);
                                # slice, norm = cont(c1recon[x,:,:].T, cmap_c1, v_max=hi_c1, v_min=lo_c1, clip01=True);
                                aax.contour(slice,  levels=levels_contour_c1,  cmap=cmap_c1, linestyles=['-'] ,linewidths=lwidths, norm=norm, alpha=0.9);
                                # aax.imshow(thresh(c1recon[:,:,z].T, cmap_c1, thresh=c1_thresh, v_max=hi_c1, v_min=lo_c1), interpolation='none', alpha=0.5);
                                slice, norm = cont(c0recon[x,:,:].T, cmap_c0, v_max=hi_c0, v_min=lo_c0);
                                aax.contourf(slice, levels=levels_contour_c0, cmap=cmap_c0c, linestyles=['-'] , norm=norm, alpha=0.8);
                                aax.contour(slice,  levels=levels_contour_c0, cmap=cmap_c0c, linestyles=['-'] ,linewidths=[0.2], norm=norm, alpha=1);
                                aax.set_ylabel(t, fontsize=s)
                                aax.set_title("coronal slice %d" %  x, size=s, y=tpos)

                            # coronal
                            idx = tuple([k,2]) if isinstance(axis3[0], (np.ndarray, np.generic)) else 2
                            for aax, s, t in zip([axis3[idx], fig_sag.add_subplot(1, 1, 1)], [fsize, '14'], ['', bb]):
                                aax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False);
                                aax.imshow(template[:,y,:].T, cmap='gray', interpolation='none', origin='lower');
                                aax.imshow(thresh(data[:,y,:].T, cmap_d, d_thresh, v_max=hi, v_min=lo), interpolation='none', origin='lower', alpha=1);
                                slice, norm = cont(c1recon[:,y,:].T, cmap_c1, v_max=hi_c1, v_min=lo_c1);
                                aax.contourf(slice, levels=levels_contourf_c1,  cmap=cmap_c1, linestyles=['-'], norm=norm, alpha=0.6);
                                # slice, norm = cont(c1recon[:,y,:].T, cmap_c1, v_max=hi_c1, v_min=lo_c1, clip01=True);
                                aax.contour(slice,  levels=levels_contour_c1,  cmap=cmap_c1, linestyles=['-'] ,linewidths=lwidths, norm=norm, alpha=0.9);
                                # aax.imshow(thresh(c1recon[:,:,z].T, cmap_c1, thresh=c1_thresh, v_max=hi_c1, v_min=lo_c1), interpolation='none', alpha=0.5);
                                slice, norm = cont(c0recon[:,y,:].T, cmap_c0, v_max=hi_c0, v_min=lo_c0);
                                aax.contourf(slice, levels=levels_contour_c0, cmap=cmap_c0c, linestyles=['-'], norm=norm, alpha=0.8);
                                aax.contour(slice,  levels=levels_contour_c0, cmap=cmap_c0c, linestyles=['-'] ,linewidths=[0.2], norm=norm, alpha=1);
                                aax.set_ylabel(t, fontsize=s)
                                aax.set_title("sagittal slice %d" %  y, size=s, y=tpos)

                            for ff,fn in zip([fig_ax,fig_cor,fig_sag], [bratsID+'_lvl-'+str(l)+'_cmp-'+str(k+1)+'_ax-'+str(z),bratsID+'_lvl-'+str(l)+'_cmp-'+str(k+1)+'_cor-'+str(x),bratsID+'_lvl-'+str(l)+'_cmp-'+str(k+1)+'_sag-'+str(y)]):
                                ff.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3);
                                ff.savefig(os.path.join(vis_path_slices, fn + '.pdf'), format='pdf', dpi=1200);
                                ff.clf()

                    # save fig to file
                    if CHART_A:
                        fname = 'chart-A-c1_nx'+str(l) if vis_c1 else 'chart-A-c0_nx'+str(l);
                        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.4);
                        fig.savefig(os.path.join(vis_path_more, fname + '.pdf'), format='pdf', dpi=1200);
                    if CHART_B:
                        fname = 'chart-B-c1_nx'+str(l) if vis_c1 else 'chart-B-c0_nx'+str(l);
                        fig2.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3);
                        fig2.savefig(os.path.join(vis_path_more, fname + '.pdf'), format='pdf', dpi=1200);
                    if CHART_C:
                        fname = 'chart-C-c1-c0_nx'+str(l);
                        fig3.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3);
                        fig3.savefig(os.path.join(vis_path, fname + '.pdf'), format='pdf', dpi=1200);



                # compute distance between two largest p_i in solution
                p_sum = pvec[l].sum();
                pvec_dict = {}
                coord_dict  = {}
                for kk in range(len(pvec[l])):
                    pvec_dict[kk] = pvec[l][kk]
                    coord_dict[kk] = np.array([phi[l][kk][0], phi[l][kk][1], phi[l][kk][2]]);
                p_sorted = sorted(pvec_dict.items(), key=lambda x: x[1], reverse=True);
                print("phi:", coord_dict)
                print("p_sorted:", p_sorted)
                FEATURES[l]["dist[max1(p) - max2(p)]"] = dist(coord_dict[p_sorted[0][0]], coord_dict[p_sorted[1][0]]);

                # visualize evolution of solution over levels
                mag = 200;
                sx = template_256.shape[0]/(2*math.pi);
                sy = template_256.shape[1]/(2*math.pi);
                sz = template_256.shape[2]/(2*math.pi);
                if l > 64:
                    for k in range(ncomps_sol[l]):
                        xs = []
                        ys = []
                        zs = []
                        for r in range(len(comps_sol[l][k])):
                            if p_comp[l][k][r] < 1e-8:
                                continue;
                            xs.append(round(comps_sol[l][k][r][2]*sx)) # x
                            ys.append(round(comps_sol[l][k][r][1]*sy)) # y
                            zs.append(round(comps_sol[l][k][r][0]*sz)) # z
                        ax_l2d[0].scatter(xs, ys, c=cc, marker=m, s=((np.array(p_comp[l][k])/p_sum * mag)))
                        ax_l2d[1].scatter(ys, zs, c=cc, marker=m, s=((np.array(p_comp[l][k])/p_sum * mag)))
                        ax_l2d[2].scatter(xs, zs, c=cc, marker=m, s=((np.array(p_comp[l][k])/p_sum * mag)))
                        Xx.extend(xs);
                        Yy.extend(ys);
                        Zz.extend(zs);
                if l == 256:
                    for k in range(ncomps_data[l]):
                            mm = 'x' if relmass[l][k] > 1E-3 else '1'
                            ss = 30 if relmass[l][k] > 1E-3 else 10
                            ccc = 'k' if relmass[l][k] > 1E-3 else 'k'
                            label = '$(l= %d)' % (math.log(level,2)-5);
                            ax_l2d[0].scatter(round(xcm_data[l][k][0]*sx), round(xcm_data[l][k][1]*sy), c=ccc, marker=mm, s=ss);
                            ax_l2d[1].scatter(round(xcm_data[l][k][1]*sy), round(xcm_data[l][k][2]*sz), c=ccc, marker=mm, s=ss);
                            ax_l2d[2].scatter(round(xcm_data[l][k][0]*sx), round(xcm_data[l][k][2]*sz), c=ccc, marker=mm, s=ss);
                            ax_l2d[0].scatter(round(wcm_labeled_dcomp[l][k+1][0]*sx), round(wcm_labeled_dcomp[l][k+1][1]*sy), c='navy', marker='1', s=20);
                            ax_l2d[1].scatter(round(wcm_labeled_dcomp[l][k+1][1]*sy), round(wcm_labeled_dcomp[l][k+1][2]*sz), c='navy', marker='1', s=20);
                            ax_l2d[2].scatter(round(wcm_labeled_dcomp[l][k+1][0]*sx), round(wcm_labeled_dcomp[l][k+1][2]*sz), c='navy', marker='1', s=20);
                            Xx.append(round(xcm_data[l][k][0]*sx));
                            Yy.append(round(xcm_data[l][k][1]*sy));
                            Zz.append(round(xcm_data[l][k][2]*sz));

        # close components.txt
        concomp_file.close();
        # save components_plot
        if args.generate_slices:
            pad=20
            Xx = np.asarray(Xx);
            Yy = np.asarray(Yy);
            Zz = np.asarray(Zz);
            max_range = np.amax(np.array([np.amax(Xx)-np.amin(Xx), np.amax(Yy)-np.amin(Yy), np.amax(Zz)-np.amin(Zz)]))
            mid_x = (np.amax(Xx)+np.amin(Xx)) * 0.5
            mid_y = (np.amax(Yy)+np.amin(Yy)) * 0.5
            mid_z = (np.amax(Zz)+np.amin(Zz)) * 0.5
            # axial
            z = int(round(xcm_data[256][0][2]/(2*math.pi)*template_256.shape[2]))
            # cmap='Spectral_r'
            ax_l2d[0].imshow(template_256[:,:,z].T, cmap='gray', interpolation='none', alpha=0.4, origin='upper');
            # 0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm
            # countours_ax = ax_l2d[0].contour(template_256[:,:,z].T, [0,1,2,4,5,6,7,8], cmap='jet', alpha=0.2, origin='lower', linewidth=0.05);
            # ax_l2d[0].clabel(countours_ax, [1,2,4,5,6,7,8],  inline=True, fontsize=5)
            ax_l2d[0].set_title("axial slice " + str(z), size='10');
            # sagittal
            x = int(round(xcm_data[256][0][0]/(2*math.pi)*template_256.shape[0]))
            ax_l2d[1].imshow(template_256[x,:,:].T, cmap='gray', interpolation='none', alpha=0.4, origin='upper');
            ax_l2d[1].set_title("sagittal slice " + str(x), size='10');
            # coronal
            y = int(round(xcm_data[256][0][1]/(2*math.pi)*template_256.shape[1]))
            ax_l2d[2].imshow(template_256[:,y,:].T, cmap='gray', interpolation='none', alpha=0.4, origin='lower');
            ax_l2d[2].set_title("coronal slice " + str(y), size='10');

            ax_l2d[0].set_aspect('equal', adjustable='box')
            ax_l2d[1].set_aspect('equal', adjustable='box')
            ax_l2d[2].set_aspect('equal', adjustable='box')

            fname = 'components_plot'
            ax_l2d[0].set_xlim(mid_x-max_range/2-pad, mid_x+max_range/2+pad)
            ax_l2d[0].set_ylim(mid_y-max_range/2-pad, mid_y+max_range/2+pad)
            ax_l2d[1].set_xlim(mid_y-max_range/2-pad, mid_y+max_range/2+pad)
            ax_l2d[1].set_ylim(mid_z-max_range/2-pad, mid_z+max_range/2+pad)
            ax_l2d[2].set_xlim(mid_x-max_range/2-pad, mid_x+max_range/2+pad)
            ax_l2d[2].set_ylim(mid_z-max_range/2-pad, mid_z+max_range/2+pad)
            ax_l2d[0].set_ylim(ax_l2d[0].get_ylim()[::-1])        # invert the axis
            sns.despine(offset=0, trim=True)
            fig_l2d.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.2, hspace=0.5);
            fig_l2d.savefig(os.path.join(vis_path, fname + '_zoom.pdf'), format='pdf', dpi=1200);

            ax_l2d[0].set_xlim([0,template_256.shape[0]])
            ax_l2d[0].set_ylim([0,template_256.shape[1]])
            ax_l2d[1].set_xlim([0,template_256.shape[1]])
            ax_l2d[1].set_ylim([0,template_256.shape[2]])
            ax_l2d[2].set_xlim([0,template_256.shape[0]])
            ax_l2d[2].set_ylim([0,template_256.shape[2]])
            ax_l2d[0].set_ylim(ax_l2d[0].get_ylim()[::-1])        # invert the axis
            sns.despine(offset=0, trim=True)
            fig_l2d.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, wspace=0.2, hspace=0.5);
            fig_l2d.savefig(os.path.join(vis_path, fname + '.pdf'), format='pdf', dpi=1200);

    if args.compute_tumor_stats:
        file_params = 'tumor_parameters-' + args.rdir + '.txt'
        infofile = open(os.path.join(args.input_path, file_params),'w');
        for l in levels:
            # paths
            lvl_prefix = os.path.join(os.path.join(args.input_path, 'tumor_inversion'), 'nx' + str(l));
            res_path   = os.path.join(lvl_prefix, args.rdir);
            init_path  = os.path.join(lvl_prefix, 'init');
            vis_path   = os.path.join(args.input_path, 'vis');
            template_fname = "template_nx"+str(l)+".nii.gz" if l < 256 else bratsID + '_seg_tu.nii.gz'

            # compute tumor stats (dice, frac, etc.)
            template_img = nib.load(os.path.join(path_256, template_fname));
            t1_recon_seg = nib.load(os.path.join(res_path,"seg1.nii.gz"));
            t1_recon_seg = t1_recon_seg.get_fdata();
            t0_recon_seg = nib.load(os.path.join(res_path, "seg0.nii.gz"));
            t0_recon_seg = t0_recon_seg.get_fdata();
            c0_recon  = nib.load(os.path.join(res_path, "c0Recon.nii.gz"));
            c0_recon  = c0_recon.get_fdata();
            c1_recon  = nib.load(os.path.join(res_path, "cRecon.nii.gz"));
            c1_recon  = c1_recon.get_fdata();
            c1_pred12 = None;
            c1_pred15 = None;
            if args.prediction:
                c1_pred12   = nib.load(os.path.join(res_path,"cPrediction_[t=1.2].nii.gz"));
                c1_pred12   = c1_pred12.get_fdata();
                c1_pred15   = nib.load(os.path.join(res_path, "cPrediction_[t=1.5].nii.gz"));
                c1_pred15   = c1_pred15.get_fdata();
            data = nib.load(os.path.join(res_path, "data.nii.gz"));
            data = data.get_fdata();
            if args.patient_labels is not None:
                for x in args.patient_labels.split(','):
                    patient_labels[int(x.split('=')[0])] = x.split('=')[1];
                patient_label_rev = {v:k for k,v in patient_labels.items()};


            component_mask = comps_data[l] if args.analyze_concomps else None;

            print("\n (4) computing DICE and tumor statistics\n");
            computeTumorStats(FEATURES[l], template_img, t1_recon_seg, t0_recon_seg, c1_recon, c0_recon, c1_pred12, c1_pred15, data, component_mask,  patient_label_rev, res_path);

            FEATURES[l]['n_comps[all]'] = len(xcm_data[l])
            for k in range(len(xcm_data[l])):
                if relmass[l][k] <= 1E-3:
                    break;
            FEATURES[l]['n_comps'] = k;

            # compute l2 error of c(0) in between levels || c(0)_256 - Ic(0)_64 || / ||c(0)_256 ||
            if 64 in levels and 128 in levels:
                print("\n (5) computing c(0) difference across levels (c(0) rescaled to 1)\n");
                c0_256 = fio.readNetCDF(os.path.join(path_256,"c0Recon.nc"));
                c0_256_norm = np.linalg.norm(c0_256.flatten(), 2);
                for ll in [64,128]:
                    lvl_prefix_ = os.path.join(os.path.join(args.input_path, 'tumor_inversion'), 'nx' + str(ll));
                    res_path_   = os.path.join(lvl_prefix_, args.rdir);
                    c0_coarse  = fio.readNetCDF(os.path.join(res_path_,"c0Recon.nc"));
                    # resample c(0) of coarser grid to 256 (3-rd order interp.)
                    c0_coarse  = imgtools.resizeImage(c0_coarse, tuple([256, 256, 256]), 1);
                    max_ic = np.amax(c0_coarse.flatten());
                    c0_coarse = c0_coarse * (1./max_ic);
                    diff = c0_256 - c0_coarse;
                    # compute l2 diff
                    l2errc0_over_levels[ll] = np.linalg.norm(diff.flatten(), 2) / c0_256_norm;

                    if l==256:
                        # convert to brats dims (interpolation order 3)
                        diff_tonii = imgtools.resizeImage(data, tuple(template_256.shape), 1);
                        # write file`
                        fio.writeNII(diff_tonii, os.path.join(path_256, "c0diff_nx"+str(ll)+"-nx256.nii.gz"), affine_256);
                        # fio.createNetCDF(os.path.join(path_256, "c0diff_nx"+str(ll)+"-nx256.nc"), [256,256,256], diff);
                        diff_nii = nib.load(os.path.join(path_256, "c0diff_nx"+str(ll)+"-nx256.nii.gz"));
                        diff_nii = diff_nii.get_fdata();

                        # for k in range(len(xcm_data[256])):
                        #     if relmass[256][k] <= 1E-2:
                        #         continue; # don't display if rel mass of component is less then 10%
                        #     z = int(round(xcm_data[256][k][2]/(2*math.pi)*template_256.shape[2]))
                        #     y = int(round(xcm_data[256][k][1]/(2*math.pi)*template_256.shape[1]))
                        #     x = int(round(xcm_data[256][k][0]/(2*math.pi)*template_256.shape[0]))
                        #
                        #     fnn = [bratsID+'_lvl-'+str(l)+'_c0diff-'+str(ll)+'-256_cmp-'+str(k+1)+'_ax-'+str(z),bratsID+'_lvl-'+str(l)+'_c0diff-'+str(ll)+'-256_cmp-'+str(k+1)+'_cor-'+str(x),bratsID+'_lvl-'+str(l)+'_c0diff-'+str(ll)+'-256_cmp-'+str(k+1)+'_sag-'+str(y)];
                        #     tn  = ["axial slice %d" %  z, "coronal slice %d" %  x, "sagittal slice %d" %  y];
                        #     tpn = [template_256[:,:,z].T, template_256[x,:,:].T, template_256[:,y,:].T]
                        #     din = [diff_nii[:,:,z].T, diff_nii[x,:,:].T, diff_nii[:,y,:].T]
                        #     for templ, c0, t, fn in zip(tpn, din, tn, fnn):
                        #         fig_c0 = plt.figure(10);
                        #         fig_c0.clf()
                        #         aax = fig_c0.add_subplot(1, 1, 1)
                        #         aax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False);
                        #         aax.imshow(templ, cmap='gray', interpolation='none', origin='upper', alpha=0.4);
                        #         im = aax.imshow(c0, cmap=mpl.cm.bwr, norm=mpl.colors.Normalize(vmin=np.amin(c0.flatten()), vmax=np.amax(c0.flatten())), interpolation='none', origin='upper', alpha=0.8);
                        #         # slice, norm = cont(c0, mpl.cm.bwr, v_max=1, v_min=-1, clip01=False);
                        #         # aax.contourf(slice,  levels=[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1],  cmap= mpl.cm.bwr, linestyles=['-'], norm=norm, alpha=0.8);
                        #         fig_c0.colorbar(im, ax=aax, extend='max');
                        #         aax.set_ylabel("$\|c(0) - I c(0)_{"+str(ll)+"} \|_2$", fontsize='14')
                        #         aax.set_title(t, size='14', y=tpos)
                        #         fig_c0.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3);
                        #         fig_c0.savefig(os.path.join(vis_path_slices, fn + '.pdf'), format='pdf', dpi=1200);
                        #         fig_c0.clf()


            print("\n (6) writing parameter file",file_params)
            text = "## level " + str(l) + " ##\n";
            if os.path.exists(os.path.join(res_path,'info.dat')):
                with open(os.path.join(res_path,'info.dat'), 'r') as f:
                    lines = f.readlines();
                    if len(lines) > 0:
                        param = lines[0].split(" ");
                        values = lines[1].split(" ");
                        text += "\nestimated diffusion coefficient       = " + "{0:.2e}".format(float(values[1]));
                        text += "\nestimated reaction coefficient        = " + "{0:.2e}".format(float(values[0]));
                        text += "\nreconstruction error (l2)             = " + "{0:.2e}".format(float(values[2]));
                    else:
                        print("  Error: output file info.dat is empty for for tumor inversion of patient ");
            else:
                print("tumor solver did not generate a parametter output file")

            text += "\n == Tumor Statistics ==";
            if args.compute_dice_healthy:
                text += "\ndice outside tumor core (csf,gm,wm)   = (" + "{0:.2f}".format(FEATURES[l]['csf_dice']*100) + "," + "{0:.2f}".format(FEATURES[l]['gm_dice']*100) + "," + "{0:.2f}".format(FEATURES[l]['wm_dice']*100) + ")";
            if args.compute_tumor_stats:
                text += "\nhealthy tissue dice (csf,gm,wm)       = (" + "{0:.2f}".format(FEATURES[l]['csf_dice']*100) + "," + "{0:.2f}".format(FEATURES[l]['gm_dice']*100) + "," + "{0:.2f}".format(FEATURES[l]['wm_dice']*100) + ")";
                text += "\ndice tumor (max): (wt,tc,nec)         = (" + "{0:.2f}".format(FEATURES[l]['wt_dice']*100)  + "," + "{0:.2f}".format(FEATURES[l]['tc_dice']*100) + "," + "{0:.2f}".format(FEATURES[l]['nec_dice']*100)  + ")";
                text += "\ndice tumor (> x): (tc9,tc8,nec1)      = (" + "{0:.2f}".format(FEATURES[l]['tc_c(1)[x>0.9]_dice']*100)  + "," + "{0:.2f}".format(FEATURES[l]['tc_c(1)[x>0.8]_dice']*100) + "," + "{0:.2f}".format(FEATURES[l]['nec_c(1)[x=1.0]_dice']*100)  + ")";
                text += "\ndice ed (x<ed<0.02): (ed8,ed7,ed61)   = (" + "{0:.2f}".format(FEATURES[l]['ed_c(1)[0.02<x<0.8]_dice']*100)  + "," + "{0:.2f}".format(FEATURES[l]['ed_c(1)[0.02<x<0.7]_dice']*100) + "," + "{0:.2f}".format(FEATURES[l]['ed_c(1)[0.02<x<0.6]_dice']*100)  + ")";
                text += "\nc(ed,1) (max,min)                     = (" + "{0:1.2f}".format(FEATURES[l]['max{c(1)|_ED}']) + "," + "{0:1.2f}".format(FEATURES[l]['min{c(1)|_ED}']) + ")";
                text += "\nc(tc,1) (max,min)                     = (" + "{0:1.2f}".format(FEATURES[l]['max{c(1)|_TC}']) + "," + "{0:1.2f}".format(FEATURES[l]['min{c(1)|_ED}']) + ")";
                text += "\nstats #tu/#brain  (wt,ed,en,nec,tc)   = (" + "{0:1.3e}".format(FEATURES[l]['vol(WT)_r']) + "," + "{0:1.3e}".format(FEATURES[l]['vol(ED)_r']) + "," + "{0:1.3e}".format(FEATURES[l]['vol(EN)_r']) + "," + "{0:1.3e}".format(FEATURES[l]['vol(NEC)_r']) + "," + "{0:1.3e}".format(FEATURES[l]['vol(TC)_r']) + ")";
                text += "\nstats #tu/#brain  (rec_tc,pred_ct,c0) = (" + "{0:1.3e}".format(FEATURES[l]['vol(c(1)[x>0.9])_r']) + "," + "{0:1.3e}".format(FEATURES[l]['vol(c(1.5)[x>0.9])_r']) + ","  + "{0:1.3e}".format(FEATURES[l]['vol(c(0))_r']) + ")";
                text += "\nstats int_B c(1) dx                   = " + "{0:1.3e}".format(FEATURES[l]['int{c(1)}']);
                text += "\nstats int_ED c(1) dx                  = " + "{0:1.3e}".format(FEATURES[l]['int{c(1)|_ED}']);
                text += "\nstats int_TC c(1) dx                  = " + "{0:1.3e}".format(FEATURES[l]['int{c(1)|_TC}']);
                text += "\nstats int_B/WT c(1) dx                = " + "{0:1.3e}".format(FEATURES[l]['int{c(1)|_B-WT}']);
                text += "\nstats int c(0)   / int c(1)           = " + "{0:1.3e}".format(FEATURES[l]['int{c(0)}/int{c(1)}']);
                text += "\nstats int c(1.5) / int c(1)           = " + "{0:1.3e}".format(FEATURES[l]['int{c(1.5)}/int{c(1)}']);
                text += "\nstats int c(1.5) / int c(1.2)         = " + "{0:1.3e}".format(FEATURES[l]['int{c(1.5)}/int{c(1.2)}']);
                text += "\nstats int c(1.5) / int d              = " + "{0:1.3e}".format(FEATURES[l]['int{c(1.5)}/int{TC}']);
                text += "\nl2err_c(1) (smooth)(virg,obs)         = (" + "{0:1.3e}".format(FEATURES[l]['l2[c(1)-TC]']) + "," + "{0:1.3e}".format(FEATURES[l]['l2[Oc(1)-TC]'])  + ")";
                text += "\nl2err_c(1)         (virg,obs)         = (" + "{0:1.3e}".format(FEATURES[l]['l2[c(1)-TC]_nonsmooth']) + "," + "{0:1.3e}".format(FEATURES[l]['l2[Oc(1)-TC]_nonsmooth'])  + ")";
                if args.analyze_concomps:
                    if 64 in levels and 128 in levels:
                        text += "\nl2ec(1) scaled,TC (l1,l2,l3)          = (" + "{0:1.3e}".format(l2err_TC[64]/l2normref_TC[64]) + "," + "{0:1.3e}".format(l2err_TC[128]/l2normref_TC[128]) + "," + "{0:1.3e}".format(l2err_TC[256]/l2normref_TC[256])  + ")";
                        FEATURES[l]['l2[c(1)|_TC-TC,scaled]_r_l64'] = l2err_TC[64]/l2normref_TC[64];
                        FEATURES[l]['l2[c(1)|_TC-TC,scaled]_r_l128'] = l2err_TC[128]/l2normref_TC[128];
                        FEATURES[l]['l2[c(1)|_TC-TC,scaled]_r_l256'] = l2err_TC[256]/l2normref_TC[256];
                    else:
                        text += "\nl2ec(1) scaled,TC (l3)                = (" + "{0:1.3e}".format(l2err_TC[256]/l2normref_TC[256])  + ")";
                        FEATURES[l]['l2[c(1)|_TC,scaled]_l256'] = l2err_TC[256]/l2normref_TC[256]
                    llabels = ["l1", "l2", "l3"] if args.gridcont else ["l3"];
                    for lvl, lll in zip(levels, llabels):
                        t1 = "\nl2ec(1) scaled,relC (" + lll + ";#1,..,#n)     = (";
                        t2 = "\nl2ec(1) scaled,relD (" + lll + ";#1,..,#n)     = (";
                        nc = len(l2err_percomp_TC[lvl]);
                        for i in range(nc):
                            t1 += "{0:1.3e}".format(l2err_percomp_TC[lvl][i]/l2normref_percomp[lvl][i]);
                            t2 += "{0:1.3e}".format(l2err_percomp_TC[lvl][i]/l2normref_TC[lvl]);
                            sep = "," if i < nc-1 else ")";
                            t1 += sep;
                            t2 += sep;
                        text += t1;
                        text += t2;
                if 64 in levels and 128 in levels:
                    text += "\nl2err_c(0) |c_h-c_H|_r (l1,l2)        = (" + "{0:1.3e}".format(l2errc0_over_levels[64]) + "," + "{0:1.3e}".format(l2errc0_over_levels[128])  + ")";
                    FEATURES[l]['l2[c(0)-Pc(0)]_128_64'] = l2errc0_over_levels[64];
                    FEATURES[l]['l2[c(0)-Pc(0)]_256_128'] = l2errc0_over_levels[128];
                text += " \n\n\n";
            infofile.write(text);
        infofile.close();

        # write featuee dict to file
        (pd.DataFrame.from_dict(data=FEATURES[64], orient='index').to_csv(os.path.join(args.input_path, 'features_64.csv'), header=False, sep=';', float_format='%.16g'));
        (pd.DataFrame.from_dict(data=FEATURES[128], orient='index').to_csv(os.path.join(args.input_path, 'features_128.csv'), header=False, sep=';', float_format='%.16g'));
        (pd.DataFrame.from_dict(data=FEATURES[256], orient='index').to_csv(os.path.join(args.input_path, 'features_256.csv'), header=False, sep=';', float_format='%.16g'));
