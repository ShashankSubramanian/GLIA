import matplotlib as mpl
mpl.use('Agg')
import os, sys
from os import listdir
import numpy as np
import pandas as pd
import imageTools as imgtools
import nibabel as nib
import nibabel.processing
import ntpath
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import file_io as fio
from pprint import pprint

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def getSurvivalClass(x):
    m = x/30.
    if m < 10:
        return 0;
    elif m < 15:
        return 1;
    else:
        return 2;

def getSizeClass(x, max):
    m = x/max;
    if m < 0.33:
        return 0;
    elif m < 0.66:
        return 1;
    else:
        return 2;

def thresh(slice, cmap, threshold=0.3, v_max=None, v_min=None):
    slice_clipped = np.clip(slice, 0, 1);
    alphas = Normalize(0, threshold, clip=True)(slice_clipped)
    max = np.amax(slice_clipped) if v_max == None else v_max;
    min = np.amin(slice_clipped) if v_min == None else v_min;
    slice_normalized = Normalize(min, max)(slice_clipped);
    cmap_ = cmap(slice_normalized)
    cmap_[..., -1] = alphas
    return cmap_;

###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    pd.options.display.float_format = '{1.2e}%'.format
    # parse arguments
    basedir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='process BRATS results')
    parser.add_argument ('-x',    '--dir',   type = str, help = 'path to the results folder');
    parser.add_argument ('-file', '--f',     type = str, help = 'path to the csv brats file');
    args = parser.parse_args();

    gen_atlas = False;
    read_atlas = True;
    gen_images = True;

    BAD_BRAINS_CSF = []
    BAD_BRAINS_OUT = []

    dir = args.dir;
    file = args.f;
    FILTER = ['Brats18_CBICA_AQJ_1', 'Brats18_TCIA08_242_1', 'Brats18_CBICA_AZD_1', 'Brats18_TCIA02_374_1', 'Brats18_CBICA_ANI_1', 'Brats18_CBICA_AUR_1']
    survival_data = pd.read_csv(os.path.join(basedir,"survival_data.csv"), header = 0, error_bad_lines=True, skipinitialspace=True);

    brats_data = None;
    brats_survival = None;
    if file is not None:
        brats_data = pd.read_csv(os.path.join(basedir,file), header = 0, error_bad_lines=True, skipinitialspace=True)
        print("read brats simulation data of length %d" % len(brats_data))
        nbshort = len(brats_data.loc[brats_data['survival_class'] ==  0])
        nbmid   = len(brats_data.loc[brats_data['survival_class'] ==  1])
        nblong  = len(brats_data.loc[brats_data['survival_class'] ==  2])
        sum     = nbshort + nbmid + nblong
        print("work dataset of length %d/%d. short: %d, mid: %d, long: %d" % (len(brats_data),len(survival_data),nbshort,nbmid,nblong))
        # filter survival data
        brats_survival = brats_data.copy();
        brats_survival['age'] = brats_survival['age'].astype('float')
        brats_survival = brats_survival.loc[brats_survival['age'] >  0]

    # tumor size statistics
    # brats_survival['vol(TC+ED)_r'] = brats_survival['vol(TC)_r'] + brats_survival['vol(ED)_r']
    brats_survival['vol(TC+0.5*ED)_r'] = brats_survival['vol(TC)_r'] + 0.5 * brats_survival['vol(ED)_r']
    max_TC_r = np.amax(brats_survival['vol(TC)_r'].values);
    min_TC_r = np.amin(brats_survival['vol(TC)_r'].values);
    max_TCED_r = np.amax(brats_survival['vol(TC+0.5*ED)_r'].values);
    min_TCED_r = np.amin(brats_survival['vol(TC+0.5*ED)_r'].values);
    # print("max vol(TC)_r: {}, min vol(TC)_r: {}".format(np.amax(brats_survival['vol(TC)_r'].values), np.amin(brats_survival['vol(TC)_r'].values)))
    # print("max vol(ED)_r: {}, min vol(ED)_r: {}".format(np.amax(brats_survival['vol(ED)_r'].values), np.amin(brats_survival['vol(ED)_r'].values)))
    # print("max vol(TC+ED)_r: {}, min vol(TC+ED)_r: {}".format(np.amax(brats_survival['vol(TC+ED)_r'].values), np.amin(brats_survival['vol(TC+ED)_r'].values)))
    # print("max vol(TC+0.5*ED)_r: {}, min vol(TC+0.5*ED)_r: {}".format(np.amax(brats_survival['vol(TC+0.5*ED)_r'].values), np.amin(brats_survival['vol(TC+0.5*ED)_r'].values)))
    # brats_survival.hist(['vol(TC)_r']);
    # brats_survival.hist(['vol(ED)_r']);
    # brats_survival.hist(['vol(TC+ED)_r']);
    # brats_survival.hist(['vol(TC+0.5*ED)_r']);

    # load atlas
    atlas = nib.load(os.path.join(dir,"jakob_segmented_with_cere_lps_240x240x155_in_brats_hdr.nii.gz"))
    affine = atlas.affine;
    ashape = atlas.shape;
    atlas = atlas.get_fdata();

    # genrate glioma atlas
    if gen_atlas:
        glioma_c0_atlas        = np.zeros_like(atlas);
        glioma_c0_atlas_short  = np.zeros_like(atlas);
        glioma_c0_atlas_mid    = np.zeros_like(atlas);
        glioma_c0_atlas_long   = np.zeros_like(atlas);
        glioma_c0_atlas_na     = np.zeros_like(atlas);
        glioma_c1_atlas        = np.zeros_like(atlas);
        glioma_c1_atlas_short  = np.zeros_like(atlas);
        glioma_c1_atlas_mid    = np.zeros_like(atlas);
        glioma_c1_atlas_long   = np.zeros_like(atlas);
        glioma_c1_atlas_na     = np.zeros_like(atlas);

        glioma_c0_atlas_small  = np.zeros_like(atlas);
        glioma_c0_atlas_medium = np.zeros_like(atlas);
        glioma_c0_atlas_large  = np.zeros_like(atlas)
        glioma_c1_atlas_small  = np.zeros_like(atlas);
        glioma_c1_atlas_medium = np.zeros_like(atlas);
        glioma_c1_atlas_large  = np.zeros_like(atlas);

        atlas_mask = (atlas <= 0).astype(int);
        csf_mask   = (atlas == 7).astype(int);
        if dir is not None:
            PATIENTS = os.listdir(dir);
            for P in PATIENTS:
                BID = str(P)
                if not P.startswith('Brats'):
                    continue;
                if BID in FILTER:
                    continue;
                srvl_class = -1;
                srvl_class = -1;
                print(bcolors.OKBLUE + " reading ", P  + bcolors.ENDC)
                if BID in survival_data['BraTS18ID'].values:
                    survival_row = survival_data.loc[survival_data['BraTS18ID'] == BID];
                    data_row = brats_survival.loc[brats_survival['BID'] == BID];
                    srvl_class = getSurvivalClass(float(survival_row.iloc[0]['Survival']));
                    # size_class = getSizeClass(float(data_row.iloc[0]['vol(TC+0.5*ED)_r']), max_TCED_r)
                    size_class = getSizeClass(float(data_row.iloc[0]['vol(TC)_r']), max_TC_r)

                patient_path = os.path.join(os.path.join(dir, P), "tumor_inversion/nx256/obs-1.0/");
                # load patient c(0) nii image
                c0_in_aspace = nib.load(os.path.join(patient_path, "c0Recon_256x256x256_aff2jakob_in_Aspace_240x240x155.nii.gz")).get_fdata();
                c1_in_aspace = nib.load(os.path.join(patient_path, "cRecon_256x256x256_aff2jakob_in_Aspace_240x240x155.nii.gz")).get_fdata();

                # filter brain IDs which c(0) lieas outside or in ventricles of atlas
                ref = np.linalg.norm(c0_in_aspace);
                if np.linalg.norm(np.multiply(c0_in_aspace, atlas_mask)) > 0.5*ref:
                    BAD_BRAINS_OUT.append(BID);
                if np.linalg.norm(np.multiply(c0_in_aspace, csf_mask)) > 0.5*ref:
                    BAD_BRAINS_CSF.append(BID);

                # superimpose in atlas
                glioma_c0_atlas += c0_in_aspace;
                glioma_c1_atlas += c1_in_aspace;
                # superimpose on srvl atlas
                if srvl_class == 0:
                    glioma_c0_atlas_short += c0_in_aspace;
                    glioma_c1_atlas_short += c1_in_aspace;
                elif srvl_class == 1:
                    glioma_c0_atlas_mid += c0_in_aspace;
                    glioma_c1_atlas_mid += c1_in_aspace;
                elif srvl_class == 2:
                    glioma_c0_atlas_long += c0_in_aspace;
                    glioma_c1_atlas_long += c1_in_aspace;
                else:
                    glioma_c0_atlas_na += c0_in_aspace;
                    glioma_c1_atlas_na += c1_in_aspace;

                # superimpose on size atlas
                if size_class == 0:
                    glioma_c0_atlas_small += c0_in_aspace;
                    glioma_c1_atlas_small += c1_in_aspace;
                elif size_class == 1:
                    glioma_c0_atlas_medium += c0_in_aspace;
                    glioma_c1_atlas_medium += c1_in_aspace;
                elif size_class == 2:
                    glioma_c0_atlas_large += c0_in_aspace;
                    glioma_c1_atlas_large += c1_in_aspace;

            atlas_c1_sml = glioma_c1_atlas_short + glioma_c1_atlas_mid + glioma_c1_atlas_long;
            atlas_c0_sml = glioma_c0_atlas_short + glioma_c0_atlas_mid + glioma_c0_atlas_long;
            atlas_c0_sml   = np.amax(atlas_c0_sml.flatten())
            atlas_c1_sml   = np.amax(atlas_c1_sml.flatten())

            print("write nii.gz files")
            # c(0) atlas
            fio.writeNII(np.abs(glioma_c0_atlas        / np.amax(glioma_c0_atlas.flatten())),   os.path.join(dir,'brats_c0_atlas_plain.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_short  / np.amax(glioma_c0_atlas_short.flatten())), os.path.join(dir,'brats[srvl]_c0_atlas_short_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_long   / np.amax(glioma_c0_atlas_long.flatten())), os.path.join(dir,'brats[srvl]_c0_atlas_long_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_mid    / np.amax(glioma_c0_atlas_mid.flatten())), os.path.join(dir,'brats[srvl]_c0_atlas_mid_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_short  / atlas_c0_sml),   os.path.join(dir,'brats[srvl]_c0_atlas_short_normalized-across.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_mid    / atlas_c0_sml),   os.path.join(dir,'brats[srvl]_c0_atlas_mid_normalized-across.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_long   / atlas_c0_sml),   os.path.join(dir,'brats[srvl]_c0_atlas_long_normalized-across.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_na     / np.amax(glioma_c0_atlas_na.flatten())), os.path.join(dir,'brats[srvl]_c0_atlas_na_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_small  / np.amax(glioma_c0_atlas_small.flatten())), os.path.join(dir,'brats[srvl]_c0_atlas_small[TC+ED]_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_medium / np.amax(glioma_c0_atlas_medium.flatten())), os.path.join(dir,'brats[srvl]_c0_atlas_medium[TC+ED]_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_large  / np.amax(glioma_c0_atlas_large.flatten())), os.path.join(dir,'brats[srvl]_c0_atlas_large[TC+ED]_normalized-ind.nii.gz'), affine);
            # c(1) atlas
            fio.writeNII(np.abs(glioma_c1_atlas        / np.amax(glioma_c1_atlas.flatten())),   os.path.join(dir,'brats_c1_atlas_plain.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_short  / np.amax(glioma_c1_atlas_short.flatten())), os.path.join(dir,'brats[srvl]_c1_atlas_short_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_mid    / np.amax(glioma_c1_atlas_mid.flatten())), os.path.join(dir,'brats[srvl]_c1_atlas_mid_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_long   / np.amax(glioma_c1_atlas_long.flatten())), os.path.join(dir,'brats[srvl]_c1_atlas_long_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_short  / atlas_c1_sml),   os.path.join(dir,'brats[srvl]_c1_atlas_short_normalized-across.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_mid    / atlas_c1_sml),   os.path.join(dir,'brats[srvl]_c1_atlas_mid_normalized-across.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_long   / atlas_c1_sml),   os.path.join(dir,'brats[srvl]_c1_atlas_long_normalized-across.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_na     / np.amax(glioma_c1_atlas_na.flatten())), os.path.join(dir,'brats[srvl]_c1_atlas_na_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_small  / np.amax(glioma_c1_atlas_small.flatten())), os.path.join(dir,'brats[srvl]_c1_atlas_small[TC+ED]_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_medium / np.amax(glioma_c1_atlas_medium.flatten())), os.path.join(dir,'brats[srvl]_c1_atlas_medium[TC+ED]_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_large  / np.amax(glioma_c1_atlas_large.flatten())), os.path.join(dir,'brats[srvl]_c1_atlas_large[TC+ED]_normalized-ind.nii.gz'), affine);



    print("Bad brains outside:", BAD_BRAINS_OUT)
    print("Bad brains in CSF:", BAD_BRAINS_CSF)
    if read_atlas:
        print("reading glioma atlasses")
        glioma_c0_atlas           = nib.load(os.path.join(dir, "brats_c0_atlas_plain.nii.gz")).get_fdata();
        glioma_c0_atlas_short_ind = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_short_normalized-ind.nii.gz")).get_fdata();
        glioma_c0_atlas_mid_ind   = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_mid_normalized-ind.nii.gz")).get_fdata();
        glioma_c0_atlas_long_ind  = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_long_normalized-ind.nii.gz")).get_fdata();
        glioma_c0_atlas_na_ind    = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_na_normalized-ind.nii.gz")).get_fdata();
        glioma_c0_atlas_short     = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_short_normalized-across.nii.gz")).get_fdata();
        glioma_c0_atlas_mid       = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_mid_normalized-across.nii.gz")).get_fdata();
        glioma_c0_atlas_long      = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_long_normalized-across.nii.gz")).get_fdata();
        glioma_c0_atlas_small     = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_small[TC+ED]_normalized-ind.nii.gz")).get_fdata();
        glioma_c0_atlas_medium    = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_medium[TC+ED]_normalized-ind.nii.gz")).get_fdata();
        glioma_c0_atlas_large     = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_large[TC+ED]_normalized-ind.nii.gz")).get_fdata();

        glioma_c1_atlas           = nib.load(os.path.join(dir, "brats_c1_atlas_plain.nii.gz")).get_fdata();
        glioma_c1_atlas_short_ind = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_short_normalized-ind.nii.gz")).get_fdata();
        glioma_c1_atlas_mid_ind   = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_mid_normalized-ind.nii.gz")).get_fdata();
        glioma_c1_atlas_long_ind  = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_long_normalized-ind.nii.gz")).get_fdata();
        glioma_c1_atlas_na_ind    = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_na_normalized-ind.nii.gz")).get_fdata();
        glioma_c1_atlas_short     = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_short_normalized-across.nii.gz")).get_fdata();
        glioma_c1_atlas_mid       = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_mid_normalized-across.nii.gz")).get_fdata();
        glioma_c1_atlas_long      = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_long_normalized-across.nii.gz")).get_fdata();
        glioma_c1_atlas_small     = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_small[TC+ED]_normalized-ind.nii.gz")).get_fdata();
        glioma_c1_atlas_medium    = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_medium[TC+ED]_normalized-ind.nii.gz")).get_fdata();
        glioma_c1_atlas_large     = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_large[TC+ED]_normalized-ind.nii.gz")).get_fdata();

    if gen_images:
        atlas_sml      = glioma_c1_atlas_short + glioma_c1_atlas_mid + glioma_c1_atlas_long;
        atlas_sml_size = glioma_c1_atlas_small + glioma_c1_atlas_medium + glioma_c1_atlas_large;
        print("creating slices")
        clrs = ['black','red', 'green', 'yellow', 'blue', 'cyan', 'orange', 'purple'];
        tick_param_kwargs = {"axis":"both", "which":"both", "bottom":False, "left":False, "labelbottom":False, "labelleft":False}
        imshow_kwargs_template = {"cmap":"gray", "aspect":"equal"}
        imshow_kwargs_c1 = {"cmap":plt.cm.rainbow, "aspect":"equal"}
        imshow_kwargs_r = {"cmap":"gray_r", "aspect":"equal"}

        cmap_c0_s = plt.cm.Reds;
        cmap_c0_m = plt.cm.Greens;
        cmap_c0_l = plt.cm.Blues;
        cmap_c1_s = plt.cm.jet;
        cmap_c1_m = plt.cm.jet;
        cmap_c1_l = plt.cm.jet;
        t_c0 = 0.05;
        t_c1 = 0.3;

        nslices  = 45;
        ax_slice = 0;
        axs_inc  = int((114-24)/float(nslices))
        nrows    = 5
        ncols    = 9
        # -------- color by survival ------- #
        # c0 glioma atlas
        fig_c0, axis_c0 = plt.subplots(nrows, ncols)#), figsize=(14,20));
        for a,i in zip(axis_c0.flatten(),np.arange(len(axis_c0.flatten()))):
            a.tick_params(**tick_param_kwargs)
            a.set_yticklabels([])
            a.set_xticklabels([])
            a.set_frame_on(False)
        # c1 glioma atlas (combined, RGB)
        fig_c1_rgb, axis_c1_rgb = plt.subplots(nrows, ncols)#), figsize=(14,20));
        for a,i in zip(axis_c1_rgb.flatten(),np.arange(len(axis_c1_rgb.flatten()))):
            a.tick_params(**tick_param_kwargs)
            a.set_yticklabels([])
            a.set_xticklabels([])
            a.set_frame_on(False)
        # c1 glioma atlas (separate)
        fig_c1, axis_c1 = plt.subplots(nrows*4, ncols, figsize=(14,34))
        for a,i in zip(axis_c1.flatten(),np.arange(len(axis_c1.flatten()))):
            a.tick_params(**tick_param_kwargs)
            a.set_yticklabels([])
            a.set_xticklabels([])
            a.set_frame_on(False)
        # -------- color by size -------- #
        # c0 glioma atlas SIZE
        fig_c0_size, axis_c0_size = plt.subplots(nrows, ncols)#), figsize=(14,20));
        for a,i in zip(axis_c0_size.flatten(),np.arange(len(axis_c0_size.flatten()))):
            a.tick_params(**tick_param_kwargs)
            a.set_yticklabels([])
            a.set_xticklabels([])
            a.set_frame_on(False)
        # c1 glioma atlas (comined, RGB) SIZE
        fig_c1_rgb_size, axis_c1_rgb_size = plt.subplots(nrows, ncols)#), figsize=(14,20));
        for a,i in zip(axis_c1_rgb_size.flatten(),np.arange(len(axis_c1_rgb_size.flatten()))):
            a.tick_params(**tick_param_kwargs)
            a.set_yticklabels([])
            a.set_xticklabels([])
            a.set_frame_on(False)
        # c1 glioma atlas (separate) SIZE
        fig_c1_size, axis_c1_size = plt.subplots(nrows*4, ncols, figsize=(14,34))
        for a,i in zip(axis_c1_size.flatten(),np.arange(len(axis_c1_size.flatten()))):
            a.tick_params(**tick_param_kwargs)
            a.set_yticklabels([])
            a.set_xticklabels([])
            a.set_frame_on(False)

        j = 0;
        i = 0;
        m = 0;
        ax_slice = 24 - axs_inc;
        for k in range(nslices):
            ax_slice += axs_inc;

            # -------- color by survival ------- #
            # # -- c(0) -- #
            axis_c0[i,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
            axis_c0[i,j].imshow(thresh(glioma_c0_atlas_short_ind[:,:,ax_slice].T, cmap=cmap_c0_s, threshold=t_c0, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=1);
            axis_c0[i,j].imshow(thresh(glioma_c0_atlas_mid_ind[:,:,ax_slice].T,   cmap=cmap_c0_m, threshold=t_c0, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=1);
            axis_c0[i,j].imshow(thresh(glioma_c0_atlas_long_ind[:,:,ax_slice].T,  cmap=cmap_c0_l, threshold=t_c0, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=1);
            axis_c0[i,j].set_title("axial slice %d" %  ax_slice , size='5', y=1.0)
            # # -- c(1) RGB -- #
            vals = np.ones((ashape[0], ashape[1], 4))
            max_max = max(max(np.amax(glioma_c1_atlas_short.flatten()), np.amax(glioma_c1_atlas_mid.flatten())), np.amax(glioma_c1_atlas_long.flatten()));
            vals[..., 0] = glioma_c1_atlas_short[:,:,ax_slice].T / max_max # red
            vals[..., 1] = glioma_c1_atlas_mid[:,:,ax_slice].T   / max_max # green
            vals[..., 2] = glioma_c1_atlas_long[:,:,ax_slice].T  / max_max # blue
            vals[..., 3] = Normalize(0, t_c1, clip=True)(atlas_sml[:,:,ax_slice].T) # alpha
            cmap_c1_combined = vals;
            axis_c1_rgb[i,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
            axis_c1_rgb[i,j].imshow(vals, interpolation='none', aspect='equal', alpha=0.8);
            axis_c1_rgb[i,j].set_title("axial slice %d" %  ax_slice , size='5', y=1.0)
            # # -- c(1) -- #
            axis_c1[m+0,j].add_patch(plt.Rectangle((-8,1.2),8, 0.01,facecolor='silver', clip_on=False, linewidth = 0))
            axis_c1[m+0,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
            axis_c1[m+1,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
            axis_c1[m+2,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
            axis_c1[m+3,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
            axis_c1[m+0,j].imshow(thresh(glioma_c1_atlas_short[:,:,ax_slice].T, cmap=cmap_c1_s, threshold=t_c1, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=0.8);
            axis_c1[m+1,j].imshow(thresh(glioma_c1_atlas_mid[:,:,ax_slice].T,   cmap=cmap_c1_m, threshold=t_c1, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=0.8);
            axis_c1[m+2,j].imshow(thresh(glioma_c1_atlas_long[:,:,ax_slice].T,  cmap=cmap_c1_l, threshold=t_c1, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=0.8);
            axis_c1[m+3,j].imshow(vals, interpolation='none', aspect='equal', alpha=0.8);
            axis_c1[m+0,j].set_title("axial slice %d" %  ax_slice , size='5', y=1.0)
            axis_c1[m+0,j].set_ylabel("short survivor", size='5')
            axis_c1[m+1,j].set_ylabel("mid survivor", size='5')
            axis_c1[m+2,j].set_ylabel("long survivor", size='5')
            axis_c1[m+3,j].set_ylabel("combined (r=short,g=mid,b=long)", size='5')


            # -------- color by size ------- #
            # # -- c(0) -- #
            axis_c0_size[i,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
            axis_c0_size[i,j].imshow(thresh(glioma_c0_atlas_small[:,:,ax_slice].T,  cmap=cmap_c0_s, threshold=t_c0, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=1);
            axis_c0_size[i,j].imshow(thresh(glioma_c0_atlas_medium[:,:,ax_slice].T, cmap=cmap_c0_m, threshold=t_c0, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=1);
            axis_c0_size[i,j].imshow(thresh(glioma_c0_atlas_large[:,:,ax_slice].T,  cmap=cmap_c0_l, threshold=t_c0, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=1);
            axis_c0_size[i,j].set_title("axial slice %d" %  ax_slice , size='5', y=1.0)
            # # -- c(1) RGB -- #
            vals = np.ones((ashape[0], ashape[1], 4))
            max_max = max(max(np.amax(glioma_c1_atlas_small.flatten()), np.amax(glioma_c1_atlas_medium.flatten())), np.amax(glioma_c1_atlas_large.flatten()));
            vals[..., 0] = glioma_c1_atlas_small[:,:,ax_slice].T # / max_max # red
            vals[..., 1] = glioma_c1_atlas_medium[:,:,ax_slice].T   #/ max_max # green
            vals[..., 2] = glioma_c1_atlas_large[:,:,ax_slice].T  #/ max_max # blue
            vals[..., 3] = Normalize(0, t_c1, clip=True)(atlas_sml_size[:,:,ax_slice].T) # alpha
            cmap_c1_combined = vals;
            axis_c1_rgb_size[i,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
            axis_c1_rgb_size[i,j].imshow(vals, interpolation='none', aspect='equal', alpha=0.8);
            axis_c1_rgb_size[i,j].set_title("axial slice %d" %  ax_slice , size='5', y=1.0)
            # # -- c(1) -- #
            axis_c1_size[m+0,j].add_patch(plt.Rectangle((-8,1.2),8, 0.01,facecolor='silver', clip_on=False, linewidth = 0))
            axis_c1_size[m+0,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
            axis_c1_size[m+1,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
            axis_c1_size[m+2,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
            axis_c1_size[m+3,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
            axis_c1_size[m+0,j].imshow(thresh(glioma_c1_atlas_small[:,:,ax_slice].T,  cmap=cmap_c1_s, threshold=t_c1, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=0.8);
            axis_c1_size[m+1,j].imshow(thresh(glioma_c1_atlas_medium[:,:,ax_slice].T, cmap=cmap_c1_m, threshold=t_c1, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=0.8);
            axis_c1_size[m+2,j].imshow(thresh(glioma_c1_atlas_large[:,:,ax_slice].T,  cmap=cmap_c1_l, threshold=t_c1, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=0.8);
            axis_c1_size[m+3,j].imshow(vals, interpolation='none', aspect='equal', alpha=0.8);
            axis_c1_size[m+0,j].set_title("axial slice %d" %  ax_slice , size='5', y=1.0)
            axis_c1_size[m+0,j].set_ylabel("small (TC+0.5*ED)", size='5')
            axis_c1_size[m+1,j].set_ylabel("medium (TC+0.5*ED)", size='5')
            axis_c1_size[m+2,j].set_ylabel("large (TC+0.5*ED)", size='5')
            axis_c1_size[m+3,j].set_ylabel("blend (r=small,g=medium,b=large)", size='5')

            i = i + 1 if k % ncols == 0 and k > 0 else i
            j = j + 1 if k % ncols != 0 else 0
            m = m + 4 if k % ncols == 0 and k > 0 else m
        vpath_c0 = os.path.join(dir,'vis-atlas/glioma-atlas-c0');
        vpath_c1 = os.path.join(dir,'vis-atlas/glioma-atlas-c1');
        fname_c0          = "brats[srvl]_gbm_atlas[colored-by-srvl]_c0_ind-normalized_ax-slice.pdf"
        fname_c1          = "brats[srvl]_gbm_atlas[colored-by-srvl]_c1_across-normalized_ax-slice.pdf"
        fname_c1_rgb      = "brats[srvl]_gbm_atlas[colored-by-srvl]_c1_rgb_across-normalized_ax-slice.pdf"
        fname_c0_size     = "brats[srvl]_gbm_atlas[colored-by-size-tc]_c0_ind-normalized_ax-slice.pdf"
        fname_c1_size     = "brats[srvl]_gbm_atlas[colored-by-size-tc]_c1_ind-normalized_ax-slice.pdf"
        fname_c1_rgb_size = "brats[srvl]_gbm_atlas[colored-by-size-tc]_c1_rgb_ax-slice.pdf"

        for vp, fn, fig in zip( [vpath_c0, vpath_c1, vpath_c1, vpath_c0, vpath_c1, vpath_c1],
                                [fname_c0, fname_c1, fname_c1_rgb, fname_c0_size, fname_c1_size, fname_c1_rgb_size],
                                [fig_c0, fig_c1, fig_c1_rgb, fig_c0_size, fig_c1_size, fig_c1_rgb_size]):
            if not os.path.exists(vp):
                os.makedirs(vp);
            if os.path.isfile(os.path.join(vp, fn)):
                os.remove(os.path.join(vp, fn))
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.4);
            fig.tight_layout();
            fig.savefig(os.path.join(vp, fn), format='pdf', dpi=1200);
