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
    args = parser.parse_args();

    gen_atlas = False;
    read_atlas = True;
    gen_images = True;

    BAD_BRAINS_CSF = []
    BAD_BRAINS_OUT = []

    dir = args.dir;
    FILTER = ['Brats18_CBICA_AQJ_1', 'Brats18_TCIA08_242_1', 'Brats18_CBICA_AZD_1', 'Brats18_TCIA02_374_1', 'Brats18_CBICA_ANI_1', 'Brats18_CBICA_AUR_1']
    survival_data = pd.read_csv(os.path.join(basedir,"survival_data.csv"), header = 0, error_bad_lines=True, skipinitialspace=True);

    # load atlas
    atlas = nib.load(os.path.join(dir,"jakob_segmented_with_cere_lps_240x240x155_in_brats_hdr.nii.gz"))
    affine = atlas.affine;
    ashape = atlas.shape;
    atlas = atlas.get_fdata();

    # genrate glioma atlas
    if gen_atlas:
        glioma_c0_atlas       = np.zeros_like(atlas);
        glioma_c0_atlas_short = np.zeros_like(atlas);
        glioma_c0_atlas_mid   = np.zeros_like(atlas);
        glioma_c0_atlas_long  = np.zeros_like(atlas);
        glioma_c0_atlas_na    = np.zeros_like(atlas);
        glioma_c1_atlas       = np.zeros_like(atlas);
        glioma_c1_atlas_short = np.zeros_like(atlas);
        glioma_c1_atlas_mid   = np.zeros_like(atlas);
        glioma_c1_atlas_long  = np.zeros_like(atlas);
        glioma_c1_atlas_na    = np.zeros_like(atlas);
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
                print(bcolors.OKBLUE + " reading ", P  + bcolors.ENDC)
                if BID in survival_data['BraTS18ID'].values:
                    survival_row = survival_data.loc[survival_data['BraTS18ID'] == BID];
                    srvl_class = getSurvivalClass(float(survival_row.iloc[0]['Survival']));

                patient_path = os.path.join(os.path.join(dir, P), "tumor_inversion/nx256/obs-1.0/");
                # load patient c(0) nii image
                c0_in_aspace = nib.load(os.path.join(patient_path, "c0Recon_256x256x256_aff2jakob_in_Aspace_240x240x155.nii.gz")).get_fdata();
                c1_in_aspace = nib.load(os.path.join(patient_path, "cRecon_256x256x256_aff2jakob_in_Aspace_240x240x155.nii.gz")).get_fdata();

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

            atlas_c1_sml = glioma_c1_atlas_short + glioma_c1_atlas_mid + glioma_c1_atlas_long;
            atlas_c0_sml = glioma_c0_atlas_short + glioma_c0_atlas_mid + glioma_c0_atlas_long;
            # compute max
            atlas_c0_sml   = np.amax(atlas_c0_sml.flatten())
            max_c0_atlas   = np.amax(glioma_c0_atlas.flatten())
            max_c0_atlas_s = np.amax(glioma_c0_atlas_short.flatten())
            max_c0_atlas_m = np.amax(glioma_c0_atlas_mid.flatten())
            max_c0_atlas_l = np.amax(glioma_c0_atlas_long.flatten())
            max_c0_atlas_n = np.amax(glioma_c0_atlas_na.flatten())
            atlas_c1_sml   = np.amax(atlas_c1_sml.flatten())
            max_c1_atlas   = np.amax(glioma_c1_atlas.flatten())
            max_c1_atlas_s = np.amax(glioma_c1_atlas_short.flatten())
            max_c1_atlas_m = np.amax(glioma_c1_atlas_mid.flatten())
            max_c1_atlas_l = np.amax(glioma_c1_atlas_long.flatten())
            max_c1_atlas_n = np.amax(glioma_c1_atlas_na.flatten())
            print("max value of glioma c(0) atlas: {}".format(max_c0_atlas));
            print("max value of glioma c(1) atlas: {}".format(max_c1_atlas));
            print("max value of [srvl] c(1) atlas: {}".format(atlas_c1_sml));

            print("write nii.gz files")
            # c(0) atlas
            fio.writeNII(np.abs(glioma_c0_atlas       / max_c0_atlas),   os.path.join(dir,'brats_c0_atlas_plain.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_short / max_c0_atlas_s), os.path.join(dir,'brats[srvl]_c0_atlas_short_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_short / atlas_c0_sml),   os.path.join(dir,'brats[srvl]_c0_atlas_short_normalized-across.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_mid   / max_c0_atlas_m), os.path.join(dir,'brats[srvl]_c0_atlas_mid_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_mid   / atlas_c0_sml),   os.path.join(dir,'brats[srvl]_c0_atlas_mid_normalized-across.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_long  / max_c0_atlas_l), os.path.join(dir,'brats[srvl]_c0_atlas_long_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_long  / atlas_c0_sml),   os.path.join(dir,'brats[srvl]_c0_atlas_long_normalized-across.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_na    / max_c0_atlas_n), os.path.join(dir,'brats[srvl]_c0_atlas_na_normalized-ind.nii.gz'), affine);
            # c(1) atlas
            fio.writeNII(np.abs(glioma_c1_atlas       / max_c1_atlas),   os.path.join(dir,'brats_c1_atlas_plain.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_short / max_c1_atlas_s), os.path.join(dir,'brats[srvl]_c1_atlas_short_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_short / atlas_c1_sml),   os.path.join(dir,'brats[srvl]_c1_atlas_short_normalized-across.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_mid   / max_c1_atlas_m), os.path.join(dir,'brats[srvl]_c1_atlas_mid_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_mid   / atlas_c1_sml),   os.path.join(dir,'brats[srvl]_c1_atlas_mid_normalized-across.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_long  / max_c1_atlas_l), os.path.join(dir,'brats[srvl]_c1_atlas_long_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_long  / atlas_c1_sml),   os.path.join(dir,'brats[srvl]_c1_atlas_long_normalized-across.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_na    / max_c1_atlas_n), os.path.join(dir,'brats[srvl]_c1_atlas_na_normalized-ind.nii.gz'), affine);



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

        glioma_c1_atlas           = nib.load(os.path.join(dir, "brats_c1_atlas_plain.nii.gz")).get_fdata();
        glioma_c1_atlas_short_ind = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_short_normalized-ind.nii.gz")).get_fdata();
        glioma_c1_atlas_mid_ind   = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_mid_normalized-ind.nii.gz")).get_fdata();
        glioma_c1_atlas_long_ind  = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_long_normalized-ind.nii.gz")).get_fdata();
        glioma_c1_atlas_na_ind    = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_na_normalized-ind.nii.gz")).get_fdata();
        glioma_c1_atlas_short     = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_short_normalized-across.nii.gz")).get_fdata();
        glioma_c1_atlas_mid       = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_mid_normalized-across.nii.gz")).get_fdata();
        glioma_c1_atlas_long      = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_long_normalized-across.nii.gz")).get_fdata();

    if gen_images:
        atlas_sml = glioma_c1_atlas_short + glioma_c1_atlas_mid + glioma_c1_atlas_long;
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
            row = np.unravel_index(i, axis_c1.shape,'C')[0]

        j = 0;
        i = 0;
        m = 0;
        ax_slice = 24 - axs_inc;
        for k in range(nslices):
            ax_slice += axs_inc;
            axis_c0[i,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
            axis_c0[i,j].imshow(thresh(glioma_c0_atlas_short_ind[:,:,ax_slice].T, cmap=cmap_c0_s, threshold=t_c0, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=1);
            axis_c0[i,j].imshow(thresh(glioma_c0_atlas_mid_ind[:,:,ax_slice].T,   cmap=cmap_c0_m, threshold=t_c0, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=1);
            axis_c0[i,j].imshow(thresh(glioma_c0_atlas_long_ind[:,:,ax_slice].T,  cmap=cmap_c0_l, threshold=t_c0, v_max=1, v_min=0), interpolation='none', aspect='equal', alpha=1);
            axis_c0[i,j].set_title("axial slice %d" %  ax_slice , size='5', y=1.0)

            vals = np.ones((ashape[0], ashape[1], 4))
            max_max = max(max(np.amax(glioma_c1_atlas_short.flatten()), np.amax(glioma_c1_atlas_mid.flatten())), np.amax(glioma_c1_atlas_long.flatten()));
            vals[..., 0] = glioma_c1_atlas_short[:,:,ax_slice].T / max_max # red
            vals[..., 1] = glioma_c1_atlas_mid[:,:,ax_slice].T   / max_max # green
            vals[..., 2] = glioma_c1_atlas_long[:,:,ax_slice].T  / max_max # blue
            vals[..., 3] = Normalize(0, t_c1, clip=True)(atlas_sml[:,:,ax_slice].T) # alpha
            cmap_c1_combined = vals; #colors.ListedColormap(vals)
            axis_c1_rgb[i,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
            axis_c1_rgb[i,j].imshow(vals, interpolation='none', aspect='equal', alpha=0.8);
            axis_c1_rgb[i,j].set_title("axial slice %d" %  ax_slice , size='5', y=1.0)

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

            i = i + 1 if k % ncols == 0 and k > 0 else i
            j = j + 1 if k % ncols != 0 else 0
            m = m + 4 if k % ncols == 0 and k > 0 else m
        vpath_c0 = os.path.join(dir,'vis-atlas/glioma-atlas-c0');
        vpath_c1 = os.path.join(dir,'vis-atlas/glioma-atlas-c1');
        fname_c0 = "glioma_atlas_c0_ax_slices.pdf"
        fname_c1 = "glioma_atlas_c1_ax_slices.pdf"
        fname_c1_rgb = "glioma_atlas_c1-rgb_ax_slices.pdf"

        for vp, fn, fig in zip([vpath_c0, vpath_c1, vpath_c1], [fname_c0, fname_c1, fname_c1_rgb], [fig_c0, fig_c1, fig_c1_rgb]):
            if not os.path.exists(vp):
                os.makedirs(vp);
            if os.path.isfile(os.path.join(vp, fn)):
                os.remove(os.path.join(vp, fn))
            fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.4);
            fig.tight_layout();
            fig.savefig(os.path.join(vp, fn), format='pdf', dpi=1200);
