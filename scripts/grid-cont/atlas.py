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
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import file_io as fio
from pprint import pprint
from tabulate import tabulate

# ### LABEL LIST FUNCTIONAL ATLAS ###
# -----------------------------------
LABELS_ATLAS = {
    0  : 'background',
    1  : 'superior_frontal_gyrus',
    2  : 'middle_frontal_gyrus',
    3  : 'inferior_frontal_gyrus',
    4  : 'precentral_gyrus',
    5  : 'middle_orbitofrontal_gyrus',
    6  : 'lateral_orbitofrontal_gyrus',
    7  : 'gyrus_rectus',
    8  : 'postcentral_gyrus',
    9  : 'superior_parietal_gyrus',
    10 : 'supramarginal_gyrus',
    11 : 'angular_gyrus',
    12 : 'precuneus',
    13 : 'superior_occipital_gyrus',
    14 : 'middle_occipital_gyrus',
    15 : 'inferior_occipital_gyrus',
    16 : 'cuneus',
    17 : 'superior_temporal_gyrus',
    18 : 'middle_temporal_gyrus',
    19 : 'inferior_temporal_gyrus',
    20 : 'parahippocampal_gyrus',
    21 : 'lingual_gyrus',
    22 : 'fusiform_gyrus',
    23 : 'insular_cortex',
    24 : 'cingulate_gyrus',
    25 : 'caudate',
    26 : 'putamen',
    27 : 'hippocampus',
    28 : 'cerebellum',
    29 : 'brainstem',
    30 : 'unlabelled white matter',
    31 : 'ventricles'}
# -----------------------------------

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

    gen_atlas  = False;
    read_atlas = False;
    gen_images = False;
    comp_stats = False;

    comp_stats  = True;
    gen_images  = True;

    BAD_BRAINS_CSF = []
    BAD_BRAINS_OUT = []

    dir = args.dir;
    file = args.f;
    FILTER = ['Brats18_CBICA_AQJ_1', 'Brats18_TCIA08_242_1', 'Brats18_CBICA_AZD_1', 'Brats18_TCIA02_374_1', 'Brats18_CBICA_ANI_1', 'Brats18_CBICA_AUR_1']
    survival_data = pd.read_csv(os.path.join(basedir,"survival_data.csv"), header = 0, error_bad_lines=True, skipinitialspace=True);

    brats_data = None;
    brats_survival = None;
    # ### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #
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
    atlas = nib.load(os.path.join(dir,"jakob_segmented_with_cere_lps_240x240x155_in_brats_hdr.nii.gz"));
    affine = atlas.affine;
    ashape = atlas.shape;
    atlas = atlas.get_fdata();
    atlas_func = nib.load(os.path.join(dir,"lpba40_combined_LR_256x256x256_aff2jakob_in_jakob_space_240x240x155.nii.gz")).get_fdata();

    # genrate glioma atlas
    # ### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #
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
            max_atlas_c0_sml   = np.amax(atlas_c0_sml.flatten())
            max_atlas_c1_sml   = np.amax(atlas_c1_sml.flatten())

            print("write nii.gz files")
            # c(0) atlas
            fio.writeNII(np.abs(glioma_c0_atlas        / np.amax(glioma_c0_atlas.flatten())),   os.path.join(dir,'brats_c0_atlas_plain.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_short  / np.amax(glioma_c0_atlas_short.flatten())), os.path.join(dir,'brats[srvl]_c0_atlas_short_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_long   / np.amax(glioma_c0_atlas_long.flatten())), os.path.join(dir,'brats[srvl]_c0_atlas_long_normalized-ind.nii.gz'),   affine);
            fio.writeNII(np.abs(glioma_c0_atlas_mid    / np.amax(glioma_c0_atlas_mid.flatten())), os.path.join(dir,'brats[srvl]_c0_atlas_mid_normalized-ind.nii.gz'),     affine);
            fio.writeNII(np.abs(glioma_c0_atlas_short  / max_atlas_c0_sml),   os.path.join(dir,'brats[srvl]_c0_atlas_short_normalized-across.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_mid    / max_atlas_c0_sml),   os.path.join(dir,'brats[srvl]_c0_atlas_mid_normalized-across.nii.gz'),   affine);
            fio.writeNII(np.abs(glioma_c0_atlas_long   / max_atlas_c0_sml),   os.path.join(dir,'brats[srvl]_c0_atlas_long_normalized-across.nii.gz'),  affine);
            fio.writeNII(np.abs(glioma_c0_atlas_short                    ),   os.path.join(dir,'brats[srvl]_c0_atlas_short_abs.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_mid                      ),   os.path.join(dir,'brats[srvl]_c0_atlas_mid_abs.nii.gz'),   affine);
            fio.writeNII(np.abs(glioma_c0_atlas_long                     ),   os.path.join(dir,'brats[srvl]_c0_atlas_long_abs.nii.gz'),  affine);
            fio.writeNII(np.abs(glioma_c0_atlas_na     / np.amax(glioma_c0_atlas_na.flatten())), os.path.join(dir,'brats[srvl]_c0_atlas_na_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_small  / np.amax(glioma_c0_atlas_small.flatten())), os.path.join(dir,'brats[srvl]_c0_atlas_small[TC+ED]_normalized-ind.nii.gz'),   affine);
            fio.writeNII(np.abs(glioma_c0_atlas_medium / np.amax(glioma_c0_atlas_medium.flatten())), os.path.join(dir,'brats[srvl]_c0_atlas_medium[TC+ED]_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c0_atlas_large  / np.amax(glioma_c0_atlas_large.flatten())), os.path.join(dir,'brats[srvl]_c0_atlas_large[TC+ED]_normalized-ind.nii.gz'),   affine);
            # c(1) atlas
            fio.writeNII(np.abs(glioma_c1_atlas        / np.amax(glioma_c1_atlas.flatten())),   os.path.join(dir,'brats_c1_atlas_plain.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_short  / np.amax(glioma_c1_atlas_short.flatten())), os.path.join(dir,'brats[srvl]_c1_atlas_short_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_mid    / np.amax(glioma_c1_atlas_mid.flatten())), os.path.join(dir,'brats[srvl]_c1_atlas_mid_normalized-ind.nii.gz'),     affine);
            fio.writeNII(np.abs(glioma_c1_atlas_long   / np.amax(glioma_c1_atlas_long.flatten())), os.path.join(dir,'brats[srvl]_c1_atlas_long_normalized-ind.nii.gz'),   affine);
            fio.writeNII(np.abs(glioma_c1_atlas_short  / max_atlas_c1_sml),   os.path.join(dir,'brats[srvl]_c1_atlas_short_normalized-across.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_mid    / max_atlas_c1_sml),   os.path.join(dir,'brats[srvl]_c1_atlas_mid_normalized-across.nii.gz'),   affine);
            fio.writeNII(np.abs(glioma_c1_atlas_long   / max_atlas_c1_sml),   os.path.join(dir,'brats[srvl]_c1_atlas_long_normalized-across.nii.gz'),  affine);
            fio.writeNII(np.abs(glioma_c1_atlas_short                    ),   os.path.join(dir,'brats[srvl]_c1_atlas_short_abs.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_mid                      ),   os.path.join(dir,'brats[srvl]_c1_atlas_mid_abs.nii.gz'),   affine);
            fio.writeNII(np.abs(glioma_c1_atlas_long                     ),   os.path.join(dir,'brats[srvl]_c1_atlas_long_abs.nii.gz'),  affine);
            fio.writeNII(np.abs(glioma_c1_atlas_na     / np.amax(glioma_c1_atlas_na.flatten())), os.path.join(dir,'brats[srvl]_c1_atlas_na_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_small  / np.amax(glioma_c1_atlas_small.flatten())), os.path.join(dir,'brats[srvl]_c1_atlas_small[TC+ED]_normalized-ind.nii.gz'),   affine);
            fio.writeNII(np.abs(glioma_c1_atlas_medium / np.amax(glioma_c1_atlas_medium.flatten())), os.path.join(dir,'brats[srvl]_c1_atlas_medium[TC+ED]_normalized-ind.nii.gz'), affine);
            fio.writeNII(np.abs(glioma_c1_atlas_large  / np.amax(glioma_c1_atlas_large.flatten())), os.path.join(dir,'brats[srvl]_c1_atlas_large[TC+ED]_normalized-ind.nii.gz'),   affine);



    # print("Bad brains outside:", BAD_BRAINS_OUT)
    # print("Bad brains in CSF:", BAD_BRAINS_CSF)
    # ### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #
    if read_atlas:
        print("reading glioma atlasses");
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

        atlas_sml      = glioma_c1_atlas_short + glioma_c1_atlas_mid + glioma_c1_atlas_long;
        atlas_sml_size = glioma_c1_atlas_small + glioma_c1_atlas_medium + glioma_c1_atlas_large;

    if comp_stats:
        print("reading glioma atlasses");
        glioma_c1_atlas_short_abs = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_short_abs.nii.gz")).get_fdata();
        glioma_c1_atlas_mid_abs   = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_mid_abs.nii.gz")).get_fdata();
        glioma_c1_atlas_long_abs  = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_long_abs.nii.gz")).get_fdata();
        glioma_c0_atlas_short_abs = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_short_abs.nii.gz")).get_fdata();
        glioma_c0_atlas_mid_abs   = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_mid_abs.nii.gz")).get_fdata();
        glioma_c0_atlas_long_abs  = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_long_abs.nii.gz")).get_fdata();
        print("computing stats")
        atlas_sml_c0_abs  = glioma_c0_atlas_short_abs + glioma_c0_atlas_mid_abs + glioma_c0_atlas_long_abs;
        atlas_sml_c1_abs  = glioma_c1_atlas_short_abs + glioma_c1_atlas_mid_abs + glioma_c1_atlas_long_abs;

        ref_c0 = np.linalg.norm(atlas_sml_c0_abs.flatten(), 2)
        ref_c1 = np.linalg.norm(atlas_sml_c1_abs.flatten(), 2)
        percentage_c0_short_in_label = {}
        percentage_c0_mid_in_label   = {}
        percentage_c0_long_in_label  = {}
        percentage_c1_short_in_label = {}
        percentage_c1_mid_in_label   = {}
        percentage_c1_long_in_label  = {}
        percentage_c0_short_in_label_per_l = {}
        percentage_c0_mid_in_label_per_l   = {}
        percentage_c0_long_in_label_per_l  = {}
        percentage_c1_short_in_label_per_l = {}
        percentage_c1_mid_in_label_per_l   = {}
        percentage_c1_long_in_label_per_l  = {}
        percentage_c0_frequency_label = {}
        percentage_c1_frequency_label = {}
        size_label = {}
        for label, descr in LABELS_ATLAS.items():
            mask = (atlas_func == label).astype(int)

            percentage_c0_short_in_label[label] = np.linalg.norm(np.multiply(glioma_c0_atlas_short_abs, mask).flatten(), 2) / ref_c0 * 100;
            percentage_c0_mid_in_label[label]   = np.linalg.norm(np.multiply(glioma_c0_atlas_mid_abs, mask).flatten(), 2)   / ref_c0 * 100;
            percentage_c0_long_in_label[label]  = np.linalg.norm(np.multiply(glioma_c0_atlas_long_abs, mask).flatten(), 2)  / ref_c0 * 100;
            percentage_c1_short_in_label[label] = np.linalg.norm(np.multiply(glioma_c1_atlas_short_abs, mask).flatten(), 2) / ref_c1 * 100;
            percentage_c1_mid_in_label[label]   = np.linalg.norm(np.multiply(glioma_c1_atlas_mid_abs, mask).flatten(), 2)   / ref_c1 * 100;
            percentage_c1_long_in_label[label]  = np.linalg.norm(np.multiply(glioma_c1_atlas_long_abs, mask).flatten(), 2)  / ref_c1 * 100;

            ref_c0_label = np.linalg.norm(np.multiply(atlas_sml_c0_abs, mask).flatten(), 2);
            ref_c1_label = np.linalg.norm(np.multiply(atlas_sml_c1_abs, mask).flatten(), 2);

            percentage_c0_frequency_label[label]  = ref_c0_label  / ref_c0 * 100;
            percentage_c1_frequency_label[label]  = ref_c1_label  / ref_c1 * 100;

            percentage_c0_short_in_label_per_l[label] = np.linalg.norm(np.multiply(glioma_c0_atlas_short_abs, mask).flatten(), 2) / ref_c0_label * 100;
            percentage_c0_mid_in_label_per_l[label]   = np.linalg.norm(np.multiply(glioma_c0_atlas_mid_abs, mask).flatten(), 2)   / ref_c0_label * 100;
            percentage_c0_long_in_label_per_l[label]  = np.linalg.norm(np.multiply(glioma_c0_atlas_long_abs, mask).flatten(), 2)  / ref_c0_label * 100;
            percentage_c1_short_in_label_per_l[label] = np.linalg.norm(np.multiply(glioma_c1_atlas_short_abs, mask).flatten(), 2) / ref_c1_label * 100;
            percentage_c1_mid_in_label_per_l[label]   = np.linalg.norm(np.multiply(glioma_c1_atlas_mid_abs, mask).flatten(), 2)   / ref_c1_label * 100;
            percentage_c1_long_in_label_per_l[label]  = np.linalg.norm(np.multiply(glioma_c1_atlas_long_abs, mask).flatten(), 2)  / ref_c1_label * 100;

            size_label[label] = np.linalg.norm(mask.flatten(),0) / np.linalg.norm(((atlas_func > 0).astype(int).flatten()),0) * 100;


        table_c0 = [['label', 'short', 'mid', 'long', 'all', 'size']]
        table_c1 = [['label', 'short', 'mid', 'long', 'all', 'size']]
        table_c0_per_l = [['label', 'short', 'mid', 'long']]
        table_c1_per_l = [['label', 'short', 'mid', 'long']]
        for label, descr in LABELS_ATLAS.items():
            table_c0.append([descr, round(percentage_c0_short_in_label[label],2), round(percentage_c0_mid_in_label[label],2), round(percentage_c0_long_in_label[label],2), round(percentage_c0_frequency_label[label],2), round(size_label[label], 2)])
            table_c1.append([descr, round(percentage_c1_short_in_label[label],2), round(percentage_c1_mid_in_label[label],2), round(percentage_c1_long_in_label[label],2), round(percentage_c1_frequency_label[label],2), round(size_label[label], 2)])
            table_c0_per_l.append([descr, round(percentage_c0_short_in_label_per_l[label],2), round(percentage_c0_mid_in_label_per_l[label],2), round(percentage_c0_long_in_label_per_l[label],2)])
            table_c1_per_l.append([descr, round(percentage_c1_short_in_label_per_l[label],2), round(percentage_c1_mid_in_label_per_l[label],2), round(percentage_c1_long_in_label_per_l[label],2)])

        print()
        print("Percentage c(0) in brain regions divided by survival (rel. to c(0) in brain):")
        print(tabulate(table_c0, tablefmt="psql"))
        print()
        print("Percentage c(0) in brain regions divided by survival (rel. to c(0) in label):")
        print(tabulate(table_c0_per_l, tablefmt="psql"))
        print()
        print("Percentage c(1) in brain regions divided by survival (rel. to c(0) in brain):")
        print(tabulate(table_c1, tablefmt="psql"))
        print()
        print("Percentage c(1) in brain regions divided by survival (rel. to c(0) in label):")
        print(tabulate(table_c1_per_l, tablefmt="psql"))


        atlas_freq_c0 = np.zeros_like(atlas_func);
        atlas_freq_c1 = np.zeros_like(atlas_func);
        atlas_freq_c0_short = np.zeros_like(atlas_func);
        atlas_freq_c0_mid   = np.zeros_like(atlas_func);
        atlas_freq_c0_long  = np.zeros_like(atlas_func);
        atlas_freq_c1_short = np.zeros_like(atlas_func);
        atlas_freq_c1_mid   = np.zeros_like(atlas_func);
        atlas_freq_c1_long  = np.zeros_like(atlas_func);
        percentage_c0_frequency_label
        freq_c0_sorted       = sorted(percentage_c0_frequency_label.items(), key=lambda x: x[1], reverse=True);
        freq_c1_sorted       = sorted(percentage_c1_frequency_label.items(), key=lambda x: x[1], reverse=True);
        freq_c0_short_sorted = sorted(percentage_c0_short_in_label.items(), key=lambda x: x[1], reverse=True);
        freq_c0_mid_sorted   = sorted(percentage_c0_mid_in_label.items(), key=lambda x: x[1], reverse=True);
        freq_c0_long_sorted  = sorted(percentage_c0_long_in_label.items(), key=lambda x: x[1], reverse=True);
        freq_c1_short_sorted = sorted(percentage_c1_short_in_label.items(), key=lambda x: x[1], reverse=True);
        freq_c1_mid_sorted   = sorted(percentage_c1_mid_in_label.items(), key=lambda x: x[1], reverse=True);
        freq_c1_long_sorted  = sorted(percentage_c1_long_in_label.items(), key=lambda x: x[1], reverse=True);

        print()
        print("max c(0) label frequencies (s+m+l)")
        for i in range(len(freq_c0_sorted)):
            label = freq_c0_sorted[i][0]
            freq = percentage_c0_frequency_label[label]
            mask = (atlas_func == label).astype(int)
            print("label: {:30} freq: {}".format(LABELS_ATLAS[label], freq))
            atlas_freq_c0 += mask * freq;
            if i >= 8:
                 break;
        print()
        print("max c(0) label frequencies (short)")
        for i in range(len(freq_c0_short_sorted)):
            label = freq_c0_short_sorted[i][0]
            freq = percentage_c0_short_in_label[label]
            mask = (atlas_func == label).astype(int)
            print("label: {:30} freq: {}".format(LABELS_ATLAS[label], freq))
            atlas_freq_c0_short += mask * freq;
            if i >= 6:
                 break;
        print()
        print("max c(0) label frequencies (mid)")
        for i in range(len(freq_c0_mid_sorted)):
            label = freq_c0_mid_sorted[i][0]
            freq = percentage_c0_mid_in_label[label]
            mask = (atlas_func == label).astype(int)
            print("label: {:30} freq: {}".format(LABELS_ATLAS[label], freq))
            atlas_freq_c0_mid += mask * freq;
            if i >= 6:
                 break;
        print()
        print("max c(0) label frequencies (long)")
        for i in range(len(freq_c0_long_sorted)):
            label = freq_c0_long_sorted[i][0]
            freq = percentage_c0_long_in_label[label]
            mask = (atlas_func == label).astype(int)
            print("label: {:30} freq: {}".format(LABELS_ATLAS[label], freq))
            atlas_freq_c0_long += mask * freq;
            if i >= 6:
                 break;
        print()
        print("c(1) label frequencies (s+m+l)")
        for i in range(len(freq_c1_sorted)):
            label = freq_c1_sorted[i][0]
            freq = percentage_c1_frequency_label[label]
            mask = (atlas_func == label).astype(int)
            atlas_freq_c1 += mask * freq;
            print("label: {:30} freq: {}".format(LABELS_ATLAS[label], freq))
            if i >= 8:
                 break;
        print()
        print("max c(1) label frequencies (short)")
        for i in range(len(freq_c1_short_sorted)):
            label = freq_c1_short_sorted[i][0]
            freq = percentage_c1_short_in_label[label]
            mask = (atlas_func == label).astype(int)
            print("label: {:30} freq: {}".format(LABELS_ATLAS[label], freq))
            atlas_freq_c1_short += mask * freq;
            if i >= 6:
                 break;
        print()
        print("max c(1) label frequencies (mid)")
        for i in range(len(freq_c1_mid_sorted)):
            label = freq_c1_mid_sorted[i][0]
            freq = percentage_c0_mid_in_label[label]
            mask = (atlas_func == label).astype(int)
            print("label: {:30} freq: {}".format(LABELS_ATLAS[label], freq))
            atlas_freq_c1_mid += mask * freq;
            if i >= 6:
                 break;
        print()
        print("max c(1) label frequencies (long)")
        for i in range(len(freq_c1_long_sorted)):
            label = freq_c1_long_sorted[i][0]
            freq = percentage_c1_long_in_label[label]
            mask = (atlas_func == label).astype(int)
            print("label: {:30} freq: {}".format(LABELS_ATLAS[label], freq))
            atlas_freq_c1_long += mask * freq;
            if i >= 6:
                 break;

        atlas_freq_c0 = atlas_freq_c0 / np.amax(atlas_freq_c0.flatten());
        atlas_freq_c1 = atlas_freq_c1 / np.amax(atlas_freq_c1.flatten());

        atlas_freq_c0_short = atlas_freq_c0_short / np.amax(atlas_freq_c0_short.flatten());
        atlas_freq_c0_mid   = atlas_freq_c0_mid   / np.amax(atlas_freq_c0_mid.flatten());
        atlas_freq_c0_long  = atlas_freq_c0_long  / np.amax(atlas_freq_c0_long.flatten());
        atlas_freq_c1_short = atlas_freq_c1_short / np.amax(atlas_freq_c1_short.flatten());
        atlas_freq_c1_mid   = atlas_freq_c1_mid   / np.amax(atlas_freq_c1_mid.flatten());
        atlas_freq_c1_long  = atlas_freq_c1_long  / np.amax(atlas_freq_c1_long.flatten());



    # ### ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #
    if gen_images:
        print("creating slices")
        clrs = ['black','red', 'green', 'yellow', 'blue', 'cyan', 'orange', 'purple'];
        tick_param_kwargs = {"axis":"both", "which":"both", "bottom":False, "left":False, "labelbottom":False, "labelleft":False}
        imshow_kwargs_template = {"cmap":"gray", "aspect":"equal"}
        imshow_kwargs_c1 = {"cmap":plt.cm.rainbow, "aspect":"equal"}
        imshow_kwargs_r = {"cmap":"gray_r", "aspect":"equal"}

        COLOR_BY_SURVIVAL  = False;
        COLOR_BY_SIZE      = False;
        COLOR_BY_FREQUENCY = True;

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
        if COLOR_BY_SURVIVAL:
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
        if COLOR_BY_SIZE:
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
        # -------- functional atlas ------- #
        if COLOR_BY_FREQUENCY:
            fig_freq, axis_freq = plt.subplots(nrows*2, ncols, figsize=(14,17));
            for a,i in zip(axis_freq.flatten(),np.arange(len(axis_freq.flatten()))):
                a.tick_params(**tick_param_kwargs)
                a.set_yticklabels([])
                a.set_xticklabels([])
                a.set_frame_on(False)
            fig_freq_srvl, axis_freq_c0_srvl = plt.subplots(nrows*4, ncols, figsize=(14,34));
            for a,i in zip(axis_freq_c0_srvl.flatten(),np.arange(len(axis_freq_c0_srvl.flatten()))):
                a.tick_params(**tick_param_kwargs)
                a.set_yticklabels([])
                a.set_xticklabels([])
                a.set_frame_on(False)

        j = 0;
        i = 0;
        m = 0;
        m2 = 0;
        ax_slice = 24 - axs_inc;
        for k in range(nslices):
            ax_slice += axs_inc;

            # -------- color by survival ------- #
            if COLOR_BY_SURVIVAL:
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
                axis_c1[m+0,0].set_ylabel("short survivor", size='5')
                axis_c1[m+1,0].set_ylabel("mid survivor",   size='5')
                axis_c1[m+2,0].set_ylabel("long survivor",  size='5')
                axis_c1[m+3,0].set_ylabel("combined (r=short,g=mid,b=long)", size='5')

            # -------- color by size ------- #
            if COLOR_BY_SIZE:
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
                axis_c1_size[m+0,0].set_ylabel("small (TC+0.5*ED)",  size='5')
                axis_c1_size[m+1,0].set_ylabel("medium (TC+0.5*ED)", size='5')
                axis_c1_size[m+2,0].set_ylabel("large (TC+0.5*ED)",  size='5')
                axis_c1_size[m+3,0].set_ylabel("blend (r=small,g=medium,b=large)", size='5')

            # -------- functional atlas ------- #
            if COLOR_BY_FREQUENCY:
                axis_freq[m2+0,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
                axis_freq[m2+1,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
                im1 = axis_freq[m2+0,j].imshow(thresh(atlas_freq_c0[:,:,ax_slice].T, cmap=plt.cm.jet, threshold=0.01, v_max=1, v_min=0), cmap=plt.cm.jet, interpolation='none', aspect='equal', alpha=0.8);
                im2 = axis_freq[m2+1,j].imshow(thresh(atlas_freq_c1[:,:,ax_slice].T, cmap=plt.cm.jet, threshold=0.01, v_max=1, v_min=0), cmap=plt.cm.jet, interpolation='none', aspect='equal', alpha=0.8);
                axis_freq[m2+0,0].set_ylabel("func. atlas by c(0) freq.", size='5')
                axis_freq[m2+1,0].set_ylabel("func. atlas by c(1) freq.", size='5')
                divider = make_axes_locatable(axis_freq[m2+0,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig_freq.colorbar(im1, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=4);
                divider = make_axes_locatable(axis_freq[m2+1,0])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig_freq.colorbar(im2, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=4);
                axis_freq[m2+0,j].set_title("axial slice %d" %  ax_slice , size='5', y=1.0)

                vals = np.ones((ashape[0], ashape[1], 4))
                # max_max = max(max(np.amax(atlas_freq_c0_short.flatten()), np.amax(atlas_freq_c0_mid.flatten())), np.amax(atlas_freq_c0_long.flatten()));
                vals[..., 0] = atlas_freq_c0_short[:,:,ax_slice].T # / max_max # red
                vals[..., 1] = atlas_freq_c0_mid[:,:,ax_slice].T   #/ max_max # green
                vals[..., 2] = atlas_freq_c0_long[:,:,ax_slice].T  #/ max_max # blue
                vals[..., 3] = Normalize(0, 0.01, clip=True)(atlas_freq_c0[:,:,ax_slice].T) # alpha
                cmap_c1_combined = vals;
                # # -- c(1) -- #
                axis_freq_c0_srvl[m+0,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
                axis_freq_c0_srvl[m+1,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
                axis_freq_c0_srvl[m+2,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
                axis_freq_c0_srvl[m+3,j].imshow(atlas[:,:,ax_slice].T, **imshow_kwargs_template);
                im1 = axis_freq_c0_srvl[m+0,j].imshow(thresh(atlas_freq_c0_short[:,:,ax_slice].T,  cmap=plt.cm.jet, threshold=0.01, v_max=1, v_min=0), cmap=plt.cm.jet, interpolation='none', aspect='equal', alpha=0.8);
                im2 = axis_freq_c0_srvl[m+1,j].imshow(thresh(atlas_freq_c0_mid[:,:,ax_slice].T,    cmap=plt.cm.jet, threshold=0.01, v_max=1, v_min=0), cmap=plt.cm.jet, interpolation='none', aspect='equal', alpha=0.8);
                im3 = axis_freq_c0_srvl[m+2,j].imshow(thresh(atlas_freq_c0_long[:,:,ax_slice].T,   cmap=plt.cm.jet, threshold=0.01, v_max=1, v_min=0), cmap=plt.cm.jet, interpolation='none', aspect='equal', alpha=0.8);
                im4 = axis_freq_c0_srvl[m+3,j].imshow(vals, interpolation='none', aspect='equal', alpha=0.8);
                divider = make_axes_locatable(axis_freq_c0_srvl[m+0,j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig_freq_srvl.colorbar(im1, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=4);
                divider = make_axes_locatable(axis_freq_c0_srvl[m+1,j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig_freq_srvl.colorbar(im2, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=4);
                divider = make_axes_locatable(axis_freq_c0_srvl[m+2,j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cbar = fig_freq_srvl.colorbar(im3, cax=cax, orientation='vertical')
                cbar.ax.tick_params(labelsize=4);
                axis_freq_c0_srvl[m+0,j].set_title("axial slice %d" %  ax_slice , size='5', y=1.0)
                axis_freq_c0_srvl[m+0,0].set_ylabel("short survivor", size='5')
                axis_freq_c0_srvl[m+1,0].set_ylabel("mid survivor",   size='5')
                axis_freq_c0_srvl[m+2,0].set_ylabel("long survivor",  size='5')
                axis_freq_c0_srvl[m+3,0].set_ylabel("combined (r=short,g=mid,b=long)", size='5')

            i  = i  + 1 if k % ncols == 0 and k > 0 else i
            j  = j  + 1 if k % ncols != 0 else 0
            m  = m  + 4 if k % ncols == 0 and k > 0 else m
            m2 = m2 + 2 if k % ncols == 0 and k > 0 else m2
        # colorbar
        # if COLOR_BY_FREQUENCY:
        #   fig_freq.colorbar(im1, cax=axis_freq[m2+1,j]  orientation="horizontal")
        vpath    = os.path.join(dir,'vis-atlas/');
        vpath_c0 = os.path.join(dir,'vis-atlas/glioma-atlas-c0');
        vpath_c1 = os.path.join(dir,'vis-atlas/glioma-atlas-c1');

        if COLOR_BY_SURVIVAL or COLOR_BY_SIZE:
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

        if COLOR_BY_FREQUENCY:
            fig_freq.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.4, hspace=0.4);
            fig_freq.tight_layout();
            fig_freq.savefig(os.path.join(vpath, "functional_atlas_freq.pdf"), format='pdf', dpi=1200);

            fig_freq_srvl.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.4, hspace=0.4);
            fig_freq_srvl.tight_layout();
            fig_freq_srvl.savefig(os.path.join(vpath, "functional_atlas_c0_freq.pdf"), format='pdf', dpi=1200);














# ### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #
# ### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #
# ### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #


# c(0) distinctive features for classification:
# ### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #
# SHORT                          highly distictive: [lateral_orbitofrontal_gyrus, gyrus_rectus, superior_occipital_gyrus]; less distinctive: [postcentral_gyrus, inferior_occipital_gyrus, insular_cortex, hippocampus]
# SHORT to MID (i.e., not LONG)  highly distictive: [];                                                                    less distinctive: [hippocampus, parahippocampal_gyrus]
# MID                            highly distictive: [caudate, putamen];                                                    less distinctive: [precentral_gyrus, brainstem]
# MID to LONG (i.e., not SHORT)  highly distictive: [inferior_frontal_gyrus, supramarginal_gyrus];                         less distinctive: [angular_gyrus, brainstem]
# LONG                           highly distictive: [superior_frontal_gyrus, background (that means close to skull)];      less distinctive: [cerebellum]
#
# c(0) frequencies:
# ### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #
#  SHORT:                         [unlabelled white matter (39.6),
#                                  superior_temporal_gyrus (20.9),
#                                  middle_temporal_gyrus   (16.2),
#                                  hippocampus             (13.6),
#                                  inferior_temporal_gyrus (13.2),
#                                  ventricles (i.e., close to) (13.1)];  further include: [superior_frontal_gyrus, superior_parietal_gyrus, insular_cortex, middle_frontal_gyrus, precuneus, superior_occipital_gyrus]

#  MID:                           [unlabelled white matter (25.3),
#                                  superior_temporal_gyrus (23.9),
#                                  middle_frontal_gyrus    (12.5),
#                                  superior_parietal_gyrus (12.1),
#                                  precentral_gyrus        (11.6),
#                                  superior_frontal_gyrus  (10.5)];     further include: [angular_gyrus, ventricles (that means close to), ]

#  LONG:                          [unlabelled white matter (32.87),
#                                  superior_frontal_gyrus  (25.6),
#                                  middle_temporal_gyrus   (20.6),
#                                  inferior_temporal_gyrus (17.2),
#                                  superior_temporal_gyrus (12.2),
#                                  middle_frontal_gyrus    (11.6)];     further include: [fusiform_gyrus, lingual_gyrus, background (that means close to skull)]

#  ALL(FREQ):                     [unlabelled white matter (58.2),
#                                  superior_temporal_gyrus (34.9),
#                                  superior_frontal_gyrus  (30.0),   // frequent + discriminative
#                                  middle_temporal_gyrus   (27.3),
#                                  inferior_temporal_gyrus (22.3),
#                                  middle_frontal_gyrus    (20.3),
#                                  superior_parietal_gyrus (19.3),
#                                  ventricles (i.e., close to) (17.5)
#                                  ]
# label sizes:
# ### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #
#                                  [unlabelled white matter (20.7),
#                                  cerebellum              (11.8),
#                                  superior_frontal_gyrus  (8.8),
#                                  middle_frontal_gyrus    (8.1),
#                                  precentral_gyrus        (3.9),
#                                  superior_temporal_gyrus (3.7),
#                                  superior_parietal_gyrus (3.7),
#                                  postcentral_gyrus       (3.8)
#                                  ]
# ### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #
#

# rel. l2norm of c(0) in brain regions divided by survival:
# ### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #
#                               # relative to gbm c(0) atlas    # relative per label     # pixel  #
# +-----------------------------+-------+-------+-------+-------++-------+-------+-------++------+
# | label                       | short | mid   | long  | all   || short | mid   | long  || size |
# +-----------------------------+-------+-------+-------+-------++-------+-------+-------++------+
# | background                  | 1.1   | 2.31  | 7.11  | 7.56  || 14.6  | 30.59 | 94.08 || ---  |  << LONG (strong)
# | superior_frontal_gyrus      | 11.43 | 10.57 | 25.65 | 30.04 || 38.06 | 35.2  | 85.39 || 8.81 |  << LONG
# | middle_frontal_gyrus        | 10.88 | 12.54 | 11.6  | 20.31 || 53.56 | 61.77 | 57.12 || 8.14 |  << (uninformative)
# | inferior_frontal_gyrus      | 0.07  | 6.41  | 6.71  | 9.28  || 0.74  | 69.08 | 72.3  || 2.98 |  << MID to LONG (strong)
# | precentral_gyrus            | 5.11  | 11.68 | 4.39  | 13.48 || 37.89 | 86.63 | 32.55 || 3.96 |  << MID
# | middle_orbitofrontal_gyrus  | 0.31  | 0.0   | 0.41  | 0.52  || 60.39 | 0.0   | 79.71 || 1.43 |  << (uninformative)
# | lateral_orbitofrontal_gyrus | 1.57  | 0.0   | 0.07  | 1.58  || 99.9  | 0.01  | 4.49  || 0.73 |  << SHORT (strong)
# | gyrus_rectus                | 0.11  | 0.0   | 0.0   | 0.11  || 100.0 | 0.0   | 0.0   || 0.45 |  << SHORT (strong)
# | postcentral_gyrus           | 9.15  | 4.68  | 1.19  | 10.46 || 87.53 | 44.77 | 11.4  || 3.08 |  << SHORT
# | superior_parietal_gyrus     | 11.07 | 12.01 | 9.6   | 19.37 || 57.18 | 62.03 | 49.56 || 3.73 |  << (uninformative)
# | supramarginal_gyrus         | 0.05  | 5.99  | 6.3   | 8.74  || 0.57  | 68.52 | 72.12 || 1.55 |  << MID to LONG
# | angular_gyrus               | 3.44  | 8.29  | 7.59  | 11.82 || 29.07 | 70.17 | 64.25 || 2.13 |  << MID to LONG (weak)
# | precuneus                   | 10.01 | 2.89  | 8.95  | 13.85 || 72.25 | 20.84 | 64.58 || 1.43 |  << (uninformative)
# | superior_occipital_gyrus    | 11.25 | 3.76  | 3.67  | 12.42 || 90.6  | 30.25 | 29.56 || 1.68 |  << SHORT
# | middle_occipital_gyrus      | 6.04  | 5.64  | 3.94  | 9.15  || 66.0  | 61.57 | 43.04 || 3.12 |  << (uninformative)
# | inferior_occipital_gyrus    | 4.76  | 0.0   | 1.37  | 4.95  || 96.1  | 0.0   | 27.66 || 1.43 |  << SHORT
# | cuneus                      | 0.1   | 0.0   | 0.0   | 0.1   || 100.0 | 0.0   | 0.02  || 0.87 |  << (uninformative)
# | superior_temporal_gyrus     | 20.94 | 23.92 | 12.22 | 34.94 || 59.94 | 68.47 | 34.98 || 3.74 |  << (uninformative)
# | middle_temporal_gyrus       | 16.27 | 5.86  | 20.68 | 27.33 || 59.51 | 21.43 | 75.65 || 3.16 |  << (uninformative);  SHORT + LONG
# | inferior_temporal_gyrus     | 13.28 | 4.67  | 17.23 | 22.34 || 59.44 | 20.89 | 77.13 || 2.99 |  << (uninformative);  SHORT + LONG
# | parahippocampal_gyrus       | 8.56  | 7.59  | 4.78  | 12.51 || 68.41 | 60.67 | 38.22 || 1.16 |  << SHORT to MID (weak)
# | lingual_gyrus               | 4.76  | 0.08  | 7.64  | 9.01  || 52.88 | 0.85  | 84.82 || 2.3  |  << (uninformative);  SHORT + LONG
# | fusiform_gyrus              | 6.02  | 1.06  | 8.15  | 10.22 || 58.92 | 10.42 | 79.77 || 1.77 |  << (uninformative);  SHORT + LONG
# | insular_cortex              | 11.65 | 4.08  | 3.34  | 13.09 || 89.01 | 31.2  | 25.48 || 0.96 |  << SHORT
# | cingulate_gyrus             | 7.41  | 3.95  | 6.33  | 10.52 || 70.44 | 37.56 | 60.19 || 1.83 |  << (uninformative)
# | caudate                     | 1.85  | 7.16  | 3.03  | 8.82  || 20.95 | 81.21 | 34.4  || 0.58 |  << MID (strong)
# | putamen                     | 2.5   | 7.88  | 1.27  | 8.37  || 29.92 | 94.18 | 15.2  || 0.56 |  << MID (strong)
# | hippocampus                 | 13.24 | 6.64  | 4.1   | 15.63 || 84.69 | 42.47 | 26.22 || 0.56 |  << STRONG
# | cerebellum                  | 0.0   | 0.0   | 3.49  | 3.49  || 0.01  | 0.0   | 100.0 || 11.85|  << LONG
# | brainstem                   | 0.0   | 2.78  | 1.53  | 3.17  || 0.11  | 87.59 | 48.24 || 1.5  |  << MID to LONG
# | unlabelled white matter     | 39.62 | 25.36 | 32.87 | 58.2  || 68.07 | 43.58 | 56.47 || 20.7 |  << (UNINFORMATIVE) NEED TO BREAK DOWN INTO SUB-LABELS
# | ventricles                  | 13.1  | 8.65  | 5.69  | 17.59 || 74.46 | 49.18 | 32.33 || 0.82 |  << (uninformative)
# +-----------------------------+-------+-------+-------+-------++-------+-------+-------++------+


# Percentage c(1) in brain regions divided by survival:
# ### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #
#                               # relative to gbm c(0) atlas    # relative per label     # pixel  #
# +-----------------------------+-------+-------+-------+-------+-+-------+-------+-------++-----+
# | label                       | short | mid   | long  | all   | | short | mid   | long  ||size |
# +-----------------------------+-------+-------+-------+-------+ +-------+-------+-------++-----+
# | background                  | 3.49  | 2.27  | 3.18  | 6.1   | | 57.2  | 37.16 | 52.11 ||---  |
# | superior_frontal_gyrus      | 5.82  | 3.83  | 4.58  | 11.93 | | 48.73 | 32.1  | 38.39 ||8.81 |
# | middle_frontal_gyrus        | 6.05  | 5.91  | 4.42  | 14.42 | | 41.99 | 40.96 | 30.67 ||8.14 |
# | inferior_frontal_gyrus      | 2.77  | 4.09  | 2.18  | 8.03  | | 34.55 | 50.95 | 27.19 ||2.98 |
# | precentral_gyrus            | 3.46  | 2.8   | 3.19  | 7.79  | | 44.4  | 35.88 | 40.96 ||3.96 |
# | middle_orbitofrontal_gyrus  | 2.97  | 0.83  | 1.56  | 4.95  | | 60.06 | 16.82 | 31.52 ||1.43 |
# | lateral_orbitofrontal_gyrus | 2.45  | 1.09  | 1.28  | 4.41  | | 55.68 | 24.74 | 29.13 ||0.73 |
# | gyrus_rectus                | 1.59  | 0.1   | 0.14  | 1.74  | | 91.43 | 5.77  | 8.26  ||0.45 |
# | postcentral_gyrus           | 4.97  | 3.36  | 3.72  | 10.53 | | 47.23 | 31.95 | 35.3  ||3.08 |
# | superior_parietal_gyrus     | 9.91  | 6.87  | 5.35  | 18.6  | | 53.27 | 36.93 | 28.74 ||3.73 |
# | supramarginal_gyrus         | 2.86  | 2.24  | 3.75  | 8.43  | | 33.95 | 26.58 | 44.41 ||1.55 |
# | angular_gyrus               | 4.92  | 3.52  | 4.11  | 11.65 | | 42.2  | 30.23 | 35.28 ||2.13 |
# | precuneus                   | 5.44  | 2.69  | 2.93  | 9.18  | | 59.29 | 29.27 | 31.88 ||1.43 |
# | superior_occipital_gyrus    | 4.03  | 1.98  | 1.6   | 6.85  | | 58.75 | 28.86 | 23.33 ||1.68 |
# | middle_occipital_gyrus      | 6.16  | 1.61  | 2.43  | 9.14  | | 67.39 | 17.57 | 26.52 ||3.12 |
# | inferior_occipital_gyrus    | 1.73  | 0.09  | 2.2   | 3.44  | | 50.47 | 2.66  | 64.06 ||1.43 |
# | cuneus                      | 1.56  | 0.31  | 0.23  | 1.85  | | 84.34 | 16.65 | 12.59 ||0.87 |
# | superior_temporal_gyrus     | 14.51 | 6.34  | 7.14  | 25.95 | | 55.89 | 24.44 | 27.5  ||3.74 |
# | middle_temporal_gyrus       | 12.76 | 5.75  | 8.15  | 24.87 | | 51.29 | 23.13 | 32.76 ||3.16 |
# | inferior_temporal_gyrus     | 8.82  | 3.16  | 6.15  | 15.98 | | 55.17 | 19.76 | 38.49 ||2.99 |
# | parahippocampal_gyrus       | 7.3   | 3.37  | 2.75  | 12.61 | | 57.88 | 26.74 | 21.81 ||1.16 |
# | lingual_gyrus               | 4.66  | 1.45  | 2.03  | 7.7   | | 60.57 | 18.9  | 26.43 ||2.3  |
# | fusiform_gyrus              | 6.3   | 2.71  | 6.1   | 14.27 | | 44.13 | 18.98 | 42.75 ||1.77 |
# | insular_cortex              | 8.51  | 5.45  | 3.1   | 15.91 | | 53.47 | 34.28 | 19.51 ||0.96 |
# | cingulate_gyrus             | 6.06  | 3.02  | 3.19  | 11.22 | | 54.03 | 26.92 | 28.39 ||1.83 |
# | caudate                     | 4.84  | 4.04  | 2.02  | 10.32 | | 46.85 | 39.15 | 19.57 ||0.58 |
# | putamen                     | 7.48  | 4.63  | 2.7   | 13.87 | | 53.95 | 33.39 | 19.49 ||0.56 |
# | hippocampus                 | 8.75  | 3.31  | 2.94  | 14.0  | | 62.49 | 23.65 | 20.99 ||0.56 |
# | cerebellum                  | 0.27  | 0.13  | 1.13  | 1.26  | | 21.81 | 10.26 | 90.01 ||11.85|
# | brainstem                   | 2.27  | 1.49  | 0.87  | 4.07  | | 55.76 | 36.47 | 21.26 ||1.5  |
# | unlabelled white matter     | 40.38 | 19.81 | 18.13 | 73.7  | | 54.78 | 26.88 | 24.61 ||20.7 |
# | ventricles                  | 9.12  | 4.26  | 3.37  | 15.9  | | 57.32 | 26.79 | 21.22 ||0.82 |
# +-----------------------------+-------+-------+-------+-------+-+-------+-------+-------++-----+
