import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../common/'))
import matplotlib as mpl
# mpl.use('Agg')
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
# # -----------------------------------
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


###
### ------------------------------------------------------------------------ ###
anthrazit = (0.2431, 0.2666, 0.2980)
mittelblau = (0., 0.31764, 0.61960)
hellblau = (0., 0.7529411, 1.)
signalred = (1., 0., 0.12549)
hellblauT = (0., 0.7529411, 1., 0.1)
signalredT = (1., 0., 0.12549, 0.1)

pastGcolrA = '#045c9b'
pastGcolrB = '#068c75'
pastGcolrC = '#7ad3e3'
pastGcolrD = '#0ca3ab'
pastGcolrE = '#7cb4d4'

###
### ------------------------------------------------------------------------ ###
def getSurvivalClass(x):
    m = x/30.
    if m < 10:
        return 0;
    elif m < 15:
        return 1;
    else:
        return 2;

###
### ------------------------------------------------------------------------ ###
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
def setBoxColors(bp):
    setp(bp['boxes'][0], color=hellblau)
    bp['boxes'][0].set(facecolor=hellblauT)
    setp(bp['caps'][0], color='black')
    setp(bp['caps'][1], color='black')
    setp(bp['whiskers'][0], color='black')
    setp(bp['whiskers'][1], color='black')
    setp(bp['fliers'][0], color='green')
    setp(bp['fliers'][1], color='green')
    setp(bp['medians'][0], color='black')

    setp(bp['boxes'][1], color=signalred)
    bp['boxes'][1].set(facecolor=signalredT)
    setp(bp['caps'][2], color='black')
    # setp(bp['caps'][3], color=signalred)
    setp(bp['whiskers'][2], color='black')
    # setp(bp['whiskers'][3], color=signalred)
    # setp(bp['fliers'][2], color=signalred)
    # setp(bp['fliers'][3], color=signalred)
    setp(bp['medians'][1], color='black')

###
### ------------------------------------------------------------------------ ###
def read_data(file):
    FILTER = ['Brats18_CBICA_AUR_1', 'Brats18_CBICA_ANI_1']
    FAILED_TO_ADD = []
    max_l2c1error=0.8  # filter out 'failed cases'
    # read brats data
    if file is not None:
        brats_data = pd.read_csv(os.path.join('./',file), header = 0, error_bad_lines=True, skipinitialspace=True)
        print("read brats simulation data of length %d" % len(brats_data))

    nbshort = len(brats_data.loc[brats_data['survival_class'] ==  0])
    nbmid   = len(brats_data.loc[brats_data['survival_class'] ==  1])
    nblong  = len(brats_data.loc[brats_data['survival_class'] ==  2])
    sum     = nbshort + nbmid + nblong
    class_weights = {}
    class_weights[0] = nbshort / sum;
    class_weights[1] = nbmid / sum;
    class_weights[2] = nblong / sum;
    # for i in range(1,4):
        # weights[i] = 1.
    print("work dataset of length %d/%d. short: %d, mid: %d, long: %d" % (len(brats_data),len(survival_data),nbshort,nbmid,nblong))
    return brats_data, class_weights;


###
### ------------------------------------------------------------------------ ###
def clean_data(brats_data, max_l2c1error = 0.8, filter_GTR=True):

    # add vol(EN)/vol(TC)
    brats_data['vol(EN)/vol(TC)'] = brats_data['vol(EN)_a'].astype('float') / brats_data['vol(TC)_a'].astype('float');

    # 1. add rho-over-k
    brats_data['rho-inv'] = brats_data['rho-inv'].astype('float')
    brats_data['k-inv']   = brats_data['k-inv'].astype('float')
    dat_out  = brats_data.loc[brats_data['k-inv'] <= 0]
    brats_data  = brats_data.loc[brats_data['k-inv'] >  0]
    dat_out["filter-reason"] = "k zero"
    dat_filtered_out = dat_out;
    brats_data["rho-over-k"] = brats_data["rho-inv"]/brats_data["k-inv"]

    # 2. filter data with too large misfit
    brats_data['l2[Oc(1)-TC]'] = brats_data['l2[Oc(1)-TC]'].astype('float')
    dat_out = brats_data.loc[brats_data['l2[Oc(1)-TC]'] >= max_l2c1error]
    brats_data = brats_data.loc[brats_data['l2[Oc(1)-TC]'] <  max_l2c1error]
    dat_out["filter-reason"] = "l2(Oc1-d)err > "+str(max_l2c1error)
    dat_filtered_out = dat_out;
    # dat_filtered_out = pd.concat([dat_filtered_out, dat_out], axis=0)

    brats_survival = brats_data.copy();

    # 3. filter survival data
    brats_survival['age'] = brats_survival['age'].astype('float')
    dat_out = brats_survival.loc[brats_survival['age'] <= 0]
    brats_survival = brats_survival.loc[brats_survival['age'] >  0]
    dat_out["filter-reason"] = "no survival data"
    # dat_filtered_out = pd.concat([dat_filtered_out, dat_out], axis=0)

    brats_clustering = brats_survival.copy();

    # 4. filter GTR resection status
    if filter_GTR:
        dat_out = brats_survival.loc[brats_survival['resection_status'] != 'GTR']
        brats_survival = brats_survival.loc[brats_survival['resection_status'] ==  'GTR']
        dat_out["filter-reason"] = "no GTR"
        dat_filtered_out = pd.concat([dat_filtered_out, dat_out], axis=0)
    else:
        brats_survival['resection_status[0]'] = brats_survival['resection_status'].apply(lambda x: np.array([0]) if pd.isna(x) else np.array([0]) if x == 'STR' else np.array([1]) );
        brats_survival['resection_status[1]'] = brats_survival['resection_status'].apply(lambda x: np.array([0]) if pd.isna(x) else np.array([1]) if x == 'STR' else np.array([0]) );


    print("\n\n### BraTS simulation data [cleaned] ### ")
    print(tabulate(brats_survival[["BID", "survival(days)", "age", "resection_status"]], headers='keys', tablefmt='psql'))
    print("\n\n### BraTS simulation data [discarded] ### ")
    print(tabulate(dat_filtered_out[["BID", "filter-reason"]], headers='keys', tablefmt='psql'))
    print()
    print("remaining data set for clustering consists of {} patients".format(len(brats_clustering)))
    print("remaining data set for survival prediction consists of {} patients".format(len(brats_survival)))
    return brats_clustering, brats_survival;


###
### ------------------------------------------------------------------------ ###
def analyze_frequenzies(dir, brats_data, LABELS_ATLAS):
    print("reading glioma atlasses");
    atlas = nib.load(os.path.join(dir,"jakob_segmented_with_cere_lps_240x240x155_in_brats_hdr.nii.gz"));
    atlas_t1 = nib.load(os.path.join(dir,"jakob_stripped_with_cere_lps_240x240x155_in_brats_hdr.nii.gz"));
    atlas_func = nib.load(os.path.join(dir,"lpba40_combined_LR_256x256x256_aff2jakob_in_jakob_space_240x240x155.nii.gz")).get_fdata();
    # atlas_func = nib.load(os.path.join(dir,"Template16_label_aff2jacob_warped.nii.gz")).get_fdata();
    affine = atlas.affine;
    ashape = atlas.shape;
    atlas = atlas.get_fdata();
    glioma_c1_atlas_short_abs = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_short_abs.nii.gz")).get_fdata();
    glioma_c1_atlas_mid_abs   = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_mid_abs.nii.gz")).get_fdata();
    glioma_c1_atlas_long_abs  = nib.load(os.path.join(dir, "brats[srvl]_c1_atlas_long_abs.nii.gz")).get_fdata();
    glioma_c0_atlas_short_abs = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_short_abs.nii.gz")).get_fdata();
    glioma_c0_atlas_mid_abs   = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_mid_abs.nii.gz")).get_fdata();
    glioma_c0_atlas_long_abs  = nib.load(os.path.join(dir, "brats[srvl]_c0_atlas_long_abs.nii.gz")).get_fdata();
    print("computing stats")
    atlas_sml_c0_abs  = glioma_c0_atlas_short_abs + glioma_c0_atlas_mid_abs + glioma_c0_atlas_long_abs;
    atlas_sml_c1_abs  = glioma_c1_atlas_short_abs + glioma_c1_atlas_mid_abs + glioma_c1_atlas_long_abs;

    ref_c0 = np.linalg.norm(atlas_sml_c0_abs.flatten(), 2)**2;
    ref_c1 = np.linalg.norm(atlas_sml_c1_abs.flatten(), 2)**2;

    l2sumc0_label_over_l2sumc0__SHORT = {}
    l2sumc0_label_over_l2sumc0__MID   = {}
    l2sumc0_label_over_l2sumc0__LONG  = {}
    l2sumc1_label_over_l2sumc1__SHORT = {}
    l2sumc1_label_over_l2sumc1__MID   = {}
    l2sumc1_label_over_l2sumc1__LONG  = {}
    l2sumc0_label_over_l2sumc0_label__SHORT = {}
    l2sumc0_label_over_l2sumc0_label__MID   = {}
    l2sumc0_label_over_l2sumc0_label__LONG  = {}
    l2sumc1_label_over_l2sumc1_label__SHORT = {}
    l2sumc1_label_over_l2sumc1_label__MID   = {}
    l2sumc1_label_over_l2sumc1_label__LONG  = {}
    l2sumc0_sml_label_over_l2sumc0 = {}
    l2sumc1_sml_label_over_l2sumc1 = {}
    size_label = {}

    # frequencies based on summed c(0) and summed c(1) over short, mid and long survivors
    for label, descr in LABELS_ATLAS.items():
        mask = (atlas_func == label).astype(int)

        ref_c0_label = np.linalg.norm(np.multiply(atlas_sml_c0_abs, mask).flatten(), 2)**2;
        ref_c1_label = np.linalg.norm(np.multiply(atlas_sml_c1_abs, mask).flatten(), 2)**2;

        l2sumc0_sml_label_over_l2sumc0[label]  = ref_c0_label  / ref_c0;
        l2sumc1_sml_label_over_l2sumc1[label]  = ref_c1_label  / ref_c1;

        l2sumc0_label_short = np.linalg.norm(np.multiply(glioma_c0_atlas_short_abs, mask).flatten(), 2)**2;
        l2sumc0_label_mid   = np.linalg.norm(np.multiply(glioma_c0_atlas_mid_abs, mask).flatten(), 2)**2;
        l2sumc0_label_long  = np.linalg.norm(np.multiply(glioma_c0_atlas_long_abs, mask).flatten(), 2)**2;
        l2sumc1_label_short = np.linalg.norm(np.multiply(glioma_c1_atlas_short_abs, mask).flatten(), 2)**2;
        l2sumc1_label_mid   = np.linalg.norm(np.multiply(glioma_c1_atlas_mid_abs, mask).flatten(), 2)**2;
        l2sumc1_label_long  = np.linalg.norm(np.multiply(glioma_c1_atlas_long_abs, mask).flatten(), 2)**2;

        l2sumc0_label_over_l2sumc0__SHORT[label] = l2sumc0_label_short / ref_c0;
        l2sumc0_label_over_l2sumc0__MID[label]   = l2sumc0_label_mid   / ref_c0;
        l2sumc0_label_over_l2sumc0__LONG[label]  = l2sumc0_label_long  / ref_c0;
        l2sumc1_label_over_l2sumc1__SHORT[label] = l2sumc1_label_short / ref_c1;
        l2sumc1_label_over_l2sumc1__MID[label]   = l2sumc1_label_mid   / ref_c1;
        l2sumc1_label_over_l2sumc1__LONG[label]  = l2sumc1_label_long  / ref_c1;

        l2sumc0_label_over_l2sumc0_label__SHORT[label] = l2sumc0_label_short / ref_c0_label;
        l2sumc0_label_over_l2sumc0_label__MID[label]   = l2sumc0_label_mid   / ref_c0_label;
        l2sumc0_label_over_l2sumc0_label__LONG[label]  = l2sumc0_label_long  / ref_c0_label;
        l2sumc1_label_over_l2sumc1_label__SHORT[label] = l2sumc1_label_short / ref_c1_label;
        l2sumc1_label_over_l2sumc1_label__MID[label]   = l2sumc1_label_mid   / ref_c1_label;
        l2sumc1_label_over_l2sumc1_label__LONG[label]  = l2sumc1_label_long  / ref_c1_label;

        size_label[label] = np.linalg.norm(mask.flatten(),0) / np.linalg.norm(((atlas_func > 0).astype(int).flatten()),0);

    # frequencies based on center fo mass of c(0) and TC, subdivided into short, mid, long survivors
    nb_short = 0;
    nb_mid   = 0;
    nb_long  = 0;
    freq_cm_c0_in_label__SHORT = {};
    freq_cm_c0_in_label__MID   = {};
    freq_cm_c0_in_label__LONG  = {};
    freq_cm_TC_in_label__SHORT = {};
    freq_cm_TC_in_label__MID   = {};
    freq_cm_TC_in_label__LONG  = {};
    for l, d in LABELS_ATLAS.items():
        freq_cm_c0_in_label__SHORT[l] = 0;
        freq_cm_c0_in_label__MID[l]   = 0;
        freq_cm_c0_in_label__LONG[l]  = 0;
        freq_cm_TC_in_label__SHORT[l] = 0;
        freq_cm_TC_in_label__MID[l]   = 0;
        freq_cm_TC_in_label__LONG[l]  = 0;
    brats_data['labels_func_atlas(cm(c0))'].astype(str);
    brats_data['labels_func_atlas(cm(TC))'].astype(str);
    UNDEFINED = []
    for index, row in brats_data.iterrows():
        undef = False;
        if pd.isna(row['labels_func_atlas(cm(c0))']) or pd.isna(row['labels_func_atlas(cm(TC))']):
            continue;
        try:
            label_cm_c0 = int(row['labels_func_atlas(cm(c0))'].split(',')[1]);
        except ValueError:
            label_cm_c0 = 0;
            undef = True;
        try:
            label_cm_TC = int(row['labels_func_atlas(cm(TC))'].split(',')[1]);
        except ValueError:
            label_cm_TC = 0;
            undef = True;

        if undef:
            UNDEFINED.append(row['BID'])

        # short
        if row['survival_class'] == 0:
            nb_short += 1;
            freq_cm_c0_in_label__SHORT[label_cm_c0] += 1;
            freq_cm_TC_in_label__SHORT[label_cm_TC] += 1;
        # mid
        if row['survival_class'] == 1:
            nb_mid += 1;
            freq_cm_c0_in_label__MID[label_cm_c0] += 1;
            freq_cm_TC_in_label__MID[label_cm_TC] += 1;

        # long
        if row['survival_class'] == 2:
            nb_long += 1;
            freq_cm_c0_in_label__LONG[label_cm_c0] += 1;
            freq_cm_TC_in_label__LONG[label_cm_TC] += 1;

    total_hits = nb_short + nb_mid + nb_long;
    for l, d in LABELS_ATLAS.items():
        freq_cm_c0_in_label__SHORT[l] /= float(total_hits);
        freq_cm_c0_in_label__MID[l]   /= float(total_hits);
        freq_cm_c0_in_label__LONG[l]  /= float(total_hits);
        freq_cm_TC_in_label__SHORT[l] /= float(total_hits);
        freq_cm_TC_in_label__MID[l]   /= float(total_hits);
        freq_cm_TC_in_label__LONG[l]  /= float(total_hits);


    print(" UNDEFINED BRAIN IDs: \n", UNDEFINED)

    # sort by size of label
    size_label_sorted = sorted(size_label.items(), key=lambda x: x[1], reverse=True);
    # sort by overall (s+m+l) frequency of label
    tumor_c0_freq_in_label_sml  = sorted(l2sumc0_sml_label_over_l2sumc0.items(), key=lambda x: x[1], reverse=True);
    tumor_c1_freq_in_label_sml  = sorted(l2sumc1_sml_label_over_l2sumc1.items(), key=lambda x: x[1], reverse=True);

    table = [];
    cols  = ['label', 'name', 'rel. size', 'c0 freq.', 'c1 freq.',
            'rel.l2[sum_{S}(c0)]', 'rel.l2[sum_{M}(c0)]', 'rel.l2[sum_{L}(c0)]', 'discriminability #1',
            'rel.#hits_{S}[cm(c0)]', 'rel.#hits_{M}[cm(c0)]', 'rel.#hits_{L}[cm(c0)]', 'discriminability #2',
            'rel.#hits_{S}[cm(TC)]', 'rel.#hits_{M}[cm(TC)]', 'rel.#hits_{L}[cm(TC)]', 'discriminability #3',
            'rel.l2[sum_{S}(c1)]', 'rel.l2[sum_{M}(c1)]', 'rel.l2[sum_{L}(c1)]', 'discriminability #4',
            ];
    cols_long  = ['label', 'name', 'rel. size', 'c0 freq.', 'c1 freq.',
            'freq_short', 'freq_mid', 'freq_long', 'discriminability', 'type'
            ];
    df      = pd.DataFrame(columns=cols,      index = [i for i in range(len(tumor_c0_freq_in_label_sml))]);
    df_long = pd.DataFrame(columns=cols_long, index = [i for i in range(4*len(tumor_c0_freq_in_label_sml))]);
    j = 0;
    for i in range(len(tumor_c0_freq_in_label_sml)):
        # label = size_label_sorted[i][0];
        label = tumor_c0_freq_in_label_sml[i][0];
        name = LABELS_ATLAS[label];
        ROW = [label, name, float(size_label[label]),
                float(l2sumc0_sml_label_over_l2sumc0[label]), float(l2sumc1_sml_label_over_l2sumc1[label]),
                float(l2sumc0_label_over_l2sumc0__SHORT[label]), float(l2sumc0_label_over_l2sumc0__MID[label]), float(l2sumc0_label_over_l2sumc0__LONG[label]),
                judge_discriminability(float(l2sumc0_label_over_l2sumc0__SHORT[label]), float(l2sumc0_label_over_l2sumc0__MID[label]), float(l2sumc0_label_over_l2sumc0__LONG[label])),
                float(freq_cm_c0_in_label__SHORT[label]), float(freq_cm_c0_in_label__MID[label]), float(freq_cm_c0_in_label__LONG[label]),
                judge_discriminability(float(freq_cm_c0_in_label__SHORT[label]), float(freq_cm_c0_in_label__MID[label]), float(freq_cm_c0_in_label__LONG[label])),
                float(freq_cm_TC_in_label__SHORT[label]), float(freq_cm_TC_in_label__MID[label]), float(freq_cm_TC_in_label__LONG[label]),
                judge_discriminability(float(freq_cm_TC_in_label__SHORT[label]), float(freq_cm_TC_in_label__MID[label]), float(freq_cm_TC_in_label__LONG[label])),
                float(l2sumc1_label_over_l2sumc1__SHORT[label]), float(l2sumc1_label_over_l2sumc1__MID[label]), float(l2sumc1_label_over_l2sumc1__LONG[label]),
                judge_discriminability(float(l2sumc1_label_over_l2sumc1__SHORT[label]), float(l2sumc1_label_over_l2sumc1__MID[label]), float(l2sumc1_label_over_l2sumc1__LONG[label])),
                ];
        df.loc[i]        = pd.Series({ c:r for c,r in zip(cols, ROW) });
        df_long.loc[j+0] = pd.Series({ cols_long[0] : ROW[0], cols_long[1] : ROW[1], cols_long[2] : ROW[2], cols_long[3] : ROW[3], cols_long[4] : ROW[4], cols_long[5] : ROW[5+0],  cols_long[6] : ROW[6+0],  cols_long[7] : ROW[7+0],  cols_long[8] : ROW[8+0],  cols_long[9] : 'c(0)'});
        df_long.loc[j+1] = pd.Series({ cols_long[0] : ROW[0], cols_long[1] : ROW[1], cols_long[2] : ROW[2], cols_long[3] : ROW[3], cols_long[4] : ROW[4], cols_long[5] : ROW[5+4],  cols_long[6] : ROW[6+4],  cols_long[7] : ROW[7+4],  cols_long[8] : ROW[8+4],  cols_long[9] : 'cm[c(0)]'});
        df_long.loc[j+2] = pd.Series({ cols_long[0] : ROW[0], cols_long[1] : ROW[1], cols_long[2] : ROW[2], cols_long[3] : ROW[3], cols_long[4] : ROW[4], cols_long[5] : ROW[5+8],  cols_long[6] : ROW[6+8],  cols_long[7] : ROW[7+8],  cols_long[8] : ROW[8+8],  cols_long[9] : 'cm[TC]'});
        df_long.loc[j+3] = pd.Series({ cols_long[0] : ROW[0], cols_long[1] : ROW[1], cols_long[2] : ROW[2], cols_long[3] : ROW[3], cols_long[4] : ROW[4], cols_long[5] : ROW[5+12], cols_long[6] : ROW[6+12], cols_long[7] : ROW[7+12], cols_long[8] : ROW[8+12], cols_long[9] : 'c(1)'});
        table.append(ROW);
        j += 4;

    print()
    print("Accumulated c(0) and c(1) frequencies per label of the functional atlas, subdivided into survival classes SHOT, MID, LONG:");
    print(tabulate(table,
                    headers = ['label', 'name', 'rel. size', 'c0 freq.', 'c1 freq.',
                        'rel.l2\n[sum_{S}(c0)]', 'rel.l2\n[sum_{M}(c0)]', 'rel.l2\n[sum_{L}(c0)]', 'discriminability #1',
                        'rel.#hits_{S}\n[cm(c0)]', 'rel.#hits_{M}\n[cm(c0)]', 'rel.#hits_{L}\n[cm(c0)]', 'discriminability #2',
                        'rel.#hits_{S}\n[cm(TC)]', 'rel.#hits_{M}\n[cm(TC)]', 'rel.#hits_{L}\n[cm(TC)]', 'discriminability #3',
                        'rel.l2\n[sum_{S}(c1)]', 'rel.l2\n[sum_{M}(c1)]', 'rel.l2\n[sum_{L}(c1)]', 'discriminability #4',
                    ],
                    tablefmt="rst", floatfmt="0.6f"));

    print(tabulate(table,
                    headers = ['label', 'name', 'rel. size', 'c0 freq.', 'c1 freq.',
                        'rel.l2\n[sum_{S}(c0)]', 'rel.l2\n[sum_{M}(c0)]', 'rel.l2\n[sum_{L}(c0)]', 'discriminability #1',
                        'rel.#hits_{S}\n[cm(c0)]', 'rel.#hits_{M}\n[cm(c0)]', 'rel.#hits_{L}\n[cm(c0)]', 'discriminability #2',
                        'rel.#hits_{S}\n[cm(TC)]', 'rel.#hits_{M}\n[cm(TC)]', 'rel.#hits_{L}\n[cm(TC)]', 'discriminability #3',
                        'rel.l2\n[sum_{S}(c1)]', 'rel.l2\n[sum_{M}(c1)]', 'rel.l2\n[sum_{L}(c1)]', 'discriminability #4',
                    ],
                    tablefmt="latex", floatfmt="0.2f"));

    df.to_csv(os.path.join(dir, "functional_atlas_stats.csv"));
    df_long.to_csv(os.path.join(dir, "functional_atlas_stats_long_format.csv"));

    return df, df_long



###
### ------------------------------------------------------------------------ ###
def judge_discriminability(frac_s, frac_m, frac_l):
    tot = frac_s + frac_m + frac_l;

    if frac_s > 0.8 * tot:
        return r'$S^{+}$ ';
    if frac_m > 0.8 * tot:
        return r'$M^{+}$ ';
    if frac_l > 0.8 * tot:
        return r'$L^{+}$ ';

    if frac_s > frac_m + frac_l:
        return r'$S^{-}$ ';
    if frac_m > frac_s + frac_l:
        return r'$M^{-}$ ';
    if frac_l > frac_m + frac_s:
        return r'$L^{-}$ ';

    if frac_s < 0.2 * tot:
        return r'$!S^{+}$';
    if frac_m < 0.2 * tot:
        return r'$!M^{+}$';
    if frac_l < 0.2 * tot:
        return r'$!M^{+}$';

    return     " ---    ";


###
### ------------------------------------------------------------------------ ###
def get_subgroup(label, muse_map, SB):
    if label == 0:
        return "BG";
    try:
        subgroup = str(muse_map.loc[muse_map['ROI_INDEX'] == label][SB].iloc[0])
    except:
        print("label {} with subgroup {}".format(label, muse_map.loc[muse_map['ROI_INDEX'] == label][SB]))
    # print("label {} with subgroup {}".format(label, subgroup))
    return subgroup;

###
### ------------------------------------------------------------------------ ###
def accumulate_frequencies(dir, long_df, muse_map, label_map):

    cols_long  = ['label', 'name', 'rel. size', 'c0 freq.', 'c1 freq.',
                  'freq_short', 'freq_mid', 'freq_long', 'discriminability', 'type'
                 ];
    long_df['SUBGROUP_0'] = long_df['label'].apply(lambda x: get_subgroup(x, muse_map, 'iSUBGROUP_0'))
    long_df['SUBGROUP_1'] = long_df['label'].apply(lambda x: get_subgroup(x, muse_map, 'iSUBGROUP_1'))
    long_df['SUBGROUP_2'] = long_df['label'].apply(lambda x: get_subgroup(x, muse_map, 'iSUBGROUP_2'))

    for SB in ['SUBGROUP_0', 'SUBGROUP_1', 'SUBGROUP_2']:
        print()
        print()
        print("SUBGROUP: ", SB)
        S0_s = long_df.groupby([SB, 'type'], as_index=False)['freq_short'].sum().reset_index()
        S0_m = long_df.groupby([SB, 'type'], as_index=False)['freq_mid'].sum().reset_index()
        S0_l = long_df.groupby([SB, 'type'], as_index=False)['freq_long'].sum().reset_index()
        S0_size = long_df.groupby([SB, 'type'], as_index=False)['rel. size'].sum().reset_index()
        S0_c0freq = long_df.groupby([SB, 'type'], as_index=False)['c0 freq.'].sum().reset_index()

        table = [];
        cols  = ['name', 'rel. size', 'c0 freq.',
                'rel.l2[sum_{S}(c0)]', 'rel.l2[sum_{M}(c0)]', 'rel.l2[sum_{L}(c0)]', 'discriminability #1',
                'rel.#hits_{S}[cm(c0)]', 'rel.#hits_{M}[cm(c0)]', 'rel.#hits_{L}[cm(c0)]', 'discriminability #2',
                'rel.#hits_{S}[cm(TC)]', 'rel.#hits_{M}[cm(TC)]', 'rel.#hits_{L}[cm(TC)]', 'discriminability #3',
                'rel.l2[sum_{S}(c1)]', 'rel.l2[sum_{M}(c1)]', 'rel.l2[sum_{L}(c1)]', 'discriminability #4',
                ];
        df = pd.DataFrame(columns=cols);
        for i in range(0,len(S0_s),4):
            name = S0_s.values[i][1]
            ROW = [name, float(S0_size.values[i][3]),
                    float(S0_c0freq.values[i][3]),
                    float(S0_s.values[i][3]), float(S0_m.values[i][3]), float(S0_l.values[i][3]),
                    judge_discriminability(float(S0_s.values[i][3]), float(S0_m.values[i][3]), float(S0_l.values[i][3])),
                    float(S0_s.values[i+3][3]), float(S0_m.values[i+3][3]), float(S0_l.values[i+3][3]),
                    judge_discriminability(float(S0_s.values[i+3][3]), float(S0_m.values[i+3][3]), float(S0_l.values[i+3][3])),
                    float(S0_s.values[i+2][3]), float(S0_m.values[i+2][3]), float(S0_l.values[i+2][3]),
                    judge_discriminability(float(S0_s.values[i+2][3]), float(S0_m.values[i+2][3]), float(S0_l.values[i+2][3])),
                    float(S0_s.values[i+1][3]), float(S0_m.values[i+1][3]), float(S0_l.values[i+1][3]),
                    judge_discriminability(float(S0_s.values[i+1][3]), float(S0_m.values[i+1][3]), float(S0_l.values[i+1][3])),
                    ];
            df = df.append(pd.Series({ c:r for c,r in zip(cols, ROW) }), ignore_index=True);
            table.append(ROW);
        #
        df.to_csv(os.path.join(dir, "MUSE/functional_atlas_stats_"+SB+".csv"));
        print()
        print(tabulate(table,
                        headers = ['label', 'name', 'rel. size', 'c0 freq.', 'c1 freq.',
                            'rel.l2\n[sum_{S}(c0)]', 'rel.l2\n[sum_{M}(c0)]', 'rel.l2\n[sum_{L}(c0)]', 'discriminability #1',
                            'rel.#hits_{S}\n[cm(c0)]', 'rel.#hits_{M}\n[cm(c0)]', 'rel.#hits_{L}\n[cm(c0)]', 'discriminability #2',
                            'rel.#hits_{S}\n[cm(TC)]', 'rel.#hits_{M}\n[cm(TC)]', 'rel.#hits_{L}\n[cm(TC)]', 'discriminability #3',
                            'rel.l2\n[sum_{S}(c1)]', 'rel.l2\n[sum_{M}(c1)]', 'rel.l2\n[sum_{L}(c1)]', 'discriminability #4',
                        ],
                        tablefmt="rst", floatfmt="0.6f"));
        print()
        print(tabulate(table,
                        headers = ['label', 'name', 'rel. size', 'c0 freq.', 'c1 freq.',
                            'rel.l2\n[sum_{S}(c0)]', 'rel.l2\n[sum_{M}(c0)]', 'rel.l2\n[sum_{L}(c0)]', 'discriminability #1',
                            'rel.#hits_{S}\n[cm(c0)]', 'rel.#hits_{M}\n[cm(c0)]', 'rel.#hits_{L}\n[cm(c0)]', 'discriminability #2',
                            'rel.#hits_{S}\n[cm(TC)]', 'rel.#hits_{M}\n[cm(TC)]', 'rel.#hits_{L}\n[cm(TC)]', 'discriminability #3',
                            'rel.l2\n[sum_{S}(c1)]', 'rel.l2\n[sum_{M}(c1)]', 'rel.l2\n[sum_{L}(c1)]', 'discriminability #4',
                        ],
                        tablefmt="latex", floatfmt="0.2f"));


def plot_frequencies(dir, long_df, df, muse_map, label_map):

    cols  = ['label', 'name', 'rel. size', 'c0 freq.', 'c1 freq.',
            'rel.l2[sum_{S}(c0)]', 'rel.l2[sum_{M}(c0)]', 'rel.l2[sum_{L}(c0)]', 'discriminability #1',
            'rel.#hits_{S}[cm(c0)]', 'rel.#hits_{M}[cm(c0)]', 'rel.#hits_{L}[cm(c0)]', 'discriminability #2',
            'rel.#hits_{S}[cm(TC)]', 'rel.#hits_{M}[cm(TC)]', 'rel.#hits_{L}[cm(TC)]', 'discriminability #3',
            'rel.l2[sum_{S}(c1)]', 'rel.l2[sum_{M}(c1)]', 'rel.l2[sum_{L}(c1)]', 'discriminability #4',
            ];


    f = plt.figure(figsize=(18,10))

    df = df[df['name'] != 'undefined/background']
    df = df[:16]

    categories = ['C0 Freq.', 'CM(C0) Freq.', 'CM(TC) Freg.', 'C1 Freq.']
    ind = np.arange(len(df));
    # print(ind)
    for i in range(4):

        if i==1 or i==2:
            continue;

        c1 = '#ff6961cc'
        c2 = '#77dd77cc'
        c3 = '#449afecc'

        c1 = pastGcolrA
        c2 = pastGcolrB
        c3 = pastGcolrC

        # ax = f.add_subplot(4,1,i+1);
        ax = f.add_subplot(2,1,int(i/3)+1);
        totals = [i+j+k for i,j,k in zip(df[cols[5+i*4]], df[cols[6+i*4]], df[cols[7+i*4]])]
        greenBars = [i / j * 100 if j > 0 else 0 for i,j in zip(df[cols[5+i*4]], totals)]
        orangeBars = [i / j * 100 if j > 0 else 0 for i,j in zip(df[cols[6+i*4]], totals)]
        blueBars = [i / j * 100 if j > 0 else 0 for i,j in zip(df[cols[7+i*4]], totals)]
        # plot
        barWidth = 1
        names = np.squeeze(df['name'].values)
        ax.bar(ind, greenBars, color=c1, edgecolor='white', width=barWidth, label='short')
        ax.bar(ind, orangeBars, bottom=greenBars, color=c2, edgecolor='white', width=barWidth, label='mid')
        ax.bar(ind, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color=c3, edgecolor='white', width=barWidth, label='long')
        plt.xticks(ind, [])
        plt.yticks([0,100],['0','100'], size='8')
        ax.set(ylabel=categories[i]);


        if i == 3:
            plt.xticks(ind, names, rotation=90, size='6')

        # else:
        ax2 = ax.twinx();
        ax2.plot(ind, np.squeeze(df['c0 freq.'].values), color='k', linestyle='--',linewidth=1.5, label='f(c0)');
        ax2.plot(ind, np.squeeze(df['c1 freq.'].values), color='white', linestyle='--', linewidth=1.5, label='f(c1)');
        ax2.plot(ind, np.squeeze(df['rel. size'].values), color='gray', linestyle='--', linewidth=1.5, label='size');
        plt.yticks(size='8')

        if i == 0:
            ax.legend(loc='upper right', fontsize='6',)
        if i == 3:
            ax2.legend(loc='upper right', fontsize='6',)

    # if i == 0:
        # handles, labels = ax.get_legend_handles_labels()
        # labels = ["short", "mid", "long"]
        # handles = []
        # labels = []
        # lgd = ax.legend(handles[0:3], labels[0:3], loc='lower right', fontsize='small', handletextpad=0.5)
        # ax.legend(loc='lower right', fontsize='small', handletextpad=0.5)

    plt.tight_layout()
    sns.despine(offset=5, left=True, right=True, top=True, bottom=True)
    # sns.despine()






    # p1 = plt.bar(ind, np.squeeze(df[['rel.l2[sum_{S}(c0)]']].values), width,)
    # p2 = plt.bar(ind, np.squeeze(df[['rel.l2[sum_{M}(c0)]']].values), width, bottom=np.squeeze(df[['rel.l2[sum_{S}(c0)]']].values))
    # p3 = plt.bar(ind, np.squeeze(df[['rel.l2[sum_{L}(c0)]']].values), width, bottom=np.squeeze(df[['rel.l2[sum_{M}(c0)]']].values))
    plt.show()


###
### ------------------------------------------------------------------------ ###
def print_table(brats_data):
    # BIDs = ['Brats18_CBICA_ASG_1', 'Brats18_CBICA_AQZ_1', 'Brats18_TCIA03_257_1','Brats18_CBICA_AQP_1', 'Brats18_CBICA_AQO_1', 'Brats18_2013_2_1', 'Brats18_TCIA05_277_1', 'Brats18_TCIA03_296_1', 'Brats18_TCIA02_606_1', 'Brats18_TCIA01_180_1', 'Brats18_CBICA_ASN_1', 'Brats18_CBICA_AXO_1', 'Brats18_TCIA02_314_1', 'Brats18_TCIA02_135_1', 'Brats18_2013_12_1']
    BIDs = ['Brats18_CBICA_ASN_1', 'Brats18_CBICA_AXO_1',  'Brats18_TCIA02_314_1', 'Brats18_TCIA02_606_1', 'Brats18_TCIA03_257_1', 'Brats18_TCIA03_296_1', 'Brats18_TCIA05_277_1']
    table =   "{:28} & ".format('BraTS ID') \
            + "{:15} & ".format('$\\rho_w$') \
            + "{:15} & ".format('$\\kappa_w$') \
            + "{:15} & ".format('$\\epsilon_{\\ell_2}\\D{O}c(1)$') \
            + "{:15} & ".format('$\\epsilon_{\\ell_2}c(1)|_{TC}$') \
            + "{:15} & ".format('$\\epsilon_{\\ell_2}c(1)$') \
            + "{:15} & ".format('$d|\\vect{x}_{cm}^{\\vect{p}},\\vect{x}_{cm}^{TC}$') \
            + "{:15} & ".format('$d|\\vect{p}_{1},\\vect{p}_2$') \
            + "{:15} & ".format('$\\int_{TC}c(1)$') \
            + "{:15} & ".format('$\\int_{ED}c(1)$') \
            + "{:15} & ".format('$\\int_{B\\setminus WT}c(1)$') \
            + "{:15} & ".format('$n_c$') \
            + "{:15} & ".format('$age[y]$') \
            + "{:15} & ".format('$survival[d]$') \
            + " \\\  "
            # + "{:15} & ".format('$age$')
    # for P in BIDs:
        # row = brats_data.loc[brats_data['BID'] == P];
        # print(row.iloc[0]['BID'])
        # s =   "\\textt{" + "{:20}".format(str(row.iloc[0]['BID'])) + "} & "\
        #     + "{:15.2f}".format(float(row.iloc[0]['rho-inv']))  + " & " \
        #     + "\\num{" + "{:9.3e}".format(float(row.iloc[0]['k-inv']))  + "} & " \
        #     + "\\num{" + "{:9.3e}".format(float(row.iloc[0]['l2[Oc(1)-TC]']))  + "} & " \
        #     + "\\num{" + "{:9.3e}".format(float(row.iloc[0]['l2[c(1)|_TC-TC,scaled]_r']))  + "} & " \
        #     + "\\num{" + "{:9.3e}".format(float(row.iloc[0]['l2[c(1)-TC]']))  + "} & " \
        #     + "${:1.2f}$".format(float(row.iloc[0]['dist[wcm(p|_#c) - cm(TC|_#c)]_(c=0)']))  + " & " \
        #     + "${:1.2f}$".format(float(row.iloc[0]['dist[max1(p) - max2(p)]']))  + " & " \
        #     + "\\num{" + "{:9.3e}".format(float(row.iloc[0]['int{c(1)|_TC}']    / row.iloc[0]['int{c(1)}']))  + "} & " \
        #     + "\\num{" + "{:9.3e}".format(float(row.iloc[0]['int{c(1)|_ED}']    / row.iloc[0]['int{c(1)}']))  + "} & " \
        #     + "\\num{" + "{:9.3e}".format(float(row.iloc[0]['int{c(1)|_B-WT}']  / row.iloc[0]['int{c(1)}']))  + "} & " \
        #     + "${:9d}$ ".format(int(row.iloc[0]['n_comps'])) \
        #     + " \\\ "
        #     # + "${:1d}$ & ".format(str(row['age']))
        # table += "\n" + s
    brats_data = brats_data.sort_values(by=['BID']);
    for index, row in brats_data.iterrows():
        # print(row['BID'])
        s =   "\\texttt{" + "{:20}".format(str(row['BID'])) + "} & "\
            + "{:15.2f}".format(float(row['rho-inv']))  + " & " \
            + "\\num{" + "{:9.3e}".format(float(row['k-inv']))  + "} & " \
            + "\\num{" + "{:9.3e}".format(float(row['l2[Oc(1)-TC]']))  + "} & " \
            + "\\num{" + "{:9.3e}".format(float(row['l2[c(1)|_TC-TC,scaled]_r']))  + "} & " \
            + "\\num{" + "{:9.3e}".format(float(row['l2[c(1)-TC]']))  + "} & " \
            + "${:9.1f}$".format(float(row['dist[wcm(p|_#c) - cm(TC|_#c)]_(c=0)']))  + " & " \
            + "${:9.1f}$".format(float(row['dist[max1(p) - max2(p)]']))  + " & " \
            + "\\num{" + "{:9.3e}".format(float(row['int{c(1)|_TC}']    / row['int{c(1)}']))  + "} & " \
            + "\\num{" + "{:9.3e}".format(float(row['int{c(1)|_ED}']    / row['int{c(1)}']))  + "} & " \
            + "\\num{" + "{:9.3e}".format(float(row['int{c(1)|_B-WT}']  / row['int{c(1)}']))  + "} & " \
            + "${:9d} & ".format(int(row['n_comps'])) \
            + "${:9d} & ".format(int(row['age'])) \
            + "${:9d}$ ".format(int(row['survival(days)'])) \
            + " \\\ "
            # + "${:1d}$ & ".format(str(row['age']))
        table += "\n" + s

    print(table);
    print(len(brats_data))

    print('std:',  brats_data['dist[max1(p) - max2(p)]'].std())
    print('mean:', brats_data['dist[max1(p) - max2(p)]'].mean())



###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    pd.options.display.float_format = '{1.2e}%'.format
    # parse arguments
    basedir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='process BRATS results')
    parser.add_argument ('-x',    '--dir',   type = str, help = 'path to the results folder');
    parser.add_argument ('-file', '--f',     type = str, help = 'path to the csv brats file');
    parser.add_argument ('-labels_file', '--lf',     type = str, help = 'path to the csv file for atlas labels');
    parser.add_argument ('-freq_file1', '--ff1',     type = str, help = 'path to the csv file for frequencies');
    parser.add_argument ('-freq_file2', '--ff2',     type = str, help = 'path to the csv file for frequencies');
    args = parser.parse_args();
    dir = args.dir;
    file = args.f;

    ANALYZE_FREQ = False;
    PLOT_STATS   = False;
    PRINT_TABLE  = True;



    # LABELS_ATLAS_REV = {v:k for k,v in LABELS_ATLAS.items()}
    FILTER = ['Brats18_CBICA_AQJ_1', 'Brats18_TCIA08_242_1', 'Brats18_CBICA_AZD_1', 'Brats18_TCIA02_374_1', 'Brats18_CBICA_ANI_1', 'Brats18_CBICA_AUR_1']
    # read SURVIVAL DATA
    survival_data = pd.read_csv(os.path.join(basedir,"survival_data.csv"), header = 0, error_bad_lines=True, skipinitialspace=True);
    if args.lf is not None:
        atlas_label_mapping = pd.read_csv(os.path.join(basedir, args.lf), header = 0, error_bad_lines=True, skipinitialspace=True);
    if args.ff1 is not None:
        freq_mapping1 = pd.read_csv(os.path.join(basedir, args.ff1), header = 0, error_bad_lines=True, skipinitialspace=True);
    if args.ff2 is not None:
        freq_mapping2 = pd.read_csv(os.path.join(basedir, args.ff2), header = 0, error_bad_lines=True, skipinitialspace=True);
    # read BRATS DATA
    brats_data, weights = read_data(args.f);
    brats_clustering, brats_survival = clean_data(brats_data, filter_GTR=False);


    if PRINT_TABLE:
        print_table(brats_data);
        exit(0)

    # label_dict = atlas_label_mapping.to_dict()
    # LABELS_ATLAS = {}
    # for i in range(len(atlas_label_mapping)):
    #     LABELS_ATLAS[label_dict['ROI_INDEX'][i]] = label_dict['ROI_NAME'][i]
    #     # print("adding dict entry {} : {}".format(label_dict['ROI_INDEX'][i], label_dict['ROI_NAME'][i]))
    # LABELS_ATLAS[0] = "undefined/background"


    accumulate_frequencies(dir, freq_mapping1, atlas_label_mapping, LABELS_ATLAS)
    plot_frequencies(dir, freq_mapping1, freq_mapping2, atlas_label_mapping, LABELS_ATLAS)



    if ANALYZE_FREQ:
        df, df_long = analyze_frequenzies(dir, brats_survival, LABELS_ATLAS);
        # accumulate_frequencies(dir, df_long, atlas_label_mapping, LABELS_ATLAS)



    if PLOT_STATS:
        # sns.set(style="whitegrid")
        mpl.style.use('seaborn')
        sns.set(style='ticks', palette='Set2', context="paper", font_scale=1.6)

        # func_atlas_stats      = pd.read_csv(os.path.join(dir,'functional_atlas_stats.csv'), header = 0, error_bad_lines=True, skipinitialspace=True);
        # func_atlas_stats_long = pd.read_csv(os.path.join(dir,'functional_atlas_stats_long_format.csv'), header = 0, error_bad_lines=True, skipinitialspace=True);
        #
        # func_atlas_stats_long.sort_values("c0 freq.", ascending=False);
        # df_c0    = func_atlas_stats_long.loc[func_atlas_stats_long['type'] == 'c(0)'];
        # df_cm_c0 = func_atlas_stats_long.loc[func_atlas_stats_long['type'] == 'cm[c(0)]'];
        # df_cm_tc = func_atlas_stats_long.loc[func_atlas_stats_long['type'] == 'cm[TC]'];
        # df_c1    = func_atlas_stats_long.loc[func_atlas_stats_long['type'] == 'c(1)'];


        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = [
            r'\usepackage{amssymb}',
            r'\usepackage{amsmath}',
            r'\usepackage{nicefrac}',
            r'\usepackage{xcolor}']
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = 'Computer Modern'

        # ### ----- plot cohort inversion stats ----- ###
        data = brats_clustering;
        data_all = brats_data;

        data["int{c(1)|_TC}_r"] = data["int{c(1)|_TC}"] / data["int{c(1)}"];
        data["int{c(1)|_ED}_r"] = data["int{c(1)|_ED}"] / data["int{c(1)}"];
        data["int{c(1)|_B-WT}_r"] = data["int{c(1)|_B-WT}"] / data["int{c(1)}"];
        data['dist[VE/BG]'] = data["dist[c(0), VE] (aspace)"] / data["dist[c(0), BG] (aspace)"]

        data['vol(ED)/k']   = 1./data["k-inv"] * data["vol(ED)_a"]
        data['vol(TC)/k']   = 1./data["k-inv"] * data["vol(TC)_a"]
        data['vol(TC)/rho'] = 1./data["rho-inv"] * data["vol(TC)_a"]

        # data['vol(TC)_comps'] = data["vol(TC)_a"] / data["n_comps"]
        # data['rho/vol(TC)_comps'] = data["rho-inv"] / data["vol(TC)_comps"]


        # corr_crop = ["survival_class","rho-inv","k-inv","rho-over-k","n_comps", "dist[wcm(p|_#c) - cm(TC|_#c)]_(c=0)","dist[c(0), VE] (aspace)", "dist[c(0), BG] (aspace)","vol(TC)_r","vol(WT)_r","vol(ED)_r",'dist[VE/BG]',"vol(NEC)_r", "age", ]
        # data_corr = data.loc[:,corr_crop]
        # corr_spearman = data_corr.corr(method='spearman')
        # cmap = sns.diverging_palette(220, 10, as_cmap=True)
        # sns.heatmap(corr_spearman, cmap=cmap, vmax=1, vmin=-1, center=0,
        #             square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        # plt.show()
        # sys.exit()

        # crop data
        flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        pal = [sns.color_palette('Paired')[4],sns.color_palette('Paired')[2],sns.color_palette('Paired')[0]]
        stat_crop = ["BID","survival_class","vol(TC)_r","vol(WT)_r","vol(ED)_r","vol(NEC)_r","int{c(1)|_TC}_r", "int{c(1)|_ED}_r", "int{c(1)|_B-WT}_r", "int{c(0)}/int{c(1)}", "l2[Oc(1)-TC]", "l2[c(1)|_TC-TC,scaled]_r"]
        # inv_crop  = ["BID","survival_class","rho-inv","k-inv","rho-over-k","dist[wcm(p|_#c) - cm(TC|_#c)]_(c=0)", "dist[max1(p) - max2(p)]","dist[c(0), VE] (aspace)", "dist[c(0), BG] (aspace)"]
        inv_crop  = ["BID","survival_class","rho-inv","k-inv","rho-over-k","dist[wcm(p|_#c) - cm(TC|_#c)]_(c=0)","dist[c(0), VE] (aspace)", "dist[c(0), BG] (aspace)"]

        stats_dat = data.loc[:,stat_crop]
        inv_dat   = data.loc[:,inv_crop]
        # melt data
        stats_sizes  = pd.melt(stats_dat, id_vars=['BID','survival_class'], value_vars=["vol(TC)_r","vol(ED)_r","vol(WT)_r"],  value_name='stat_val', var_name=['stats_size'])
        stats_inv_1  = pd.melt(stats_dat, id_vars=['BID','survival_class'], value_vars=["int{c(1)|_B-WT}_r", "int{c(1)|_ED}_r", "int{c(1)|_TC}_r"], value_name='stat_val', var_name=['stats_inv_1'])
        # stats_inv_1  = pd.melt(stats_dat, id_vars=['BID','survival_class'], value_vars=["int{c(1)|_TC}_r", "int{c(1)|_ED}_r", "int{c(1)|_B-WT}_r", "int{c(0)}/int{c(1)}", "l2[Oc(1)-TC]", "l2[c(1)|_TC-TC,scaled]_r"], value_name='stat_val', var_name=['stats_inv_1'])
        stats_inv_2   = pd.melt(data,   id_vars=['BID','survival_class'], value_vars=["l2[Oc(1)-TC]", "l2[c(1)|_TC-TC,scaled]_r"], value_name='value', var_name=['stats_inv_2'])
        # stats_inv_22  = pd.melt(data,   id_vars=['BID','survival_class'], value_vars=["dist[wcm(p|_#c) - cm(TC|_#c)]_(c=0)", "dist[max1(p) - max2(p)]"], value_name='value', var_name=['stats_inv_2'])
        stats_inv_23  = pd.melt(data,   id_vars=['BID', 'survival_class'], value_vars=["dist[c(0), VE] (aspace)", "dist[c(0), BG] (aspace)"], value_name='value', var_name=['stats_inv_2'])
        stats_inv_24  = pd.melt(data,   id_vars=['BID', 'survival_class'], value_vars=["dist[VE/BG]"], value_name='value', var_name=['stats_inv_2'])

        stats_inv_25  = pd.melt(data,   id_vars=['BID', 'survival_class'], value_vars=["vol(ED)/k"], value_name='value', var_name=['stats_inv_2'])
        stats_inv_26  = pd.melt(data,   id_vars=['BID', 'survival_class'], value_vars=["vol(TC)/k"], value_name='value', var_name=['stats_inv_2'])
        stats_inv_27  = pd.melt(data,   id_vars=['BID', 'survival_class'], value_vars=["vol(TC)/rho"], value_name='value', var_name=['stats_inv_2'])

        stats_inv_2_all   = pd.melt(data_all,   id_vars=['BID'], value_vars=["l2[Oc(1)-TC]", "l2[c(1)|_TC-TC,scaled]_r"], value_name='value', var_name=['stats_inv_2'])
        # stats_inv_22_all  = pd.melt(data_all,   id_vars=['BID'], value_vars=["dist[wcm(p|_#c) - cm(TC|_#c)]_(c=0)", "dist[max1(p) - max2(p)]"], value_name='value', var_name=['stats_inv_2'])


        stats_inv_rho  = pd.melt(inv_dat,   id_vars=['BID','survival_class'], value_vars=["rho-inv"],         value_name='value', var_name=['inv_vars'])
        stats_inv_k    = pd.melt(inv_dat,   id_vars=['BID','survival_class'], value_vars=["k-inv"],           value_name='value', var_name=['inv_vars'])
        stats_inv_rhok = pd.melt(inv_dat,   id_vars=['BID','survival_class'], value_vars=["rho-over-k"],      value_name='value', var_name=['inv_vars'])

        # figure
        # f = plt.figure(figsize=(5,5))
        # grid1 = plt.GridSpec(4, 1, wspace=0.2, hspace=0.4, height_ratios=[4,4,1,1])

        # boxplot settings
        boxprops    = {'edgecolor': 'w', 'linewidth': 1}
        lineprops   = {'color': 'k', 'linewidth': 1}
        medianprops = {'color': 'r', 'linewidth': 2}
        kwargs      = {'palette': pal}
        boxplot_kwargs = dict({'boxprops': boxprops, 'medianprops': medianprops,
                               'whiskerprops': lineprops, 'capprops': lineprops,
                               'width': 0.75, 'palette' : pal, 'notch' : 0},
                               **kwargs)
        stripplot_kwargs = dict({'linewidth': 0.2, 'size': 3, 'alpha': 0.1, 'color' : [0.1,0.1,0.1], 'edgecolor' : 'w'}) #[0.1,0.1,0.1]

        if True:

            f = plt.figure(figsize=(5,3))
            ax = f.add_subplot(1,1,1);
            sns.boxplot(  x="stat_val", y="stats_size", hue="survival_class", data=stats_sizes, fliersize=0, ax=ax, **boxplot_kwargs)
            sns.stripplot(x='stat_val', y='stats_size', hue='survival_class', data=stats_sizes, jitter=True, split=True, ax=ax, **stripplot_kwargs)
            # fix legend
            handles, labels = ax.get_legend_handles_labels()
            labels = ["short", "mid", "long"]
            lgd = ax.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)
            lgd.legendHandles[0]._sizes = [40]
            lgd.legendHandles[1]._sizes = [40]
            ax.xaxis.grid(True);
            ax.set(xlabel="");
            ax.set(ylabel="");
            ax.set_yticklabels(["$vol_{(TC)_r}$", "$vol_{(ED)_r}$", "$vol_{(WT)_r}$"]);
            plt.tight_layout()
            sns.despine(offset=5, left=True, right=True, top=True, bottom=True)

            f = plt.figure(figsize=(5,3))
            ax = f.add_subplot(1,1,1);
            sns.boxplot(  x="stat_val", y="stats_inv_1", hue="survival_class", data=stats_inv_1, fliersize=0, ax=ax, **boxplot_kwargs)
            sns.stripplot(x='stat_val', y='stats_inv_1', hue='survival_class', data=stats_inv_1, jitter=True, split=True, ax=ax, **stripplot_kwargs)
            # fix legend
            handles, labels = ax.get_legend_handles_labels()
            labels = ["short", "mid", "long"]
            handles = []
            labels = []
            lgd = ax.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)
            # lgd.legendHandles[0]._sizes = [40]
            # lgd.legendHandles[1]._sizes = [40]
            ax.xaxis.grid(True);
            ax.set(xlabel="");
            ax.set(ylabel="");
            ax.set_yticklabels(["$\\int_{B\\setminus WT}c_1$", "$\\int_{ED}c_1$", "$\\int_{TC}c_1$"])#, "$\\int{c(0)}$", "$\\mu_{\\mathcal{O}c,L^2}$", "$\\mu_{c|_{TC},L^2}$"]);
            plt.tight_layout()
            sns.despine(offset=5, left=True, right=True, top=True, bottom=True)

            f = plt.figure(figsize=(5,3))
            ax = f.add_subplot(1,1,1);
            sns.boxplot(  x="value", y="inv_vars", hue="survival_class", data=stats_inv_rho, fliersize=0, ax=ax, **boxplot_kwargs)
            sns.stripplot(x='value', y='inv_vars', hue='survival_class', data=stats_inv_rho, jitter=True, split=True, ax=ax, **stripplot_kwargs)
            # fix legend
            handles, labels = ax.get_legend_handles_labels()
            labels = ["short", "mid", "long"]
            handles = []
            labels = []
            lgd = ax.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)
            # lgd.legendHandles[0]._sizes = [40]
            # lgd.legendHandles[1]._sizes = [40]
            ax.xaxis.grid(True);
            ax.set(xlabel="");
            ax.set(ylabel="");
            ax.set_yticklabels(["$\\rho_w$"])
            plt.tight_layout()
            sns.despine(offset=5, left=True, right=True, top=True, bottom=True)

            f = plt.figure(figsize=(5,3))
            ax = f.add_subplot(1,1,1);
            sns.boxplot(  x="value", y="inv_vars", hue="survival_class", data=stats_inv_rhok, fliersize=0, ax=ax, **boxplot_kwargs)
            sns.stripplot(x='value', y='inv_vars', hue='survival_class', data=stats_inv_rhok, jitter=True, split=True, ax=ax, **stripplot_kwargs)
            # fix legend
            handles, labels = ax.get_legend_handles_labels()
            labels = ["short", "mid", "long"]
            handles = []
            labels = []
            lgd = ax.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)
            # lgd.legendHandles[0]._sizes = [40]
            # lgd.legendHandles[1]._sizes = [40]
            ax.xaxis.grid(True);
            # ax.set_xscale('log')
            # ax.set_xlim([1E2,1e4])
            ax.set(xlabel="");
            ax.set(ylabel="");
            ax.set_yticklabels(["$\\nicefrac{\\rho_w}{\\kappa_w}$"])
            plt.tight_layout()
            sns.despine(offset=5, left=True, right=True, top=True, bottom=True)

            f = plt.figure(figsize=(5,3))
            ax = f.add_subplot(1,1,1);
            sns.boxplot(  x="value", y="inv_vars", hue="survival_class", data=stats_inv_k, fliersize=0, ax=ax, **boxplot_kwargs)
            sns.stripplot(x='value', y='inv_vars', hue='survival_class', data=stats_inv_k, jitter=True, split=True, ax=ax, **stripplot_kwargs)
            # fix legend
            handles, labels = ax.get_legend_handles_labels()
            labels = ["short", "mid", "long"]
            handles = []
            labels = []
            lgd = ax.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)
            # lgd.legendHandles[0]._sizes = [40]
            # lgd.legendHandles[1]._sizes = [40]
            ax.xaxis.grid(True);
            ax.set_xscale('log')
            ax.set_xlim([1e-3,1e-1])
            ax.set(xlabel="");
            ax.set(ylabel="");
            ax.set_yticklabels(["$\\kappa_w$"])
            plt.tight_layout()
            sns.despine(offset=5, left=True, right=True, top=True, bottom=True)


            f = plt.figure(figsize=(5,3))
            ax = f.add_subplot(1,1,1);
            sns.boxplot(  x="value", y="stats_inv_2",  data=stats_inv_2_all, fliersize=1, ax=ax, **boxplot_kwargs)
            sns.stripplot(x='value', y='stats_inv_2',  data=stats_inv_2_all, jitter=True, split=True, ax=ax, **stripplot_kwargs)
            ax.xaxis.grid(True);
            ax.set(xlabel="");
            ax.set(ylabel="");
            ax.set_yticklabels(["$\\mu_{\\mathcal{O}c,L^2}$", "$\\mu_{c|_{TC},L^2}$"])
            plt.tight_layout()
            sns.despine(offset=5, left=True, right=True, top=True, bottom=True)

            # f = plt.figure(figsize=(5,3))
            # ax = f.add_subplot(1,1,1);
            # sns.boxplot(  x="value", y="stats_inv_2",  data=stats_inv_22_all, fliersize=1, ax=ax, **boxplot_kwargs)
            # sns.stripplot(x='value', y='stats_inv_2',  data=stats_inv_22_all, jitter=True, split=True, ax=ax, **stripplot_kwargs)
            # ax.xaxis.grid(True);
            # ax.set(xlabel="");
            # ax.set(ylabel="");
            # ax.set_yticklabels(["$D_{cm}$", "$D_{p}$"])


            f = plt.figure(figsize=(5,3))
            ax = f.add_subplot(1,1,1);
            sns.boxplot(  x="value", y="stats_inv_2", hue="survival_class", data=stats_inv_23, fliersize=1, ax=ax, **boxplot_kwargs)
            sns.stripplot(x='value', y='stats_inv_2', hue="survival_class", data=stats_inv_23, jitter=True, split=True, ax=ax, **stripplot_kwargs)
            ax.xaxis.grid(True);
            # fix legend
            handles, labels = ax.get_legend_handles_labels()
            labels = ["short", "mid", "long"]
            handles = []
            labels = []
            lgd = ax.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)
            # lgd.legendHandles[0]._sizes = [40]
            # lgd.legendHandles[1]._sizes = [40]
            ax.xaxis.grid(True);
            ax.set(xlabel="");
            ax.set(ylabel="");
            ax.set_yticklabels(["$D_{VE}$", "$D_{\\partial\\mathcal{B}}$"])
            plt.tight_layout()
            sns.despine(offset=5, left=True, right=True, top=True, bottom=True)

            f = plt.figure(figsize=(5,3))
            ax = f.add_subplot(1,1,1);
            sns.boxplot(  x="value", y="stats_inv_2", hue="survival_class", data=stats_inv_24, fliersize=1, ax=ax, **boxplot_kwargs)
            sns.stripplot(x='value', y='stats_inv_2', hue="survival_class", data=stats_inv_24, jitter=True, split=True, ax=ax, **stripplot_kwargs)
            ax.xaxis.grid(True);
            # fix legend
            ax.set_xscale('log')
            handles, labels = ax.get_legend_handles_labels()
            labels = ["short", "mid", "long"]
            handles = []
            labels = []
            lgd = ax.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)
            # lgd.legendHandles[0]._sizes = [40]
            # lgd.legendHandles[1]._sizes = [40]
            ax.xaxis.grid(True);
            ax.set(xlabel="");
            ax.set(ylabel="");
            ax.set_yticklabels([ "$\\nicefrac{D_{VE}}{D_{\\partial\\mathcal{B}}}$"])
            plt.tight_layout()
            sns.despine(offset=5, left=True, right=True, top=True, bottom=True)

        f = plt.figure(figsize=(5,3))
        ax = f.add_subplot(1,1,1);
        sns.boxplot(  x="value", y="stats_inv_2", hue="survival_class", data=stats_inv_25, fliersize=1, ax=ax, **boxplot_kwargs)
        sns.stripplot(x='value', y='stats_inv_2', hue="survival_class", data=stats_inv_25, jitter=True, split=True, ax=ax, **stripplot_kwargs)
        ax.xaxis.grid(True);
        # fix legend
        ax.set_xscale('log')
        handles, labels = ax.get_legend_handles_labels()
        labels = ["short", "mid", "long"]
        handles = []
        labels = []
        lgd = ax.legend(handles[0:3], labels[0:3], loc='upper left', fontsize='small', handletextpad=0.5)
        # lgd.legendHandles[0]._sizes = [40]
        # lgd.legendHandles[1]._sizes = [40]
        ax.xaxis.grid(True);
        ax.set(xlabel="");
        ax.set(ylabel="");
        ax.set_yticklabels([ "$\\nicefrac{vol(ED)}{\\kappa_w}$"])
        plt.tight_layout()
        sns.despine(offset=5, left=True, right=True, top=True, bottom=True)

        f = plt.figure(figsize=(5,3))
        ax = f.add_subplot(1,1,1);
        sns.boxplot(  x="value", y="stats_inv_2", hue="survival_class", data=stats_inv_26, fliersize=1, ax=ax, **boxplot_kwargs)
        sns.stripplot(x='value', y='stats_inv_2', hue="survival_class", data=stats_inv_26, jitter=True, split=True, ax=ax, **stripplot_kwargs)
        ax.xaxis.grid(True);
        # fix legend
        ax.set_xscale('log')
        handles, labels = ax.get_legend_handles_labels()
        labels = ["short", "mid", "long"]
        handles = []
        labels = []
        lgd = ax.legend(handles[0:3], labels[0:3], loc='upper left', fontsize='small', handletextpad=0.5)
        # lgd.legendHandles[0]._sizes = [40]
        # lgd.legendHandles[1]._sizes = [40]
        ax.xaxis.grid(True);
        ax.set(xlabel="");
        ax.set(ylabel="");
        ax.set_yticklabels([ "$\\nicefrac{vol(TC)}{\\kappa_w}$"])
        plt.tight_layout()
        sns.despine(offset=5, left=True, right=True, top=True, bottom=True)

        f = plt.figure(figsize=(5,3))
        ax = f.add_subplot(1,1,1);
        sns.boxplot(  x="value", y="stats_inv_2", hue="survival_class", data=stats_inv_27, fliersize=1, ax=ax, **boxplot_kwargs)
        sns.stripplot(x='value', y='stats_inv_2', hue="survival_class", data=stats_inv_27, jitter=True, split=True, ax=ax, **stripplot_kwargs)
        ax.xaxis.grid(True);
        # fix legend
        ax.set_xscale('log')
        handles, labels = ax.get_legend_handles_labels()
        labels = ["short", "mid", "long"]
        handles = []
        labels = []
        lgd = ax.legend(handles[0:3], labels[0:3], loc='upper left', fontsize='small', handletextpad=0.5)
        # lgd.legendHandles[0]._sizes = [40]
        # lgd.legendHandles[1]._sizes = [40]
        ax.xaxis.grid(True);
        ax.set(xlabel="");
        ax.set(ylabel="");
        ax.set_yticklabels([ "$\\nicefrac{vol(TC)}{\\rho_w}$"])
        plt.tight_layout()
        sns.despine(offset=5, left=True, right=True, top=True, bottom=True)


        pal = [sns.color_palette('Paired')[4],sns.color_palette('Paired')[2],sns.color_palette('Paired')[0]]

        f = plt.figure(figsize=(5,3))
        ax = f.add_subplot(1,1,1);
        sns.scatterplot(x="k-inv", y="vol(TC)_r", hue="survival_class", size="vol(ED)_r", palette=pal, data=data)
        ax.set_xscale('log')
        handles, labels = ax.get_legend_handles_labels()
        labels = ["short", "mid", "long"]
        handles = []
        labels = []
        lgd = ax.legend(handles[0:3], labels[0:3], loc='upper left', fontsize='small', handletextpad=0.5)
        # lgd.legendHandles[0]._sizes = [40]
        # lgd.legendHandles[1]._sizes = [40]
        ax.xaxis.grid(True);
        ax.set(xlabel="$\\kappa_w$");
        ax.set(ylabel="$vol(TC)$");
        # ax.set_yticklabels([ "$vol(TC)$"])
        # ax.set_xticklabels([ "$\\kappa_w$"])
        plt.tight_layout()
        sns.despine(offset=5, left=True, right=True, top=True, bottom=True)

        f = plt.figure(figsize=(5,3))
        ax = f.add_subplot(1,1,1);
        sns.scatterplot(x="k-inv", y="vol(ED)_r", hue="survival_class", palette=pal, data=data)
        ax.set_xscale('log')
        handles, labels = ax.get_legend_handles_labels()
        labels = ["short", "mid", "long"]
        handles = []
        labels = []
        lgd = ax.legend(handles[0:3], labels[0:3], loc='upper left', fontsize='small', handletextpad=0.5)
        # lgd.legendHandles[0]._sizes = [40]
        # lgd.legendHandles[1]._sizes = [40]
        ax.xaxis.grid(True);
        ax.set(xlabel="$\\kappa_w$");
        ax.set(ylabel="$vol(ED)$");
        # ax.set_yticklabels([ "$vol(ED)$"])
        # ax.set_xticklabels([ "$\\kappa_w$"])
        plt.tight_layout()
        sns.despine(offset=5, left=True, right=True, top=True, bottom=True)

        f = plt.figure(figsize=(5,3))
        ax = f.add_subplot(1,1,1);
        sns.scatterplot(x="rho-inv", y="vol(TC)_r", hue="survival_class", size="n_comps", palette=pal, data=data, legend=False)
        # ax.set_yscale('log')
        # handles, labels = ax.get_legend_handles_labels()
        # labels = ["short", "mid", "long"]
        # lgd = ax.legend(handles[0:3], labels[0:3], loc='upper left', fontsize='small', handletextpad=0.5)
        # lgd.legendHandles[0]._sizes = [40]
        # lgd.legendHandles[1]._sizes = [40]
        ax.xaxis.grid(True);
        ax.set(xlabel="$\\rho_w$");
        ax.set(ylabel="$vol(TC)$");
        # ax.set_yticklabels([ "$vol(TC)$"])
        # ax.set_xticklabels([ "$\\rho_w$"])
        plt.tight_layout()
        sns.despine(offset=5, left=True, right=True, top=True, bottom=True)




        plt.tight_layout()
        sns.despine(offset=5, left=True, right=True, top=True, bottom=True)
        plt.show()



    # select features
    # X_ib, Y_ib, cols = get_feature_subset(brats_survival, type=["image_based", "physics_based"], purpose='prediction', estimator='classifier');



































# ### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #
# ### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #
# ### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ### #


# Accumulated c(0) and c(1) frequencies per label of the functional atlas, subdivided into survival classes SHOT, MID, LONG:
# =======  ===========================  ===========  ==========  ==========  ===============  ===============  ===============  ==================  ===============  ===============  ===============  ==================  ===============  ===============  ===============  ==================  ===============  ===============  ===============  ==================
#   label  name                           rel. size    c0 freq.    c1 freq.           rel.l2           rel.l2           rel.l2  discriminability      rel.#hits_{S}    rel.#hits_{M}    rel.#hits_{L}  discriminability      rel.#hits_{S}    rel.#hits_{M}    rel.#hits_{L}  discriminability             rel.l2           rel.l2           rel.l2  discriminability
#                                                                              [sum_{S}(c0)]    [sum_{M}(c0)]    [sum_{L}(c0)]                             [cm(c0)]         [cm(c0)]         [cm(c0)]                             [cm(TC)]         [cm(TC)]         [cm(TC)]                        [sum_{S}(c1)]    [sum_{M}(c1)]    [sum_{L}(c1)]
# =======  ===========================  ===========  ==========  ==========  ===============  ===============  ===============  ==================  ===============  ===============  ===============  ==================  ===============  ===============  ===============  ==================  ===============  ===============  ===============  ==================
     # 30  unlabelled white matter         0.206968    0.582038    0.737011         0.396205         0.253626         0.328654  uninformative              0.164557         0.088608         0.151899  uninformative              0.183544         0.088608         0.145570  uninformative              0.403766         0.198074         0.181347  short (weak)
     # 17  superior_temporal_gyrus         0.037368    0.349364    0.259545         0.209397         0.239195         0.122191  uninformative              0.044304         0.025316         0.025316  uninformative              0.037975         0.025316         0.018987  uninformative              0.145061         0.063429         0.071373  short (weak)
     #  1  superior_frontal_gyrus          0.088114    0.300364    0.119337         0.114332         0.105733         0.256487  long  (weak)               0.018987         0.006329         0.025316  not mid   (strong)         0.025316         0.006329         0.018987  not mid   (strong)         0.058152         0.038306         0.045811  uninformative
     # 18  middle_temporal_gyrus           0.031556    0.273342    0.248740         0.162666         0.058584         0.206777  not mid   (strong)         0.031646         0.006329         0.063291  long  (weak)               0.031646         0.000000         0.075949  long  (weak)               0.127587         0.057534         0.081481  uninformative
     # 19  inferior_temporal_gyrus         0.029945    0.223427    0.159816         0.132816         0.046682         0.172340  not mid   (strong)         0.006329         0.006329         0.018987  long  (weak)               0.006329         0.006329         0.006329  uninformative              0.088166         0.031584         0.061513  not mid   (strong)
     #  2  middle_frontal_gyrus            0.081404    0.203066    0.144171         0.108755         0.125438         0.115982  uninformative              0.012658         0.018987         0.012658  uninformative              0.012658         0.018987         0.012658  uninformative              0.060533         0.059055         0.044219  uninformative
     #  9  superior_parietal_gyrus         0.037282    0.193650    0.186019         0.110738         0.120124         0.095971  uninformative              0.018987         0.012658         0.006329  not long  (strong)         0.018987         0.018987         0.012658  uninformative              0.099092         0.068690         0.053466  uninformative
     # 31  ventricles                      0.008236    0.175940    0.159045         0.131007         0.086525         0.056884  uninformative              0.018987         0.000000         0.000000  short (strong)             0.012658         0.000000         0.012658  not mid   (strong)         0.091157         0.042609         0.033748  short (weak)
     # 27  hippocampus                     0.005559    0.156318    0.140001         0.132378         0.066382         0.040985  short (weak)               0.018987         0.006329         0.006329  short (weak)               0.018987         0.006329         0.000000  short (weak)               0.087483         0.033107         0.029382  short (weak)
     # 12  precuneus                       0.014327    0.138529    0.091777         0.100090         0.028873         0.089469  not mid   (strong)         0.000000         0.000000         0.006329  long  (strong)             0.006329         0.000000         0.006329  not mid   (strong)         0.054415         0.026861         0.029260  uninformative
     #  4  precentral_gyrus                0.039586    0.134846    0.077934         0.051088         0.116819         0.043893  mid   (weak)               0.000000         0.006329         0.006329  not short (strong)         0.000000         0.006329         0.012658  long  (weak)               0.034600         0.027963         0.031919  uninformative
     # 23  insular_cortex                  0.009557    0.130899    0.159114         0.116514         0.040843         0.033352  short (weak)               0.012658         0.006329         0.000000  short (weak)               0.006329         0.012658         0.000000  mid   (weak)               0.085077         0.054537         0.031037  not long  (strong)
     # 20  parahippocampal_gyrus           0.011645    0.125105    0.126145         0.085586         0.075907         0.047812  uninformative              0.012658         0.000000         0.006329  short (weak)               0.000000         0.000000         0.006329  long  (strong)             0.073015         0.033737         0.027510  short (weak)
     # 13  superior_occipital_gyrus        0.016805    0.124211    0.068526         0.112532         0.037575         0.036719  short (weak)               0.006329         0.006329         0.000000  not long  (strong)         0.006329         0.006329         0.006329  uninformative              0.040260         0.019777         0.015989  short (weak)
     # 11  angular_gyrus                   0.021301    0.118202    0.116513         0.034359         0.082938         0.075944  not short (strong)         0.000000         0.006329         0.000000  mid   (strong)             0.000000         0.006329         0.000000  mid   (strong)             0.049172         0.035217         0.041102  uninformative
     # 24  cingulate_gyrus                 0.018288    0.105206    0.112187         0.074104         0.039513         0.063324  uninformative              0.006329         0.006329         0.006329  uninformative              0.012658         0.006329         0.006329  uninformative              0.060610         0.030202         0.031854  uninformative
     #  8  postcentral_gyrus               0.030802    0.104555    0.105252         0.091514         0.046808         0.011922  short (weak)               0.006329         0.006329         0.000000  not long  (strong)         0.000000         0.006329         0.000000  mid   (strong)             0.049713         0.033626         0.037154  uninformative
     # 22  fusiform_gyrus                  0.017746    0.102209    0.142704         0.060218         0.010648         0.081527  long  (weak)               0.006329         0.000000         0.000000  short (strong)             0.006329         0.000000         0.000000  short (strong)             0.062977         0.027086         0.061009  not mid   (strong)
     #  3  inferior_frontal_gyrus          0.029796    0.092831    0.080281         0.000683         0.064126         0.067118  long  (weak)               0.000000         0.006329         0.000000  mid   (strong)             0.000000         0.006329         0.000000  mid   (strong)             0.027734         0.040903         0.021831  uninformative
     # 14  middle_occipital_gyrus          0.031222    0.091529    0.091428         0.060413         0.056355         0.039395  uninformative              0.000000         0.000000         0.000000  uninformative              0.006329         0.000000         0.000000  short (strong)             0.061610         0.016064         0.024251  short (weak)
     # 21  lingual_gyrus                   0.023024    0.090077    0.076965         0.047628         0.000765         0.076407  long  (weak)               0.000000         0.000000         0.006329  long  (strong)             0.000000         0.000000         0.006329  long  (strong)             0.046615         0.014545         0.020342  short (weak)
     # 25  caudate                         0.005789    0.088155    0.103215         0.018472         0.071595         0.030321  mid   (weak)               0.000000         0.006329         0.006329  not short (strong)         0.000000         0.012658         0.000000  mid   (strong)             0.048357         0.040409         0.020203  not long  (strong)
     # 10  supramarginal_gyrus             0.015521    0.087378    0.084332         0.000496         0.059871         0.063014  long  (weak)               0.000000         0.012658         0.006329  mid   (weak)               0.000000         0.006329         0.006329  not short (strong)         0.028631         0.022415         0.037454  uninformative
     # 26  putamen                         0.005594    0.083717    0.138726         0.025044         0.078847         0.012728  mid   (weak)               0.000000         0.018987         0.000000  mid   (strong)             0.000000         0.012658         0.000000  mid   (strong)             0.074848         0.046319         0.027034  short (weak)
     #  0  background                      4.535199    0.075555    0.060986         0.011029         0.023112         0.071083  long  (weak)               0.000000         0.000000         0.000000  uninformative              0.000000         0.000000         0.000000  uninformative              0.034883         0.022665         0.031777  uninformative
     # 15  inferior_occipital_gyrus        0.014321    0.049519    0.034351         0.047587         0.000000         0.013695  short (weak)               0.006329         0.000000         0.000000  short (strong)             0.000000         0.000000         0.000000  uninformative              0.017336         0.000912         0.022004  long  (weak)
     # 28  cerebellum                      0.118459    0.034933    0.012569         0.000003         0.000000         0.034933  long  (strong)             0.000000         0.000000         0.000000  uninformative              0.000000         0.000000         0.000000  uninformative              0.002741         0.001290         0.011313  long  (weak)
     # 29  brainstem                       0.014992    0.031735    0.040730         0.000035         0.027796         0.015309  mid   (weak)               0.000000         0.000000         0.000000  uninformative              0.000000         0.000000         0.000000  uninformative              0.022712         0.014855         0.008660  not long  (strong)
     #  6  lateral_orbitofrontal_gyrus     0.007328    0.015762    0.044055         0.015746         0.000001         0.000708  short (strong)             0.000000         0.000000         0.000000  uninformative              0.000000         0.000000         0.000000  uninformative              0.024528         0.010898         0.012832  short (weak)
     #  5  middle_orbitofrontal_gyrus      0.014292    0.005154    0.049462         0.003112         0.000000         0.004108  long  (weak)               0.000000         0.000000         0.000000  uninformative              0.000000         0.000000         0.000000  uninformative              0.029708         0.008319         0.015591  short (weak)
     #  7  gyrus_rectus                    0.004494    0.001094    0.017367         0.001094         0.000000         0.000000  short (strong)             0.000000         0.000000         0.000000  uninformative              0.000000         0.000000         0.000000  uninformative              0.015879         0.001001         0.001435  short (strong)
     # 16  cuneus                          0.008677    0.000963    0.018549         0.000963         0.000000         0.000000  short (strong)             0.000000         0.000000         0.000000  uninformative              0.000000         0.000000         0.000000  uninformative              0.015644         0.003089         0.002335  short (weak)
# =======  ===========================  ===========  ==========  ==========  ===============  ===============  ===============  ==================  ===============  ===============  ===============  ==================  ===============  ===============  ===============  ==================  ===============  ===============  ===============  ==================
