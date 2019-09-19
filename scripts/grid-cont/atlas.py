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


###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    pd.options.display.float_format = '{1.2e}%'.format
    # parse arguments
    basedir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='process BRATS results')
    parser.add_argument ('-x',    '--dir',   type = str, help = 'path to the results folder');
    args = parser.parse_args();

    dir = args.dir;
    FILTER = ['Brats18_CBICA_AQJ_1', 'Brats18_TCIA08_242_1', 'Brats18_CBICA_AZD_1', 'Brats18_TCIA02_374_1', 'Brats18_CBICA_ANI_1', 'Brats18_CBICA_AUR_1']
    survival_data = pd.read_csv(os.path.join(basedir,"survival_data.csv"), header = 0, error_bad_lines=True, skipinitialspace=True);

    # load atlas
    atlas = nib.load(os.path.join(dir,"jakob_segmented_with_cere_lps_240x240x155_in_brats_hdr.nii.gz"))
    affine = atlas.affine;
    shape = atlas.shape;
    atlas = atlas.get_fdata();
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
            c1_in_aspace = nib.load(os.path.join(patient_path, "c1Recon_256x256x256_aff2jakob_in_Aspace_240x240x155.nii.gz")).get_fdata();

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

        # compute max
        max_c0_atlas   = np.amax(glioma_c0_atlas.flatten())
        max_c0_atlas_s = np.amax(glioma_c0_atlas_short.flatten())
        max_c0_atlas_m = np.amax(glioma_c0_atlas_mid.flatten())
        max_c0_atlas_l = np.amax(glioma_c0_atlas_long.flatten())
        max_c0_atlas_n = np.amax(glioma_c0_atlas_na.flatten())
        max_c1_atlas   = np.amax(glioma_c1_atlas.flatten())
        max_c1_atlas_s = np.amax(glioma_c1_atlas_short.flatten())
        max_c1_atlas_m = np.amax(glioma_c1_atlas_mid.flatten())
        max_c1_atlas_l = np.amax(glioma_c1_atlas_long.flatten())
        max_c1_atlas_n = np.amax(glioma_c1_atlas_na.flatten())
        print("max value of glioma c(0) atlas: {}".format(max_c0_atlas));
        print("max value of glioma c(1) atlas: {}".format(max_c1_atlas));
        # rescale
        glioma_c0_atlas       /= max_c0_atlas;
        glioma_c0_atlas_short /= max_c0_atlas_s;
        glioma_c0_atlas_mid   /= max_c0_atlas_m;
        glioma_c0_atlas_long  /= max_c0_atlas_l;
        glioma_c0_atlas_na    /= max_c0_atlas_n;
        glioma_c1_atlas       /= max_c1_atlas;
        glioma_c1_atlas_short /= max_c1_atlas_s;
        glioma_c1_atlas_mid   /= max_c1_atlas_m;
        glioma_c1_atlas_long  /= max_c1_atlas_l;
        glioma_c1_atlas_na    /= max_c1_atlas_n;
        glioma_c0_atlas        = np.abs(glioma_c0_atlas);
        glioma_c0_atlas_short  = np.abs(glioma_c0_atlas);
        glioma_c0_atlas_mid    = np.abs(glioma_c0_atlas);
        glioma_c0_atlas_long   = np.abs(glioma_c0_atlas);
        glioma_c0_atlas_na     = np.abs(glioma_c0_atlas);
        glioma_c1_atlas        = np.abs(glioma_c1_atlas);
        glioma_c1_atlas_short  = np.abs(glioma_c1_atlas);
        glioma_c1_atlas_mid    = np.abs(glioma_c1_atlas);
        glioma_c1_atlas_long   = np.abs(glioma_c1_atlas);
        glioma_c1_atlas_na     = np.abs(glioma_c1_atlas);
        print("max value of glioma c(0) atlas after rescaling: {}".format(np.amax(glioma_c0_atlas.flatten())));
        print("max value of glioma c(1) atlas after rescaling: {}".format(np.amax(glioma_c1_atlas.flatten())));
        # write out
        print("write nii.gz files")
        fio.writeNII(glioma_c0_atlas, os.path.join(dir,'glioma_c0_atlas_plain.nii.gz'), affine);
        fio.writeNII(glioma_c0_atlas_short, os.path.join(dir,'glioma_c0_atlas_short.nii.gz'), affine);
        fio.writeNII(glioma_c0_atlas_mid, os.path.join(dir,'glioma_c0_atlas_mid.nii.gz'), affine);
        fio.writeNII(glioma_c0_atlas_long, os.path.join(dir,'glioma_c0_atlas_long.nii.gz'), affine);
        fio.writeNII(glioma_c0_atlas_na, os.path.join(dir,'glioma_c0_atlas_na.nii.gz'), affine);
        fio.writeNII(glioma_c1_atlas, os.path.join(dir,'glioma_c1_atlas_plain.nii.gz'), affine);
        fio.writeNII(glioma_c1_atlas_short, os.path.join(dir,'glioma_c1_atlas_short.nii.gz'), affine);
        fio.writeNII(glioma_c1_atlas_mid, os.path.join(dir,'glioma_c1_atlas_mid.nii.gz'), affine);
        fio.writeNII(glioma_c1_atlas_long, os.path.join(dir,'glioma_c1_atlas_long.nii.gz'), affine);
        fio.writeNII(glioma_c1_atlas_na, os.path.join(dir,'glioma_c1_atlas_na.nii.gz'), affine);
        print("write nc files")
        fio.createNetCDF(os.path.join(dir,'glioma_c0_atlas_plain.nc') , np.shape(glioma_c0_atlas), np.swapaxes(glioma_c0_atlas,0,2));
        fio.createNetCDF(os.path.join(dir,'glioma_c0_atlas_short.nc') , np.shape(glioma_c0_atlas_short), np.swapaxes(glioma_c0_atlas_short,0,2));
        fio.createNetCDF(os.path.join(dir,'glioma_c0_atlas_mid.nc') ,   np.shape(glioma_c0_atlas_mid), np.swapaxes(glioma_c0_atlas_mid,0,2));
        fio.createNetCDF(os.path.join(dir,'glioma_c0_atlas_long.nc') ,  np.shape(glioma_c0_atlas_long), np.swapaxes(glioma_c0_atlas_long,0,2));
        fio.createNetCDF(os.path.join(dir,'glioma_c0_atlas_na.nc') ,    np.shape(glioma_c0_atlas_na), np.swapaxes(glioma_c0_atlas_na,0,2));
