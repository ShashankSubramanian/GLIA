import matplotlib as mpl
mpl.use('Agg')
import os, sys, warnings, argparse, subprocess
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../common/'))
from shutil import copyfile
import scipy
from scipy.spatial import distance
import nibabel as nib
import numpy as np
import nibabel as nib
import pandas as pd
import shutil
import imageTools as imgtools
import file_io as fio
import utils as ut
import random

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize


###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    # repository base directory
    basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
    basedir_real = os.path.dirname(os.path.realpath(__file__));
    # parse arguments
    parser = argparse.ArgumentParser(description='Process input images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument ('-patient_labels', '--patient_segmentation_labels', type=str,   help = 'comma separated patient segmented image labels. for ex.\n  0=bg,1=nec,2=ed,4=enh,5=wm,6=gm,7=vt,8=csf\n for BRATS type segmentation. DISCLAIMER vt and every extra label mentioned will be merged with csf');
    parser.add_argument ('-x', '--results_directory',                        type = str, help = 'path to destination')
    parser.add_argument ('-base_dir',                                        type = str, help = 'path to predictions')
    parser.add_argument ('-csv_path', type = str, help = 'path to csv files')
    args = parser.parse_args();


    col_names = ["abnormal_seg", "abnormal_t1", "normal_seg", "normal_t1", "normal_c0"]

    CM_VE = True;

    if args.csv_path is not None:
        test_df = pd.read_csv(args.csv_path);

    fig = plt.figure(1);
    for index, row in test_df.iterrows():
        if "NORMAL" in row['abnormal_seg']:
            print("normal brain sample")
            atlas = row['abnormal_seg'].split('/')[1]
            patient = atlas
        else:
            atlas      = row['abnormal_seg'].split('/')[2].split('_regto_')[1]
            patient    = row['abnormal_seg'].split('/')[2].split('_regto_')[0]
        abnormal_p = os.path.join(args.base_dir, row['abnormal_seg'])
        normal_p   = os.path.join(args.base_dir, row['normal_seg'])

        if "NORMAL" in row['abnormal_seg']:
            orig_p = normal_p;
        else:
            orig_p     = os.path.join(os.path.dirname(abnormal_p), str(patient) + "_seg_aff2_" + str(atlas) + "_256x256x124.nii.gz")
        out_p      = os.path.join(os.path.join(args.results_directory, atlas), patient);


        print("processing {} / {}".format(patient,atlas))

        try:
            abnormal = nib.load(abnormal_p).get_fdata();
            normal   = nib.load(normal_p).get_fdata();
        except Exception as e:
            print("Error while trying to load images: ", e);
            continue;


        try:
            orig     = nib.load(orig_p).get_fdata();
        except Exception as e:
            print("no orig brats data found. Check if normal brain. ", e)
            orig = np.zeros_like(abnormal)

        if not os.path.exists(out_p):
            os.makedirs(out_p);

        # cm of connected component
        comps = {}
        count = {}
        sums  = {}
        relmass = {}
        relmass_sorted = {}
        xcm_data_px = {}
        xcm_data_px_sorted = {}
        structure = np.ones((3, 3, 3), dtype=np.int);
        total_mass = 0
        tc = abnormal==3
        ve = abnormal==7
        dat = ve if CM_VE else tc
        prefix = "cm-VE" if CM_VE else ""
        labeled, ncomponents = scipy.ndimage.measurements.label(dat, structure);
        for i in range(ncomponents):
            comps[i] = (labeled == i+1)
            a, b = scipy.ndimage.measurements._stats(comps[i])
            total_mass += b
        for i in range(ncomponents):
            count[i], sums[i]  = scipy.ndimage.measurements._stats(comps[i])
            relmass[i] = sums[i]/float(total_mass);
            xcm_data_px[i] = scipy.ndimage.measurements.center_of_mass(comps[i])

        # sort components according to their size
        sorted_rmass = sorted(relmass.items(), key=lambda x: x[1], reverse=True);
        perm = {}
        temp = {}
        labeled_sorted = np.zeros_like(labeled);
        for i in range(len(sorted_rmass)):
            perm[i]               = sorted_rmass[i][0] # get key from sorted list
            relmass_sorted[i]     = relmass[perm[i]];
            xcm_data_px_sorted[i] = xcm_data_px[perm[i]];

            temp[i] = (labeled == perm[i]+1).astype(int)*(i+1);
            labeled_sorted += temp[i];


        try:
            z = int(xcm_data_px_sorted[0][2])
            for img, capt, fname in zip([abnormal,  normal, orig], ['abnormal (syn)', 'normal (true)', 'abnormal (brats)'],
                ['abnormal_syn_'+prefix+'_ax-slice-'+str(z), 'normal_true_'+prefix+'_ax-slice-'+str(z), 'abnormal_brats_'+prefix+'_ax-slice-'+str(z)]):
                aax = fig.add_subplot(1, 1, 1);
                aax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False);
                aax.imshow(img[:,:,z].T, cmap='gray', interpolation='none', origin='upper');
                aax.set_title("axial slice %d" %  z, size='14', y=1.0)
                fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3);
                fig.savefig(os.path.join(out_p, fname + '.pdf'), format='pdf', dpi=1200);
                fig.clf()
            x = int(xcm_data_px_sorted[0][0])
            for img, capt, fname in zip([abnormal, normal, orig], ['abnormal (syn)', 'normal (true)', 'abnormal (brats)'],
                ['abnormal_syn_'+prefix+'_sag-slice-'+str(x),'normal_true_'+prefix+'_sag-slice-'+str(x), 'abnormal_brats_'+prefix+'_sag-slice-'+str(x)]):
                aax = fig.add_subplot(1, 1, 1);
                aax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False);
                aax.imshow(img[x,:,:].T, cmap='gray', interpolation='none', origin='upper');
                aax.set_title("sagittal slice %d" %  x, size='14', y=1.0)
                fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3);
                fig.savefig(os.path.join(out_p, fname + '.pdf'), format='pdf', dpi=1200);
                fig.clf()
            y = int(xcm_data_px_sorted[0][1])
            for img, capt, fname in zip([abnormal, normal, orig], ['abnormal (syn)',  'normal (true)', 'abnormal (brats)'],
                ['abnormal_syn_'+prefix+'_cor-slice-'+str(y), 'normal_true_'+prefix+'_cor-slice-'+str(y), 'abnormal_brats_'+prefix+'_cor-slice-'+str(y)]):
                aax = fig.add_subplot(1, 1, 1);
                aax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False);
                aax.imshow(img[:,y,:].T, cmap='gray', interpolation='none', origin='upper');
                aax.set_title("coronal slice %d" %  y, size='14', y=1.0)
                fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3);
                fig.savefig(os.path.join(out_p, fname + '.pdf'), format='pdf', dpi=1200);
                fig.clf()
        except Exception as e:
            print("ERROR in line {}, BID: {}, ATLAS: {}. Exception: {}".format(index, patient, atlas, e));


