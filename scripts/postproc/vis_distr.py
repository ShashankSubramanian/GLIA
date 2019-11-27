import matplotlib as mpl
mpl.use('Agg')
import os, sys
sys.path.append('/Users/scheufele/workspace/code/SIBIA/tumor-tools/3rdparty/pglistr_tumor/scripts/common/')
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
import file_io as fio
from pprint import pprint
from tabulate import tabulate
from scipy import stats
from mayavi import mlab
from mayavi.mlab import *
import moviepy.editor as mpy # to animate the data

duration=10


def visualize_surf(data):


    # kde = stats.gaussian_kde(values)

    # Create a regular 3D grid with 50 points in each dimension
    # xmin, ymin, zmin = data.min(axis=0)
    # xmax, ymax, zmax = data.max(axis=0)
    # xi, yi, zi = np.mgrid[xmin:xmax:50j, ymin:ymax:50j, zmin:zmax:50j]
    #
    # # Evaluate the KDE on a regular grid...
    # coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
    # density = kde(coords).reshape(xi.shape)

    # Visualize the density estimate as isosurfaces
    # mlab.contour3d(xi, yi, zi, density, opacity=0.5)
    mlab.contour3d(data, opacity=0.5)
    # mlab.axes()
    # mlab.show()


def make_frame(t):
    """ Generates and returns the frame for time t. """
    # mlab.actor.actor.rotate_y(t/duration*360)
    view(180,t/duration,0)
    return mlab.screenshot(antialiased=True) # return a RGB image


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


    # atlas_func = nib.load(os.path.join(args.dir,"lpba40_combined_LR_256x256x256_aff2jakob_in_jakob_space_240x240x155.nii.gz")).get_fdata();
    atlas = nib.load(os.path.join(args.dir,"jakob_segmented_with_cere_lps_240x240x155_in_brats_hdr.nii.gz")).get_fdata();

    glioma_c0_atlas           = nib.load(os.path.join(args.dir, "brats_c0_atlas_plain.nii.gz")).get_fdata();
    glioma_c0_atlas_short     = nib.load(os.path.join(args.dir, "brats[srvl]_c0_atlas_short_normalized-across.nii.gz")).get_fdata();
    glioma_c0_atlas_mid       = nib.load(os.path.join(args.dir, "brats[srvl]_c0_atlas_mid_normalized-across.nii.gz")).get_fdata();
    glioma_c0_atlas_long      = nib.load(os.path.join(args.dir, "brats[srvl]_c0_atlas_long_normalized-across.nii.gz")).get_fdata();
    glioma_c1_atlas           = nib.load(os.path.join(args.dir, "brats_c1_atlas_plain.nii.gz")).get_fdata();

    glioma_c1_atlas_short     = nib.load(os.path.join(args.dir, "brats[srvl]_c1_atlas_short_normalized-across.nii.gz")).get_fdata();
    glioma_c1_atlas_mid       = nib.load(os.path.join(args.dir, "brats[srvl]_c1_atlas_mid_normalized-across.nii.gz")).get_fdata();
    glioma_c1_atlas_long      = nib.load(os.path.join(args.dir, "brats[srvl]_c1_atlas_long_normalized-across.nii.gz")).get_fdata();

    atlas_c0_sml   = glioma_c0_atlas_short + glioma_c0_atlas_mid + glioma_c0_atlas_long;
    max_atlas_c0_sml = np.amax(atlas_c0_sml.flatten());
    # fio.writeNII((np.abs(glioma_c0_atlas_short  / max_atlas_c0_sml) > 0.1).astype(float),   os.path.join(dir,'brats[srvl]_c0_atlas_short_normalized-across_SEG.nii.gz'), affine);
    # fio.writeNII((np.abs(glioma_c0_atlas_mid    / max_atlas_c0_sml) > 0.1).astype(float),   os.path.join(dir,'brats[srvl]_c0_atlas_mid_normalized-across_SEG.nii.gz'),   affine);
    # fio.writeNII((np.abs(glioma_c0_atlas_long   / max_atlas_c0_sml) > 0.1).astype(float),   os.path.join(dir,'brats[srvl]_c0_atlas_long_normalized-across_SEG.nii.gz'),  affine);


    # visualize_surf(glioma_c0_atlas)

    # c1_vis = glioma_c1_atlas
    # c0_vis = glioma_c0_atlas
    # filename = "gbm_atlas_3d_c0_c1_all_distrb.png"

    # c1_vis = glioma_c1_atlas_short
    # c0_vis = glioma_c0_atlas_short
    # filename = "gbm_atlas_3d_c0_c1_short_distrb.png"

    c1_vis = glioma_c1_atlas_mid
    c0_vis = glioma_c0_atlas_mid
    filename = "gbm_atlas_3d_c0_c1_mid_distrb.png"

    # c1_vis = glioma_c1_atlas_long
    # c0_vis = glioma_c0_atlas_long
    # filename = "gbm_atlas_3d_c0_c1_long_distrb.png"

    ve = (atlas == 7).astype(float)
    bg = (atlas == 0).astype(float)
    for c1_vis, c0_vis, fname in zip(
        [glioma_c1_atlas, glioma_c1_atlas_short, glioma_c1_atlas_mid, glioma_c1_atlas_long],
        [glioma_c0_atlas, glioma_c0_atlas_short, glioma_c0_atlas_mid, glioma_c0_atlas_long],
        ['gbm_atlas_3d_c0_c1_all_distrb', 'gbm_atlas_3d_c0_c1_short_distrb', 'gbm_atlas_3d_c0_c1_mid_distrb', 'gbm_atlas_3d_c0_c1_long_distrb']
        ):

        fig = mlab.figure(size = (1024,1024), bgcolor = (0.1,0.1,0.1), fgcolor = (0.5, 0.5, 0.5))
        mlab.contour3d(bg, opacity=0.05, colormap='bone')
        mlab.contour3d(ve, opacity=0.6, colormap='bone')
        # mlab.contour3d(c1_vis, opacity=0.2, contours=6, colormap='coolwarm')#, transparent=True)
        mlab.contour3d(c1_vis, opacity=0.2, contours=6, colormap='Reds', transparent=True)
        mlab.contour3d(c0_vis, opacity=0.5, colormap='Greens')
        view(180,0)
        # view(azimuth=180, elevation=0)#, distance=130, focalpoint=np.array([0,0,1]), roll=0, reset_roll=True, figure=fig)
        mlab.savefig(fname+"_3.png", figure=fig)

        # animation = mpy.VideoClip(make_frame, duration=duration).resize(0.5)
        # Video generation takes 10 seconds, GIF generation takes 25s
        # animation.write_videofile("wireframe.mp4", fps=20)
        # animation.write_gif("wireframe.gif", fps=20)

        # break;
