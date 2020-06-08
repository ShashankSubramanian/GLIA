import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset
from numpy import linalg as la
import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['axes.formatter.use_mathtext'] = True
import matplotlib.pyplot as plt
import cv2
import skimage
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import pandas as pd
import seaborn as sns
import math

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


d_thresh  = 0.3;  # values d(x)  > thres have zero transparency
c0_thresh = 0.3;  # values c0(x) > thres have zero transparency
c1_thresh = 0.4;  # values c1(x) > thres have zero transparency
cmap_c0 = plt.cm.Reds;
cmap_c1 = plt.cm.cool;
cmap_d  = plt.cm.winter;
levels_contour_c0  = [0.2,0.5, 0.7, 0.9];
levels_contour_c1  = [0.1, 0.3, 0.6, 0.9, 1.0];
levels_contourf_c1 = [0.05, 0.1, 0.5, 0.7];
lwidths            = [0.5, 0.5, 0.5, 0.5, 0.5]

fsize=6;
tpos = 1.0;
cmap_d = plt.cm.YlOrBr_r
cmap_c1 = mpl.cm.get_cmap(plt.cm.rainbow, len(levels_contour_c1)-1);
cmap_seg = mpl.cm.get_cmap(plt.cm.gray, len(levels_contour_c1)-1);
from matplotlib.colors import LinearSegmentedColormap
colors = plt.cm.spring;
colors = colors(np.linspace(0.0, 0.4, colors.N // 2))
cmap_c0c = LinearSegmentedColormap.from_list('c0_cmap', colors)

### imaging results
scripts_path = os.path.dirname(os.path.realpath(__file__)) + "/../"
inv_suff     = ""
#p_list  = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1", "Brats18_CBICA_AAP_1"]
p_list  = ["Brats18_CBICA_AAP_1"]
#atlas_l = ["atlas-6", "atlas-1", "atlas-5", "atlas-2"]
#atlas_l = ["50446", "50466", "51153", "51372"]
atlas_l = ["50665"]
idx     = 0
#sax_l   = [112, 112, 110, 125]
sax_l   = [125]

for patient in p_list:
  #patient = "Brats18_CBICA_ABO_1"
  #atlas   = "atlas-" + str(3)
  atlas   = atlas_l[idx]
  invdir  = scripts_path + "../results/" + patient  + "/"
  reg_dir = invdir + "/reg/" + atlas + "/"
  inv_dir = invdir + "/tu/" + atlas + inv_suff + "/"
  st_dir  = invdir +"/stat/"
  vis_dir = st_dir + "vis"
  if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
  atlas   = nib.load(reg_dir + atlas + "_labels.nii.gz").get_fdata()
  pat     = nib.load(invdir + patient + "_labels.nii.gz").get_fdata()
  pat_tr  = nib.load(reg_dir + patient + "_labels_transported.nii.gz").get_fdata()
  at_tu   = nib.load(inv_dir + "seg_rec_final.nii.gz").get_fdata()
  at_c    = nib.load(inv_dir + "c_rec_final.nii.gz").get_fdata()
  at_u    = nib.load(inv_dir + "displacement_rec_final.nii.gz").get_fdata()

  seg_tu  = np.logical_or(np.logical_or(pat == 1, pat == 2), pat == 4)
  seg_tc  = np.logical_or(pat == 1, pat == 4)

  fig, ax = plt.subplots(1,4,figsize=(12,4))
  ax[0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
  ax[1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
  ax[2].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
  ax[3].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
#  ax[4].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)

### slice
  sax = sax_l[idx]

### plot
  slice, norm = cont(at_c[:,:,sax].T, cmap_c1);
  ax[0].imshow(atlas[:,:,sax].T, cmap='gray', interpolation='none', origin='upper')
  ax[1].imshow(pat[:,:,sax].T, cmap='gray', interpolation='none', origin='upper')
 # ax[2].imshow(pat_tr[:,:,sax].T, cmap='gray', interpolation='none', origin='upper')
  ax[2].imshow(at_tu[:,:,sax].T, cmap='gray', interpolation='none', origin='upper')
  im = ax[2].imshow(thresh(at_c[:,:,sax].T, cmap_c1, thresh=0.4), cmap=plt.cm.rainbow,interpolation='none', alpha=1)
  ax[2].contour(seg_tc[:,:,sax].T,  levels=[0.0,1.0],  cmap=plt.cm.gray, linestyles=['--'] ,linewidths=0.8, norm=norm, alpha=0.9)
  ax[3].imshow(at_tu[:,:,sax].T, cmap='gray', interpolation='none', origin='upper')
  im = ax[3].imshow(thresh(at_u[:,:,sax].T, cmap_c1, thresh=0.3), cmap=plt.cm.rainbow,interpolation='none', alpha=1)
  ax[3].contour(seg_tc[:,:,sax].T,  levels=[0.0,1.0],  cmap=plt.cm.gray, linestyles=['--'] ,linewidths=0.8, norm=norm, alpha=0.9)


### aesthetics
  pad_x = 47
  pad_y = 30
  ox = 0
  oy = 10
  for ct in range(0,4):
    ax[ct].set_xlim(pad_x,256-pad_x+ox)
    ax[ct].set_ylim(pad_y-oy,256-pad_y)
    ax[ct].set_ylim(ax[ct].get_ylim()[::-1])

### titles
#ax[0].set_title("Atlas", fontsize="20",fontweight='bold')
#ax[1].set_title("Patient", fontsize="20",fontweight='bold')
#ax[2].set_title("Pat-trans", fontsize="20",fontweight='bold')
#ax[3].set_title("At-tumor", fontsize="20",fontweight='bold')
#ax[4].set_title("At-disp", fontsize="20",fontweight='bold')
  ax[0].set_title("A", fontsize="15")
  ax[1].set_title("P", fontsize="15")
#  ax[2].set_title("Pat-trans", fontsize="15")
  ax[2].set_title("AT (c)", fontsize="15")
  ax[3].set_title("AT (u)", fontsize="15")

### colorbar
  cb_ax = fig.add_axes([0.96, 0.22, 0.02, 0.56])
  cbar  = fig.colorbar(im, cax=cb_ax)
#  set the colorbar ticks and tick labels
  cbar.set_ticks([0,1])
  cbar.ax.tick_params(labelsize=20) 
  cbar.set_ticklabels(['0', '1'])
  fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3)
  fig.savefig(vis_dir + "/comp-vis-50665.pdf", format='pdf', dpi=1200);
  fig.clf()
  idx += 1
