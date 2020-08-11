import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import scipy as sc
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset
from numpy import linalg as la
import matplotlib as mpl
### latex params -- switch off if not on local machine
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
from matplotlib.backends.backend_pdf import PdfPages

## other helpers
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../utils/')
from file_io import writeNII, createNetCDF
from image_tools import resizeImage, resizeNIIImage

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


def compute_com(img):
  return tuple(int(s) for s in ndimage.measurements.center_of_mass(img))

#--------------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Mass effect inversion',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-p', '--patient_dir', type = str, help = 'path to patients', required = True) 
  r_args.add_argument('-x', '--results_dir', type = str, help = 'results path', required = True) 
  args = parser.parse_args();

  ### some graph variables ###
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

  #patient_list = os.listdir(args.patient_dir)

  brats_format = False
  if brats_format:
    with open(args.patient_dir + "/brats-pat-stats.csv", "r") as f:
      brats_pats = f.readlines()
    patient_list = []
    for l in brats_pats:
      patient_list.append(l.split(",")[0])
  else:
    patient_list = []
    for f in os.listdir(args.patient_dir):
      patient_list.append(f)

  for pat in patient_list:
    if not os.path.exists(args.patient_dir + "/" + pat + "/" + pat + "_t1.nii.gz"):
      print("removing ", pat)
      patient_list.remove(pat)

###  patient_list = patient_list[0:10] ## snafu: checking
  num_pats = len(patient_list)
  num_pats_per_page = 5

  with PdfPages(args.results_dir + "/penn_mri.pdf") as pdf:
    for it in range(0,math.ceil(num_pats/num_pats_per_page)):
      cur_pats = patient_list[it*num_pats_per_page:it*num_pats_per_page + num_pats_per_page]
      ct = 0
      fig, ax = plt.subplots(num_pats_per_page, 6, figsize=(4,5))
      for pat in cur_pats:
        print("printing patient ", pat)
        for ax_id in range(0,6):
          ax[ct][ax_id].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
        if brats_format:
          t1ce = nib.load(args.patient_dir + "/" + pat + "/aff2jakob/" + pat + "_t1ce_aff2jakob.nii.gz").get_fdata()
          seg = nib.load(args.patient_dir + "/" + pat + "/aff2jakob/" + pat + "_seg_ants_aff2jakob.nii.gz").get_fdata()
          com = compute_com(seg)
        else:
          t1ce = nib.load(args.patient_dir + "/" + pat + "/aff2jakob/" + pat + "_t1ce_aff2jakob.nii.gz").get_fdata()
          seg = nib.load(args.patient_dir + "/" + pat + "/aff2jakob/" + pat + "_seg_ants_aff2jakob.nii.gz").get_fdata()
#          t1ce = nib.load(args.patient_dir + "/" + pat + "/" + pat + "_t1ce.nii.gz").get_fdata()
#          seg = nib.load(args.patient_dir + "/" + pat + "/" + pat + "_seg_epoch200.nii.gz").get_fdata()
          com = compute_com(seg)
        ax[ct][0].imshow(t1ce[:,:,com[2]].T, cmap='gray', interpolation='none', origin='upper')
        ax[ct][1].imshow(t1ce[:,com[1],:].T, cmap='gray', interpolation='none', origin='upper')
        ax[ct][2].imshow(t1ce[com[0],:,:].T, cmap='gray', interpolation='none', origin='upper')
        ax[ct][3].imshow(seg[:,:,com[2]].T, cmap='gray', interpolation='none', origin='upper')
        ax[ct][4].imshow(seg[:,com[1],:].T, cmap='gray', interpolation='none', origin='upper')
        ax[ct][5].imshow(seg[com[0],:,:].T, cmap='gray', interpolation='none', origin='upper')
        
        ### aesthetics
        pad_x = 47
        pad_y = 30
        ox = 0
        oy = 10

        for ax_id in range(0,6):
          if ax_id is not 0 and ax_id is not 3:
            ax[ct][ax_id].set_xlim(pad_x,256-pad_x+ox)
            ax[ct][ax_id].set_ylim(pad_y-oy,256-pad_y)
          else:
            ax[ct][ax_id].set_xlim(pad_x,256-pad_x+ox)
            ax[ct][ax_id].set_ylim(pad_y-oy,256-pad_y)
          if ax_id == 0 or ax_id == 3:
            ax[ct][ax_id].set_ylim(ax[ct][ax_id].get_ylim()[::-1])
        #  ax[ct][ax_id].set_title("slice {}".format(com[3 - ((ax_id+1%3) if (ax_id+1)%3 is not 0 else 3)]))
        ax[ct][0].set_ylabel(pat, fontsize=5)
        ct += 1

      fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3)
      pdf.savefig()
      fig.clf()


