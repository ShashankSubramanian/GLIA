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
#import skimage
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
  r_args.add_argument('-p', '--patient_dir', type = str, help = 'path to patient dir', required = True) 
  r_args.add_argument('-nm', '--patient_name', type = str, help = 'path to patient', required = True) 
  r_args.add_argument('-t', '--til_dir', type = str, help = 'path to til results', required = True) 
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
   

  ### since patients have multiple components, add pseudo patients to the list for each important component
  pat = args.patient_name
  wts = {}
  cms = {}
  comp_patient_list = []
  
  inv_dir = os.path.join(args.til_dir, pat) + "/inversion/nx256/obs-1.0/"
  with open(inv_dir + "/dcomp.dat", "r") as f:
    lines = f.readlines()
  rel_idx = lines.index("relative mass:\n")
  wt_l = []
  cm_l = []
  for c,l in enumerate(lines[rel_idx+1:]):
    wt = (float(l.strip("\n")))
    if wt < 0.1:
      break
    wt_l.append(wt)
    cm = lines[3+c].strip("\n")
    cm = tuple([float(x) for x in cm.split(",")])
    cm_l.append(cm[::-1])
  num_components = len(wt_l)
  if num_components <= 1:
    comp_patient_list.append(pat)
    wts[pat] = wt_l[0]
    cms[pat] = cm_l[0]
  else:
    for n in range(1,num_components+1):
      name_key = pat + "_comp" + str(n)
      comp_patient_list.append(name_key)
      wts[name_key] = wt_l[n-1]
      cms[name_key] = cm_l[n-1]

  num_pats = len(comp_patient_list)
  print(num_pats)
  sx = 256 / (2 * math.pi)
#  with PdfPages(args.results_dir + "/penn_til_" + pat + ".pdf") as pdf:
  for it in range(0,math.ceil(num_pats)):
    cur_pat = comp_patient_list[it:it+1]
    left_over_pats = 1
    ct = 0
    numax = 3
    fig, ax = plt.subplots(left_over_pats, numax, figsize=(12,4))
    pat_name = pat
    print("printing patient ", pat)
    t1ce = nib.load(args.patient_dir + "/" + pat_name + "/aff2jakob/" + pat_name + "_t1_aff2jakob.nii.gz").get_fdata()
    seg = nib.load(args.patient_dir + "/" + pat_name + "/aff2jakob/" + pat_name + "_seg_ants_aff2jakob.nii.gz").get_fdata()
    tu = np.logical_or(seg == 1, seg == 4) # tumor core
    ed = (seg == 2)
    com = cms[cur_pat[0]] 
    com = tuple([round(x*sx) for x in com])
    for ax_id in range(0,numax):
      ax[ax_id].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
    
    ## show the base image - mri or seg
    ax[0].imshow(t1ce[:,:,com[2]].T, cmap='gray', interpolation='none', origin='upper')
    ax[1].imshow(t1ce[:,com[1],:].T, cmap='gray', interpolation='none', origin='upper')
    ax[2].imshow(t1ce[com[0],:,:].T, cmap='gray', interpolation='none', origin='upper')

    ## scatter the com of that component
    marker_size = 12
    ax[0].scatter(com[0], com[1], c = 'k', marker = 'x', s = marker_size )
    ax[1].scatter(com[0], com[2], c = 'k', marker = 'x', s = marker_size )
    ax[2].scatter(com[1], com[2], c = 'k', marker = 'x', s = marker_size )

    ## get the tils
    inv_dir = os.path.join(args.til_dir, pat_name) + "/inversion/nx256/obs-1.0/"
    cmvec = "phi-mesh-scaled.txt"
    pvec  = "p-rec-scaled.txt"
    phix, phiy, phiz = np.loadtxt(os.path.join(inv_dir, cmvec), comments = ["]", "#"], delimiter=',', skiprows=2, unpack=True)
    p_vec = np.loadtxt(os.path.join(inv_dir, pvec),  comments = ["]", "#"], skiprows=1);
    phi = []
    for x,y,z in zip(phix, phiy, phiz):
      phi.append(tuple([z,y,x]))


      ## scatter the pvecs and phis
      Xx = []
      Yy = []
      Zz = []
      marker_size = 12
      for k in range(len(phi)):
        ax[0].scatter(round(phi[k][0]*sx), round(phi[k][1]*sx), c='b',  marker='o', s=marker_size*p_vec[k])
        ax[2].scatter(round(phi[k][1]*sx), round(phi[k][2]*sx), c='b',  marker='o', s=marker_size*p_vec[k])
        ax[1].scatter(round(phi[k][0]*sx), round(phi[k][2]*sx), c='b',  marker='o', s=marker_size*p_vec[k])
        Xx.append(round(phi[k][0]*sx))
        Yy.append(round(phi[k][2]*sx))
        Zz.append(round(phi[k][1]*sx))

      ## get some ranges to zoom into the til region
      pad=55
      Xx = np.asarray(Xx);
      Yy = np.asarray(Yy);
      Zz = np.asarray(Zz);
      max_range = np.amax(np.array([np.amax(Xx)-np.amin(Xx), np.amax(Yy)-np.amin(Yy), np.amax(Zz)-np.amin(Zz)]))
      max_range_a = (np.array([np.amax(Xx)-np.amin(Xx), np.amax(Yy)-np.amin(Yy), np.amax(Zz)-np.amin(Zz)]))
      mid_x = (np.amax(Xx)+np.amin(Xx)) * 0.5
      mid_y = (np.amax(Yy)+np.amin(Yy)) * 0.5
      mid_z = (np.amax(Zz)+np.amin(Zz)) * 0.5

      ## get the reconstructed solution
      c1 = nib.load(os.path.join(inv_dir, "c1_rec_256256256.nii.gz")).get_fdata()

      ## overlay the tumor conc 
      ax[0].imshow(thresh(c1[:,:,com[2]].T, cmap_c1, thresh=0.4), interpolation='none', alpha=0.25)
      ax[1].imshow(thresh(c1[:,com[1],:].T, cmap_c1, thresh=0.4), interpolation='none', alpha=0.25)
      ax[2].imshow(thresh(c1[com[0],:,:].T, cmap_c1, thresh=0.4), interpolation='none', alpha=0.25)

      ## show the tumor data core as a contour
      ax[0].contour(tu[:,:,com[2]].T,  levels=[0.0,1.0],  cmap=plt.cm.gray, linestyles=['--'] ,linewidths=0.8, alpha=0.9)
      ax[1].contour(tu[:,com[1],:].T,  levels=[0.0,1.0],  cmap=plt.cm.gray, linestyles=['--'] ,linewidths=0.8, alpha=0.9)
      ax[2].contour(tu[com[0],:,:].T,  levels=[0.0,1.0],  cmap=plt.cm.gray, linestyles=['--'] ,linewidths=0.8, alpha=0.9)
      ## show the tumor ed data as a contour
      ax[0].contour(ed[:,:,com[2]].T,  levels=[0.0,1.0],  cmap=plt.cm.binary, linestyles=['--'] ,linewidths=0.8, alpha=0.9)
      ax[1].contour(ed[:,com[1],:].T,  levels=[0.0,1.0],  cmap=plt.cm.binary, linestyles=['--'] ,linewidths=0.8, alpha=0.9)
      ax[2].contour(ed[com[0],:,:].T,  levels=[0.0,1.0],  cmap=plt.cm.binary, linestyles=['--'] ,linewidths=0.8, alpha=0.9)
      
      ### zoom aesthetics
#        pad_x = 47
#        pad_y = 30
#        ox = 0
#        oy = 10
#        for ax_id in range(0,numax):
#          if ax_id is not 0:
#            ax[ax_id].set_xlim(pad_x,256-pad_x+ox)
#            ax[ax_id].set_ylim(pad_y-oy,256-pad_y)
#          else:
#            ax[ax_id].set_xlim(pad_x,256-pad_x+ox)
#            ax[ax_id].set_ylim(pad_y-oy,256-pad_y)
#          if ax_id == 0:
#            ax[ax_id].set_ylim(ax[ax_id].get_ylim()[::-1])
      pad=30
#      ax[0].set_xlim(mid_x-max_range_a[0]/2-pad, mid_x+max_range_a[0]/2+pad)
#      ax[0].set_ylim(mid_y-max_range_a[1]/2-pad, mid_y+max_range_a[1]/2+pad)
#      ax[1].set_xlim(mid_y-max_range_a[1]/2-pad, mid_y+max_range_a[1]/2+pad)
#      ax[1].set_ylim(mid_z-max_range_a[2]/2-pad, mid_z+max_range_a[2]/2+pad)
#      ax[2].set_xlim(mid_x-max_range_a[0]/2-pad, mid_x+max_range_a[0]/2+pad)
#      ax[2].set_ylim(mid_z-max_range_a[2]/2-pad, mid_z+max_range_a[2]/2+pad) 
      ax[0].set_xlim(mid_x-max_range/2-pad, mid_x+max_range/2+pad)
      ax[0].set_ylim(mid_y-max_range/2-pad+20, mid_y+max_range/2+pad+20)
      ax[1].set_xlim(mid_y-max_range/2-pad+25, mid_y+max_range/2+pad+25)
      ax[1].set_ylim(mid_z-max_range/2-pad-10, mid_z+max_range/2+pad-10)
      ax[2].set_xlim(mid_x-max_range/2-pad, mid_x+max_range/2+pad)
      ax[2].set_ylim(mid_z-max_range/2-pad, mid_z+max_range/2+pad) 
      ax[0].set_ylim(ax[0].get_ylim()[::-1])        # invert the axis
#        ax[0].set_ylabel(pat, fontsize=5)
    
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')
    ax[2].set_aspect('equal', adjustable='box')
    sns.despine(offset=0, trim=True)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.25)
#      pdf.savefig()
    fig.savefig(args.results_dir + "/penn_til_" + pat_name + "_comp[" + str(it) + "].pdf", format="pdf", dpi=1200)
    fig.clf()
