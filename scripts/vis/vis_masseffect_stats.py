import matplotlib as mpl
import pandas as pd
import shutil
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.ticker import NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import scipy
from scipy import ndimage
import numpy as np
import os, sys, argparse
from pandas.plotting import scatter_matrix
from netCDF4 import Dataset
import nibabel as nib
from matplotlib.backends.backend_pdf import PdfPages
from numpy import linalg as la

def read_netcdf(filename):
  '''
  function to read a netcdf image file and return its contents
  '''
  imgfile = Dataset(filename);
  img = imgfile.variables['data'][:]
  imgfile.close();
  return np.transpose(img)


def colorbar(mappable):
  last_axes = plt.gca()
  ax = mappable.axes
  fig = ax.figure
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  cbar = fig.colorbar(mappable, cax=cax)
  plt.sca(last_axes)
  return cbar

def create_segmentation(root_path, is_patient = False):
  if is_patient:
    wm  = read_netcdf(root_path + "/p_wm.nc")  
    gm  = read_netcdf(root_path + "/p_gm.nc")  
    csf = read_netcdf(root_path + "/p_csf.nc")  
    vt  = read_netcdf(root_path + "/p_vt.nc")  
    tu  = read_netcdf(root_path + "/data.nc")
    ed  = 1 - read_netcdf(root_path + "/obs.nc")
    wm  = wm * (1 - tu - ed)
    gm  = gm * (1 - tu - ed)
    csf = csf * (1 - tu - ed)
    vt  = vt * (1 - tu - ed)
    bg  = 1 - (vt + csf + gm + wm + tu + ed)
  else:
    wm  = read_netcdf(root_path + "/wm.nc")  
    gm  = read_netcdf(root_path + "/gm.nc")  
    csf = read_netcdf(root_path + "/csf.nc")  
    vt  = read_netcdf(root_path + "/vt.nc") 
    tu  = 0 * vt
    ed  = 0 * vt
    bg  = 1 - (vt + csf + wm + gm)
  max_arr = np.maximum(wm, gm)
  max_arr = np.maximum(max_arr, csf)
  max_arr = np.maximum(max_arr, vt)
  if is_patient:
    max_arr = np.maximum(max_arr, tu)
    max_arr = np.maximum(max_arr, ed)
  max_arr = np.maximum(max_arr, bg)
  return ((max_arr == wm) * 6 + (max_arr == gm) * 5 + (max_arr == csf) * 8 + (max_arr == vt) * 7 + (max_arr == tu) * 1 + (max_arr == ed) * 2 + (max_arr == bg) * 0)

def compute_com(img):
  return tuple(int(s) for s in ndimage.measurements.center_of_mass(img))

def compute_vol(mat):
  sz      = mat.shape[0]
  h       = (2.0 * math.pi) /  sz
  measure = h * h * h
  vol = np.sum(mat.flatten())
  vol *= measure
  return vol

#--------------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='script to visualize some stats for mass effect reconstructions',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-p', '--patient_dir', type = str, help = 'path to masseffect patient reconstructions', required = True) 
  r_args.add_argument('-x', '--results_dir', type = str, help = 'results path', required = True) 
  args = parser.parse_args();

  me_res_dir = args.patient_dir
  stat_file  = me_res_dir + "/penn_masseffect_inversion_stats.csv"
  stats      = pd.read_csv(stat_file, header=0)
  stats.reset_index(inplace=True)

  ### add some important columns
  stats['diff-vol'] = stats['mu-vol-err-nm'] - stats['mu-vol-err']
  stats['diff-l2'] = stats['mu-l2-err-nm'] - stats['mu-l2-err']

  ### subselect patients
  ### pats with diff-vol < 0
  patient_list = stats.loc[stats['diff-vol'] < 0].sort_values(by=['diff-vol'])['PATIENT'].tolist()
  patient_list = patient_list[0:4]
  print("selected patients: ")
  for pat in patient_list:
    print(pat)
  im_sz = 160
  atlases = np.zeros((im_sz,im_sz,im_sz,16))
  recon   = atlases.copy()
  disp    = atlases.copy()
  ### aesthetics
  pad_x = 25
  pad_y = 25
  ox = 0
  oy = 10
  im_at = np.zeros((im_sz,im_sz,im_sz,4))
  im_rec = im_at.copy()
  im_disp = im_at.copy()
  
  with PdfPages(args.results_dir + "/penn_masseffect_stats.pdf") as pdf:
    for pat in patient_list:
      pat_name = pat
      atlases[:] = 0
      recon[:]   = 0
      disp[:]    = 0
      fig, ax = plt.subplots(4,4,figsize=(12,16))
      for ct in range(0,4):
          ax[ct][0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
          ax[ct][1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
          ax[ct][2].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
          ax[ct][3].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
      print("printing for patient ", pat)
      path_to_results = os.path.join(me_res_dir, pat)
      atlas_list = os.path.join(path_to_results, "atlas-list.txt")
      with open(atlas_list, "r") as f:
          lines = f.readlines()
      atlas_names = []
      for ln in lines:
          atlas_names.append(ln.strip("\n"))

      num_atlases = len(atlas_names)
      patient = nib.load(path_to_results + "/tu/160/" + pat + "_seg_ants_aff2jakob_160.nii.gz").get_fdata()

      pat_compute = False
      vt_at = []
      for idx, atlas in enumerate(atlas_names):
          curr_path = path_to_results + "/tu/160/" + atlas + "/"
          atlases[:,:,:,idx] = create_segmentation(root_path = curr_path)
          vt_at.append(compute_vol(atlases[:,:,:,idx] == 7))
          if not pat_compute:
              pat = create_segmentation(root_path = curr_path, is_patient = True)
              pat_compute = True
          recon[:,:,:,idx]   = read_netcdf(curr_path + "seg_rec_final.nc")
          disp[:,:,:,idx]    = read_netcdf(curr_path + "displacement_rec_final.nc")
          
      vt_pat = compute_vol(pat == 7)
      vt_at_dict = dict(zip(atlas_names, vt_at))
      sorted_names = [x for _,x in sorted(zip(vt_at,atlas_names))]
      vt_diff = {k:v-vt_pat for k,v in vt_at_dict.items()}
      vt_diff_abs = {k:np.abs(v) for k,v in vt_diff.items()}
      atlas_failed = (min(vt_diff.values()) < 0) # atlas selection failed if some atlases with smaller vt vols were selected using nearest neighbors
      min_at = min(vt_diff_abs, key=vt_diff_abs.get)
      max_at = max(vt_diff_abs, key=vt_diff_abs.get)
      med_at = sorted_names[num_atlases//2]

      max_disp = round(np.max(disp.flatten()),1)

      ## first image is probabilistic
      im_at[:,:,:,0] = atlases.sum(axis=3)/num_atlases
      im_rec[:,:,:,0] = recon.sum(axis=3)/num_atlases
      im_disp[:,:,:,0] = disp.sum(axis=3)/num_atlases

      ## next is closest to patient vt vol
      im_at[:,:,:,1] = atlases[:,:,:,atlas_names.index(min_at)]
      im_rec[:,:,:,1] = recon[:,:,:,atlas_names.index(min_at)]
      im_disp[:,:,:,1] = disp[:,:,:,atlas_names.index(min_at)]

      ## next is farthest to patient vt vol
      im_at[:,:,:,2] = atlases[:,:,:,atlas_names.index(max_at)]
      im_rec[:,:,:,2] = recon[:,:,:,atlas_names.index(max_at)]
      im_disp[:,:,:,2] = disp[:,:,:,atlas_names.index(max_at)]

      ## last is median atlas
      im_at[:,:,:,3] = atlases[:,:,:,atlas_names.index(med_at)]
      im_rec[:,:,:,3] = recon[:,:,:,atlas_names.index(med_at)]
      im_disp[:,:,:,3] = disp[:,:,:,atlas_names.index(med_at)]
      tu     = (pat == 1)
      com    = compute_com(tu)
      axial  = com[2] 
      for idx in range(0,4):
        ax[idx][0].imshow(im_at[:,:,axial,idx].T, cmap='gray', interpolation='none', origin='upper')
        ax[idx][1].imshow(im_rec[:,:,axial,idx].T, cmap='gray', interpolation='none', origin='upper')
        ax[idx][2].imshow(pat[:,:,axial].T, cmap='gray', interpolation='none', origin='upper')
        im = ax[idx][3].imshow(im_disp[:,:,axial,idx].T, vmin=0, vmax=max_disp,cmap=plt.cm.coolwarm, interpolation='none', origin='upper') #viridis
        
        for ct in range(0,4):
            ax[idx][ct].set_xlim(pad_x,160-pad_x+ox)
            ax[idx][ct].set_ylim(pad_y-oy,160-pad_y)
            ax[idx][ct].set_ylim(ax[idx][ct].get_ylim()[::-1])
        
        cb_ax = fig.add_axes([0.92,0.768 - 0.239*idx,0.02,0.18])
        cbar = fig.colorbar(im, cax=cb_ax)
        # cbar = colorbar(im)
        ##  set the colorbar ticks and tick labels
        cbar.set_ticks([0,max_disp])
        cbar.ax.tick_params(labelsize=20) 
        cbar.set_ticklabels(['0', str(max_disp)])

      ### titles
      ax[0][0].set_title("Atlas", fontsize="12",fontweight='bold')
      ax[0][1].set_title("Atlas-Rec", fontsize="12",fontweight='bold')
      ax[0][2].set_title("Patient", fontsize="12",fontweight='bold')
      ax[0][3].set_title("Disp", fontsize="12",fontweight='bold')
      ## side titles (hack: use labels)
      ax[0][0].set_ylabel("Probalistic", fontsize="12", fontweight="bold")
      ax[1][0].set_ylabel("Closest", fontsize="12", fontweight="bold")
      ax[2][0].set_ylabel("Farthest", fontsize="12", fontweight="bold")
      ax[3][0].set_ylabel("Median", fontsize="12", fontweight="bold")

      fig.subplots_adjust(left=0.05, right=0.9, bottom=0.05, top=0.95, wspace=0.0, hspace=0.3)
      df_row = stats.loc[stats["PATIENT"] == pat_name]
      fig.suptitle("{}; (vol,l2,status)=({:3G},{:3G},{})".format(pat_name, df_row['diff-vol'].values[0]*100, df_row['diff-l2'].values[0]*100, atlas_failed))
      pdf.savefig()
      fig.clf()


