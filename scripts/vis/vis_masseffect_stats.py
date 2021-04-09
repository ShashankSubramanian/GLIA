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
  return ((max_arr == wm) * 6 + (max_arr == gm) * 5 + (max_arr == csf) * 8 + (max_arr == vt) * 7 + (max_arr == tu) * 2 + (max_arr == ed) * 3 + (max_arr == bg) * 0)

def compute_com(img):
  return tuple(int(s) for s in ndimage.measurements.center_of_mass(img))

def compute_vol(mat):
  sz      = mat.shape[0]
  h       = (2.0 * math.pi) /  sz
  measure = h * h * h
  vol = np.sum(mat.flatten())
  vol *= measure
  return vol

def normalize_img(img):
  #preprocess
#  img -= img.mean()
#  img /= img.std()
#  img = img.clip(-np.percentile(img, 99.9), np.percentile(img, 99.9))
#  img = (img - np.min(img))/(np.max(img) - np.min(img))
  return img

def create_prob_img(img, mri):
  """
    Creates a probabilistic image of a series of segmentations
    Assumes segmentation labels are in standard brats format
  """
  shp = list(img.shape)
  num_atlases = shp[-1]
  num_props = 6 ## bg, tu, gm, wm, vt, csf
  sz = shp[:3].copy()
  sz.append(num_props)
  prob = np.zeros(tuple(sz))
  idx_dict = {0:0, 1:1, 2:5, 3:6, 4:7, 5:8}
  # for each label, compute all the atlases which predict that label and divide by total number of atlases
  # this gives a probability of each label
  mri_select = np.zeros(tuple(sz))
  temp = np.zeros(tuple(sz[:3]))
  for idx in range(num_props):
    mask = (img == idx_dict[idx])
    prob[:,:,:,idx] = np.count_nonzero(mask, axis=3)
    # get intensities for each label averaged across the atlases
    temp = prob[:,:,:,idx]
    temp[temp == 0] = -1 #avoid division by zero
    mri_select[:,:,:,idx] = np.sum(mask * mri, axis=3)/temp 
  # resegment using the above probabilites by computing the most likely label
  #prob_seg = np.sum(prob, axis=3)
  prob_seg = np.argmax(prob, axis=3)
  prob_seg_mod = prob_seg.copy()
  mri_prob = np.zeros(tuple(sz[:3]))
  for idx in range(num_props):
    prob_seg_mod[prob_seg == idx] = idx_dict[idx]

  for k,i in idx_dict.items():
    # set the intensities only in the correct regions
    temp = mri_select[:,:,:,k]
    mri_prob[prob_seg_mod == i] = temp[prob_seg_mod == i]

  mri_prob = ndimage.gaussian_filter(mri_prob, sigma=0.6)
  return prob_seg_mod, normalize_img(mri_prob)


#--------------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='script to visualize some stats for mass effect reconstructions',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('-p', '--patient_dir', type = str, help = 'path to masseffect patient reconstructions', required = True) 
  r_args.add_argument('-a', '--atlas_dir', type = str, help = 'path to atlases', required = True) 
  r_args.add_argument('-d', '--data_dir', type = str, help = 'path to masseffect inversion input data (if different from patient_dir)') 
  r_args.add_argument('-x', '--results_dir', type = str, help = 'results path', required = True) 
  args = parser.parse_args();

  me_res_dir = args.patient_dir
  atlas_dir = args.atlas_dir
  if not args.data_dir:
    data_dir = me_res_dir
  else:
    data_dir = args.data_dir
  temp = os.listdir(me_res_dir)
  for tmp in temp:
    if "csv" in tmp:
      stat_file_name = tmp
      break
  stat_file  = os.path.join(me_res_dir, stat_file_name) 
  stats      = pd.read_csv(stat_file, header=0)
  stats.reset_index(inplace=True)
  if not data_dir == me_res_dir:
    temp = os.listdir(data_dir)
    for tmp in temp:
      if "csv" in tmp:
        stat_file_name = tmp
        break
    stat_file  = os.path.join(data_dir, stat_file_name) 
    stats_me   = pd.read_csv(stat_file, header=0)
  else:
    stats_me = stats 

  ### add some important columns
#  if not 'diff-vol' in stats:
#    stats['diff-vol'] = stats['mu-vol-err-nm'] - stats['mu-vol-err']
#    stats['diff-l2'] = stats['mu-l2-err-nm'] - stats['mu-l2-err']

  ### subselect patients
  ### pats with diff-vol < 0
  #patient_list = stats_me.loc[stats_me['diff-vol'] < 0].sort_values(by=['diff-vol'])['PATIENT'].tolist()
  patient_list = stats['PATIENT'].tolist()
  q1 = np.quantile(stats_me["mu-u"], 0.34)
  q2 = np.quantile(stats_me["mu-u"], 0.67)
  patient_list = stats_me.loc[stats_me['mu-u'] >= q2]['PATIENT'].tolist()
  #patient_list = stats_me.loc[np.logical_and(stats_me['mu-u'] >= q1, stats_me['mu-u'] < q2)]['PATIENT'].tolist()
  #patient_list = patient_list[0:10]
  
  print("selected patients: ")
  for pat in patient_list:
    print(pat)
  im_sz = 160
  atlases = np.zeros((im_sz,im_sz,im_sz,16))
  recon   = atlases.copy()
  mris    = atlases.copy()
  rec_mris = atlases.copy()
  disp    = atlases.copy()
  c       = atlases.copy()
  ### aesthetics
  pad_x = 25
  pad_y = 25
  ox = 0
  oy = 10
  cmap_c1 = mpl.cm.get_cmap(plt.cm.rainbow, 4);
  im_at = np.zeros((im_sz,im_sz,im_sz,4))
  at = im_at.copy()
  im_rec = im_at.copy()
  im_disp = im_at.copy()
  im_c = im_at.copy()

#  stat_file = open(args.results_dir + "/inversion_stats.txt", "w")
#  stat_file.write("PATIENT,atlas-status,diff-vol,diff-l2,pat-vol,prob-vol,min-vol,max-vol,med-vol\n")
  
  with PdfPages(args.results_dir + "/penn_masseffect_stats_high.pdf") as pdf:
    for pat in patient_list:
      pat_name = pat
      atlases[:] = 0
      mris[:]    = 0
      rec_mris[:] = 0
      recon[:]   = 0
      disp[:]    = 0
      c[:]       = 0
      fig, ax = plt.subplots(4,4,figsize=(12,16))
      for ct in range(0,4):
          ax[ct][0].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
          ax[ct][1].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
          ax[ct][2].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
          ax[ct][3].tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
      print("printing for patient ", pat)
      path_to_results = os.path.join(me_res_dir, pat)
      path_to_data = os.path.join(data_dir, pat)
      atlas_list = os.path.join(path_to_results, "atlas-list.txt")
      with open(atlas_list, "r") as f:
          lines = f.readlines()
      atlas_names = []
      for ln in lines:
          atlas_names.append(ln.strip("\n"))

      num_atlases = len(atlas_names)
      patient = nib.load(path_to_data + "/tu/160/" + pat + "_seg_ants_aff2jakob_160.nii.gz").get_fdata()

      pat_compute = False
      vt_at = []
      for idx, atlas in enumerate(atlas_names):
          curr_path = path_to_results + "/tu/160/" + atlas + "/"
          atlases[:,:,:,idx] = create_segmentation(root_path = curr_path)
          vt_at.append(compute_vol(atlases[:,:,:,idx] == 7))
          if not pat_compute:
              pat = create_segmentation(root_path = curr_path, is_patient = True)
              pat[np.logical_and(pat != 2, pat != 3)] = 0 # kill everything that is not tumor
              pat_mri = read_netcdf(path_to_results + "/tu/160/" + pat_name + "_t1_aff2jakob_160.nc")
              pat_mri = normalize_img(pat_mri)    
              pat_compute = True
          recon[:,:,:,idx]   = read_netcdf(curr_path + "seg_rec_final.nc")
          mris[:,:,:,idx]    = normalize_img(read_netcdf(atlas_dir + "/" + atlas + "_t1_aff2jakob_160.nc"))
          rec_mris[:,:,:,idx] = normalize_img(read_netcdf(curr_path + "mri_rec_final.nc"))
          c[:,:,:,idx]       = read_netcdf(curr_path + "c_rec_final.nc")
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
      at[:,:,:,0],im_at[:,:,:,0] = create_prob_img(atlases[:,:,:,0:num_atlases], mris[:,:,:,0:num_atlases])
      _,im_rec[:,:,:,0] = create_prob_img(recon[:,:,:,0:num_atlases], rec_mris[:,:,:,0:num_atlases])
      #im_rec[:,:,:,0] = np.median(rec_mris[:,:,:,0:num_atlases],axis=3)
      #im_at[:,:,:,0] = atlases.sum(axis=3)/num_atlases
      #im_rec[:,:,:,0] = recon.sum(axis=3)/num_atlases
      #im_disp[:,:,:,0] = disp.sum(axis=3)/num_atlases
      #im_c[:,:,:,0] = c.sum(axis=3)/num_atlases
      im_disp[:,:,:,0] = np.median(disp[:,:,:,0:num_atlases], axis=3)
      im_c[:,:,:,0] = np.median(c[:,:,:,0:num_atlases], axis=3)

      ## next is closest to patient vt vol
      im_at[:,:,:,1] = mris[:,:,:,atlas_names.index(min_at)]
      at[:,:,:,1] = atlases[:,:,:,atlas_names.index(min_at)]
      im_rec[:,:,:,1] = rec_mris[:,:,:,atlas_names.index(min_at)]
      im_disp[:,:,:,1] = disp[:,:,:,atlas_names.index(min_at)]
      im_c[:,:,:,1] = c[:,:,:,atlas_names.index(min_at)]

      ## next is farthest to patient vt vol
      im_at[:,:,:,2] = mris[:,:,:,atlas_names.index(max_at)]
      at[:,:,:,2] = atlases[:,:,:,atlas_names.index(max_at)]
      im_rec[:,:,:,2] = rec_mris[:,:,:,atlas_names.index(max_at)]
      im_disp[:,:,:,2] = disp[:,:,:,atlas_names.index(max_at)]
      im_c[:,:,:,2] = c[:,:,:,atlas_names.index(max_at)]

      ## last is median atlas
      im_at[:,:,:,3] = mris[:,:,:,atlas_names.index(med_at)]
      at[:,:,:,3] = atlases[:,:,:,atlas_names.index(med_at)]
      im_rec[:,:,:,3] = rec_mris[:,:,:,atlas_names.index(med_at)]
      im_disp[:,:,:,3] = disp[:,:,:,atlas_names.index(med_at)]
      im_c[:,:,:,3] = c[:,:,:,atlas_names.index(med_at)]

#      df_row = stats.loc[stats["PATIENT"] == pat_name]
#      diff_vol = df_row['diff-vol'].values[0]*100
#      diff_l2  = df_row['diff-l2'].values[0]*100
#      prob_vol = compute_vol((np.abs(im_at[:,:,:,0] - 7) < 2.5E-1))
#      min_vol  = compute_vol(im_at[:,:,:,1] == 7)
#      max_vol  = compute_vol(im_at[:,:,:,2] == 7)
#      med_vol  = compute_vol(im_at[:,:,:,3] == 7)
#      param_list = [pat_name, atlas_failed, diff_vol, diff_l2, vt_pat, prob_vol, min_vol, max_vol, med_vol]
#      for item in param_list:
#        stat_file.write(str(item) + ",")
#      stat_file.write("\n")

      tu     = (pat == 2)
      com    = compute_com(tu)
      axial  = com[2] 
      for idx in range(0,4):
        ax[idx][0].imshow(im_at[:,:,axial,idx].T, cmap='gray', interpolation='none', origin='upper')
        ax[idx][1].imshow(im_rec[:,:,axial,idx].T, cmap='gray', interpolation='none', origin='upper')
        ax[idx][1].imshow(thresh(im_c[:,:,axial,idx].T, cmap_c1, thresh=0.4), cmap=plt.cm.rainbow,interpolation='none', alpha=1) 
        ax[idx][1].contour(tu[:,:,axial].T,  levels=1.0,  cmap=plt.cm.gray, linestyles=['--'] ,linewidths=0.8, alpha=0.9)
        ax[idx][1].contour((at[:,:,:,idx] == 7)[:,:,axial].T,  levels=1.0,  cmap=plt.cm.binary, linestyles=['--'] ,linewidths=0.8, alpha=0.9)
        ax[idx][2].imshow(pat_mri[:,:,axial].T, cmap='gray', interpolation='none', origin='upper')
        ax[idx][2].contour((pat == 2)[:,:,axial].T,  levels=1.0,  cmap=plt.cm.gray, linestyles=['--'] ,linewidths=0.8, alpha=0.9)
        ax[idx][2].contour((pat == 3)[:,:,axial].T,  levels=1.0,  cmap=plt.cm.binary, linestyles=['--'] ,linewidths=0.8, alpha=0.9)
        im = ax[idx][3].imshow(im_disp[:,:,axial,idx].T, vmin=0.01, vmax=max_disp,cmap=plt.cm.coolwarm, interpolation='none', origin='upper') #viridis
        
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
      fig.suptitle("{}".format(pat_name), fontsize="12", fontweight="bold")
      #fig.suptitle("{}; (vol,l2,status)=({:3G},{:3G},{})".format(pat_name, diff_vol, diff_l2, atlas_failed))
      pdf.savefig()
      fig.clf()

  #stat_file.close()
