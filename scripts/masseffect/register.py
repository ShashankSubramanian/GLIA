import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
import nibabel as nib
import nibabel.processing
import scipy as sc
from scipy.ndimage import gaussian_filter
from netCDF4 import Dataset
from numpy import linalg as la
import math

sys.path.append('../utils/')
from file_io import writeNII, createNetCDF
from image_tools import resizeImage, resizeNIIImage
sys.path.append('../')

def create_patient_labels(patient_image_path, results_path, patient_name):
  nii = nib.load(patient_image_path)
  patient_seg = nii.get_fdata()
  if not os.path.exists(results_path):
    os.makedirs(results_path)
  writeNII(patient_seg, results_path + "/" + patient_name + "_labels.nii.gz", ref_image=nii)
  patient_mat_img = 0 * patient_seg
  patient_mat_img[patient_seg == 7] = 1
  writeNII(patient_mat_img, results_path + "/" + patient_name + "_vt.nii.gz", ref_image=nii)
  patient_mat_img = 0 * patient_seg
  patient_mat_img[patient_seg == 8] = 1
  writeNII(patient_mat_img, results_path + "/" + patient_name + "_csf.nii.gz", ref_image=nii)
  patient_mat_img = 0 * patient_seg
  patient_mat_img[patient_seg == 5] = 1
  writeNII(patient_mat_img, results_path + "/" + patient_name + "_gm.nii.gz", ref_image=nii)
  patient_mat_img = 0 * patient_seg
  patient_mat_img[(np.logical_or(patient_seg == 6, patient_seg == 8))] = 1
  writeNII(patient_mat_img, results_path + "/" + patient_name + "_wm_csf.nii.gz", ref_image=nii)
  patient_mat_img = 0 * patient_seg + 1
  patient_mat_img[np.logical_or(np.logical_or(patient_seg == 4, patient_seg == 1), patient_seg == 2)] = 0 #WT mask
  patient_mat_img = gaussian_filter(patient_mat_img, sigma=2) # claire requires smoothing of masks
  writeNII(patient_mat_img, results_path + "/" + patient_name + "_mask.nii.gz", ref_image=nii)

def create_atlas_labels(atlas_image_path, results_path, atlas_name):
  nii = nib.load(atlas_image_path)
  altas_seg = nii.get_fdata()
  if not os.path.exists(results_path):
    os.makedirs(results_path)
  writeNII(altas_seg, results_path + "/" + atlas_name + "_labels.nii.gz", ref_image=nii)
  altas_mat_img = 0 * altas_seg
  altas_mat_img[altas_seg == 7] = 1
  writeNII(altas_mat_img, results_path + "/" + atlas_name + "_vt.nii.gz", ref_image=nii)
  altas_mat_img = 0 * altas_seg
  altas_mat_img[altas_seg == 8] = 1
  writeNII(altas_mat_img, results_path + "/" + atlas_name + "_csf.nii.gz", ref_image=nii)
  altas_mat_img = 0 * altas_seg
  altas_mat_img[altas_seg == 5] = 1
  writeNII(altas_mat_img, results_path + "/" + atlas_name + "_gm.nii.gz", ref_image=nii)
  altas_mat_img = 0 * altas_seg
  altas_mat_img[np.logical_or(altas_seg == 6, altas_seg == 8)] = 1
  writeNII(altas_mat_img, results_path + "/" + atlas_name + "_wm_csf.nii.gz", ref_image=nii)
    

def register(claire_bin_path, results_path, atlas_name, pat_path, patient_name, bash_filename, idx, multigpu = True):
  bash_file = open(bash_filename, 'a')
  if multigpu:
    cmd = "CUDA_VISIBLE_DEVICES=" + str(idx) + " "
  else:
    cmd = ""

  ##cmd += "ibrun " + claire_bin_path + "/claire -mrc 3 " + results_path + "/" + atlas_name + "_vt.nii.gz " + results_path + "/" + atlas_name + "_gm.nii.gz " + results_path + "/" + atlas_name + "_wm_csf.nii.gz " + "-mtc 3 " + pat_path + "/" + patient_name + "_vt.nii.gz " + pat_path + "/" + patient_name + "_gm.nii.gz " + pat_path + "/" + patient_name + "_wm_csf.nii.gz " + "-mask " + pat_path + "/" + patient_name + "_mask.nii.gz " + "-nx 256 -train binary -jbound 0.2 -regnorm h1s-div -opttol 5e-2 -maxit 25 -krylovmaxit 50 -beta-div 1e-4 -velocity -detdefgrad -deffield -defmap -residual -x " + results_path + "/" + " -monitordefgrad -verbosity 1 -disablerescaling -format nifti -sigma 2" + " > " + results_path + "/registration_log.txt &";
  cmd += "ibrun " + claire_bin_path + "/claire -mrc 3 " + results_path + "/" + atlas_name + "_vt.nii.gz " + results_path + "/" + atlas_name + "_gm.nii.gz " + results_path + "/" + atlas_name + "_wm_csf.nii.gz " + "-mtc 3 " + pat_path + "/" + patient_name + "_vt.nii.gz " + pat_path + "/" + patient_name + "_gm.nii.gz " + pat_path + "/" + patient_name + "_wm_csf.nii.gz " + "-mask " + pat_path + "/" + patient_name + "_mask.nii.gz " + "-nx 256 -train binary -jbound 0.2 -regnorm h1s-div -opttol 5e-2 -maxit 25 -krylovmaxit 50 -beta-div 1e-4 -velocity -deffield -x " + results_path + "/" + " -monitordefgrad -verbosity 1 -disablerescaling -format nifti -sigma 2" + " > " + results_path + "/registration_log.txt &";
  bash_file.write(cmd)
  bash_file.write("\n\n")
  bash_file.close()
  return bash_filename

# transport
def transport(claire_bin_path, results_path, trans_image, name, bash_filename, idx, r2t=False, multigpu=True):
  bash_file = open(bash_filename, 'a')
  if multigpu:
    cmd = "CUDA_VISIBLE_DEVICES=" + str(idx) + " "
  else:
    cmd = ""

  transport_image = results_path + "/" + name + "_transported.nii.gz"

  if r2t == True:
    if name.find("labels") is not -1:
      cmd += "ibrun " + claire_bin_path + "/clairetools -v1 " + results_path + "/velocity-field-x1.nii.gz -v2 " + results_path + "/velocity-field-x2.nii.gz -v3 " + results_path + "/velocity-field-x3.nii.gz -r2t -tlabelmap -labels 0,1,2,4,5,6,7,8 -ifile " + trans_image + " -xfile " + transport_image + " -iporder 1" + " > " + results_path + "/transport_log_" + name + ".txt &";
  else:
### enable -r2t if reference to transport is needed. keep disabled for now.
    if name.find("labels") is not -1:
      cmd += "ibrun " + claire_bin_path + "/clairetools -v1 " + results_path + "/velocity-field-x1.nii.gz -v2 " + results_path + "/velocity-field-x2.nii.gz -v3 " + results_path + "/velocity-field-x3.nii.gz -tlabelmap -labels 0,1,2,4,5,6,7,8 -ifile " + trans_image + " -xfile " + transport_image + " -iporder 1" + " > " + results_path + "/transport_log_" + name + ".txt &";
    else:
      cmd += "ibrun " + claire_bin_path + "/clairetools -v1 " + results_path + "/velocity-field-x1.nii.gz -v2 " + results_path + "/velocity-field-x2.nii.gz -v3 " + results_path + "/velocity-field-x3.nii.gz -ifile " + trans_image + " -xfile " + transport_image + " -deformimage -iporder 1" + " > " + results_path + "/transport_log_" + name + ".txt &";

  bash_file.write(cmd)
  bash_file.write("\n\n")
  bash_file.close()

  return bash_filename

def convert_and_move_moi(path, out_path, patient_name, refniiimage):
  file = Dataset(path, mode='r', format="NETCDF3_CLASSIC")
  c0 = np.transpose(file.variables['data'])
  nii = nib.load(refniiimage) 
  writeNII(c0, out_path + "/" + patient_name + "_c0Recon.nii.gz", ref_image=nii)


if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Process images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument ('-tp',   '--transport', type = int, help = 'atlas list', default = 0) 
  r_args.add_argument ('-mg',   '--multigpu', type = int, help = 'use multiple gpus', default = 1) 
  args = parser.parse_args();
  tp = args.transport
  mg = args.multigpu

  tumor_dir       = os.path.dirname(os.path.realpath(__file__)) + "/../../"
  pat_names = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AAP_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1"]
  submit_job      = True
  for patient_name in pat_names:
    patient_path    = tumor_dir + "brain_data/real_data/" + patient_name + "/data/" + patient_name + "_seg_tu_aff2jakob.nii.gz"
    map_of_interest = tumor_dir + "brain_data/real_data/" + patient_name + "/data/" + patient_name + "_c0Recon_aff2jakob.nc"

    patient         = nib.load(patient_path).get_fdata()
    atlas_path      = tumor_dir + "../data/adni/"
    pat_path        = tumor_dir + "results/" + patient_name + "/"
    results_path    = tumor_dir + "results/" + patient_name + "/reg/"
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    claire_path = tumor_dir + "../claire-dev/bingpu/"
    listfile    = tumor_dir + "results/" + patient_name + "/atlas_list"
    if not os.path.exists(listfile):
      f  = open(listfile, "w+")
      fa = open(atlas_path + "/adni-atlas-stats.csv", "r")
      fp = open(tumor_dir + "/brain_data/cbica-pat-stats.csv", "r")
      la = fa.readlines()
      lp = fp.readlines()
      for l in lp:
        if patient_name in l:
          vt_pat = float(l.split(",")[1])

      for l in la:
        vals = l.split(",")
        if float(vals[1]) >= vt_pat and float(vals[1]) < 1.5*vt_pat: ### choose atlases whose vt vol is greater than pat
          f.write(vals[0] + "\n")

      f.close()
      fa.close()
      fp.close()

    f = open(listfile, "r")
    atlases = f.readlines()
    min_at  = 16
    num_at  = len(atlases)
    if num_at < min_at:
      min_at = int(num_at/4) * 4

#      create_patient_labels(patient_path, pat_path, patient_name)
#      convert_and_move_moi(map_of_interest, pat_path, patient_name, patient_path)
#      for idx in range(0, int(min_at/4)):
#        bash_file  = create_sbatch_header(results_path, idx, compute_sys = "longhorn")
#        for i in range(0,4):
#          atlas_name = atlases[4*idx + i].strip("\n")
#          atlas      = atlas_name + "_seg_aff2jakob_ants.nii.gz"
#          atlas_dir  = results_path + atlas_name
#          if not os.path.exists(atlas_dir):
#            os.makedirs(atlas_dir)
#          create_atlas_labels(atlas_path + atlas, atlas_dir, atlas_name)
#          ### create reg job scripts
#          bash_file = register(claire_path, atlas_dir, atlas_name, pat_path, patient_name, bash_file, i, multigpu=mg)
#        fio = open(bash_file, 'a')
#        fio.write("wait\n\n")
#        fio.close()

    for idx in range(0, int(min_at/4)):
#        bash_file = results_path + "/registration-job-" + str(idx) + ".sh"
      for i in range(0,4):
        atlas_name = atlases[4*idx + i].strip("\n")
        atlas      = atlas_name + "_seg_aff2jakob_ants.nii.gz"
        atlas_dir  = results_path + atlas_name
        ### create transport scripts 
#          bash_file = transport(claire_path, atlas_dir, pat_path + patient_name + "_c0Recon.nii.gz", "c0Recon", bash_file, i, r2t=False, multigpu=mg)
#          bash_file = transport(claire_path, atlas_dir, pat_path + patient_name + "_labels.nii.gz", patient_name + "_labels", bash_file, i, r2t=False, multigpu=mg)
#          bash_file = transport(claire_path, atlas_dir, atlas_dir + "/" + atlas_name + "_labels.nii.gz", atlas_name + "_labels", bash_file, i, r2t=True, multigpu=mg)

        ### some postproc: keep comments
        c0trans = nib.load(atlas_dir + "/c0Recon_transported.nii.gz").get_fdata()
        createNetCDFFile(atlas_dir + "/c0Recon_transported.nc", 256 * np.ones(3), np.transpose(c0trans))


#        fio = open(bash_file, 'a')
#        fio.write("wait\n\n")
#        fio.close()
#
#        if submit_job:
#          subprocess.call(['sbatch', bash_file])
