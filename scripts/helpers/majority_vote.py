import os, sys, warnings, argparse, subprocess
import nibabel as nib
import numpy as np
from numpy import linalg as LA


def writeNII(img, filename, affine=None, ref_image=None):
    '''
    function to write a nifti image, creates a new nifti object
    '''
    if ref_image is not None:
        data = nib.Nifti1Image(img, affine=ref_image.affine, header=ref_image.header);
        data.header['datatype'] = 64
        data.header['glmax'] = np.max(img)
        data.header['glmin'] = np.min(img)
    elif affine is not None:
        data = nib.Nifti1Image(img, affine=affine);
    else:
        data = nib.Nifti1Image(img, np.eye(4))

    nib.save(data, filename);

def normalize(x):
  x -= x.mean()
  x /= x.std()
  x = x.clip(-np.percentile(x,99.9), np.percentile(x,99.9))
  x = (x - np.min(x))/(np.max(x) - np.min(x))
  return x

def majorityvote(flatseg, sz = [240,240,155]):
  fsz = np.prod(sz)
  labels = np.zeros(fsz)
  numseg = np.shape(flatseg)[-1]
#  wts = []
#  for i in range(0,numat):
#    at[:,i] = normalize(at[:,i])
#    wts.append(1 / LA.norm(pat - at[:,i]))
#
#  print(wts)

  for i in range(0,fsz):
    lst = flatseg[i,:].tolist()
#    votes = [wts[k]*lst.count(lst[k]) for k in range(0,len(lst))]
    labels[i] = max(lst, key=lst.count)
#    labels[i] = lst[np.argmax(np.array(votes))]

  return labels.reshape(tuple(sz))

#--------------------------------------------------------------------------------------------------------------------------
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='Majority voting for segmentation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument('--patient_dir', type = str, help = 'path to patient folders containing images', required = True)
  r_args.add_argument('--patient', type = str, help = 'patient_name', required = True)
  r_args.add_argument('--model_list', type = str, help = 'path to file containing list of output folders for the different models', required = True)
  r_args.add_argument('--seg_dir', type = str, help = 'path to folders containing inference results', required = True)
  args = parser.parse_args();

  model_list = []
  with open(args.model_list, "r") as f:
    lines = f.readlines()
  for l in lines:
    model_list.append(l.strip("\n"))

  sz = [240,240,155]  ## TODO change this by getting the shape of the input
  flat_seg = np.zeros((np.prod(sz),len(model_list))) 
  patient = args.patient
  print("majority voting for patient ", patient)
  ## go through every model seg
  for mc, model in enumerate(model_list):
    seg_nii = nib.load(os.path.join(args.seg_dir, model) + "/predictions/" + patient + "_seg.nii.gz")
    seg = seg_nii.get_fdata().flatten()
    flat_seg[:,mc] = seg
  labels = majorityvote(flat_seg, sz)
  writeNII(labels, os.path.join(args.patient_dir, patient) + "/" + patient + "_seg_majorityvote.nii.gz", ref_image = seg_nii)
  print("done for patient ", patient)




