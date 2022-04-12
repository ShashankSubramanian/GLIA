import os, sys, shutil
import argparse

def copy_inv_til(args):
  for pat in os.listdir(args.patient_dir):
    '''
    if not os.path.exists(os.path.join(*[args.patient_dir, pat, "aff2jakob", pat + "_t1_aff2jakob.nii.gz"])):
      continue
    '''
    til = os.path.join(*[args.til_dir, "inversion/nx256/obs-1.0/c0_rec_256256256.nii.gz"]) 
    if not os.path.exists(til):
      print("ERROR! pat {} does not have til at loc {}".format(pat, til))
    else:
      dst = os.path.join(*[args.patient_dir, pat + "_c0Recon_aff2jakob.nii.gz"])
      print("copying TIL {} to DIR {}".format(til, dst)) 
      shutil.copy(til, dst) 

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='copy TILs to data dir for mass effect inversion',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument ('--patient_dir', type = str, help = 'path to all patients', required = True) 
  r_args.add_argument ('--til_dir', type = str, help = 'path to all patients TIL recon results', required = True) 
  args = parser.parse_args();

  copy_inv_til(args)
