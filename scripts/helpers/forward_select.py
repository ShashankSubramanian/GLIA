import os, sys, shutil
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../utils/')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import params as par

def get_recon_parameters(recon_file):
  with open(recon_file, "r") as f:
    l = f.readlines()[1]
    params = l.split(" ")

  return params[0:3]

def create_tusolver_config(n, pat, pat_dir, atlas_dir, res_dir, atlas, recon_params):
  r = {}
  p = {}
  submit_job = False;
  case_str    = ""  ### for different cases to go in different dir with this suffix
  n_str       = "_" + str(n)

  atlas                   = atlas.strip("\n")

  p['feature_compute']    = 1 # enables feature comp
  p['n']                  = n
  p['multilevel']         = 1                 # rescale p activations according to Gaussian width on each level
  p['output_dir']         = os.path.join(res_dir, atlas + case_str + '/');    # results path
  p['d1_path']            = ""  # from segmentation directly
  p['d0_path']            = pat_dir + "/" + atlas + '/c0_rec.nc'              # path to initial condition for tumor
  p['atlas_labels']       = "[wm=6,gm=5,vt=7,csf=8]"# brats'[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
  p['patient_labels']     = "[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]"# brats'[wm=6,gm=5,vt=7,csf=8,ed=2,nec=1,en=4]'
  p['a_seg_path']         = atlas_dir + "/" + atlas + "_seg_aff2jakob_ants" + n_str + ".nc"
  p['p_seg_path']         = pat_dir + "/" + pat + '_seg_ants_aff2jakob' + n_str + '.nc'
  p['mri_path']           = atlas_dir + "/" + atlas + "_t1_aff2jakob" + n_str + ".nc"
  p['solver']             = 'forward'
  p['model']              = 4                         # mass effect model
  p['regularization']     = "L2"                      # L2, L1
  p['obs_lambda']         = 1.0                       # if > 0: creates observation mask OBS = 1[TC] + lambda*1[B/WT] from segmentation file
  p['verbosity']          = 3                         # various levels of output density
  p['syn_flag']           = 0                         # create synthetic data
  p['rho_data']           = recon_params[0]           # tumor parameters for synthetic data
  p['k_data']             = recon_params[1]
  p['gamma_data']         = recon_params[2]
  p['nt_data']            = 100
  p['dt_data']            = 0.04
  p['k_gm_wm']            = 0.2                         # kappa ratio gm/wm (if zero, kappa=0 in gm)
  p['r_gm_wm']            = 1                         # rho ratio gm/wm (if zero, rho=0 in gm)
  p['time_history_off']   = 1                         # 1: do not allocate time history (only works with forward solver or FD inversion)
  p['smoothing_factor_data_t0'] = 0
  p['smoothing_factor'] = 1                         # kernel width for smoothing of data and material properties

###############=== write config to write_path and submit job
  par.submit(p, r, submit_job);

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='select an atlas from inversion results and setup forward solver configs',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument ('--inv_dir', type = str, help = 'path to inversion results', required = True) 
  r_args.add_argument ('--atlas_dir', type = str, help = 'path to atlases', required = True) 
  r_args.add_argument ('--res_dir', type = str, help = 'path to results', required = True) 
  r_args.add_argument ('--size', type = int, help = 'size of inversion results', default = 160) 
  args = parser.parse_args();

  sz = args.size
  atlas_dir = args.atlas_dir
  for pat in os.listdir(args.inv_dir):
    inv_path = os.path.join(args.inv_dir, pat)
    if not os.path.exists(inv_path + "/tu/"):
      continue

    print("forward select for pat ", pat)
    stats = os.path.join(*[inv_path, "stat", "stats.csv"])
    if (os.path.exists(stats)):
      with open(stats, "r") as f:
        rep_line = f.readlines()[-1]
        rep_atlas = rep_line.split(" = ")[1].strip()
    else:
      rep_atlas = ""

    src_dir = os.path.join(*[inv_path, "tu", str(sz)])
    dst_dir = os.path.join(*[args.res_dir, pat, "tu", str(sz)])
    if not os.path.exists(dst_dir):
      os.makedirs(dst_dir)
    files_to_move = [s for s in os.listdir(src_dir) if s.find(".nc") is not -1]

    for f in files_to_move:
      src_file = os.path.join(src_dir, f)
      dst_file = os.path.join(dst_dir, f)
      if not os.path.exists(dst_file):
        shutil.copy(src_file, dst_file)

    atlist = os.listdir(src_dir)
    for a in atlist:
      if os.path.exists(os.path.join(*[src_dir, a, "reconstruction_info.dat"])):
        rep_atlas = a

    recon_file = os.path.join(*[src_dir, rep_atlas, "reconstruction_info.dat"])
    if not os.path.exists(recon_file):
      print("recon file does not exist... moving on.")
      continue

    recon_parameters = get_recon_parameters(recon_file)
    atlas_dir_level = atlas_dir + "/" + str(sz) + "/"
    create_tusolver_config(sz, pat, dst_dir, atlas_dir_level, dst_dir, rep_atlas, recon_parameters)

    src_dir = os.path.join(*[inv_path, "tu", str(sz), rep_atlas])
    dst_dir = os.path.join(*[args.res_dir, pat, "tu", str(sz), rep_atlas])
    if not os.path.exists(dst_dir):
      os.makedirs(dst_dir)

    files_to_move = ["c0_rec.nc", "reconstruction_info.dat"] 
    for f in files_to_move:
      src_file = os.path.join(src_dir, f)
      dst_file = os.path.join(dst_dir, f)
      if not os.path.exists(dst_file):
        shutil.copy(src_file, dst_file)
      
