import os, sys, shutil
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../utils/')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import params as par


def update_tu_config(config_file):
  with open(config_file, "r") as f: 
    lines = f.readlines()

  # modify some params
  smooth = 1
  k = 0.01
  with open(config_file, "w+") as f: 
    for line in lines:
      if "smoothing_factor=" in line:
        f.write("smoothing_factor="+str(smooth)+"\n")
        continue
      if "k_data=" in line:
        f.write("k_data="+str(k)+"\n")
        continue
      f.write(line)

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='runs several forward solves',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  r_args = parser.add_argument_group('required arguments')
  r_args.add_argument ('--run_dir', type = str, help = 'path to patients configs', required = True) 
  r_args.add_argument ('--job_dir', type = str, help = 'path to job outputs', required = True) 
  r_args.add_argument ('--code_dir', type = str, help = 'path to code', required = True) 
  r_args.add_argument ('--sz', type = int, help = 'size of images', default = 160) 
  args = parser.parse_args();


  tu_params = {}
  tu_params["output_dir"] = args.job_dir
  run_params = {}
  run_params["compute_sys"] = "longhorn"
  run_params["wtime_h"] = 2
  run_params["wtime_m"] = 0

  num_grps = 10
  num_gpus = 4 # per compute node
  num_pats_per_job = num_gpus * num_grps

  if not os.path.exists(args.job_dir):
    os.makedirs(args.job_dir)

  pat_list = []
  run_list = []
  fail_string = ""
  #fail_string = "fail_"
  if len(fail_string) > 0:
    with open(os.path.join(args.run_dir, "failed.txt"), "r") as f:
      lines = f.readlines()
    for l in lines:
      run_list.append(l.strip("\n"))
  else:
    run_list = os.listdir(args.run_dir)

  for pat in run_list:
    fwd_path = os.path.join(*[args.run_dir, pat, "tu", str(args.sz)])
    if not os.path.exists(fwd_path):
      continue
    subdirs = os.listdir(fwd_path)
    for a in subdirs:
      if os.path.exists(os.path.join(*[fwd_path, a, "reconstruction_info.dat"])):
        rep_atlas = a
        break
    if not os.path.exists(os.path.join(*[fwd_path, rep_atlas, "solver_config.txt"])):
      continue

    if len(fail_string) > 0:
      # update tu config to try other variations
      print("updating config in ", fwd_path)
      update_tu_config(os.path.join(*[fwd_path, rep_atlas, "solver_config.txt"]))

    pat_list.append((pat,rep_atlas))

  total_pats = len(pat_list)
  num_jobs = int(np.ceil(total_pats/num_pats_per_job))
  rem = total_pats%num_pats_per_job

  print("patients: " ,pat_list)
  print("divided into ", num_jobs)

  for idx in range(0,num_jobs):
    ender = (idx+1)*num_pats_per_job if (idx+1)*num_pats_per_job <= total_pats else idx*num_pats_per_job + rem
    local_list = pat_list[idx*num_pats_per_job:ender]
    local_len = len(local_list)
    num_local_grps = int(np.ceil(local_len/num_gpus))
    run_params["log_name"] = "log_" + str(idx)
    with open(args.job_dir + "/job_" + fail_string + str(idx) + ".sh", "w+") as f:
      header = par.write_jobscript_header(tu_params, run_params, use_gpu=True) 
      f.write(header)
      f.write("\nbin_path=" + args.code_dir + "/build/last/tusolver\n\n")
      for g_idx in range(0,num_local_grps):
        grp_ender = (g_idx+1)*num_gpus if (g_idx+1)*num_gpus <= local_len else g_idx*num_gpus + (local_len%num_gpus)
        grp_list = local_list[g_idx*num_gpus:grp_ender]
        gpu_idx = 0
        for pair in grp_list:
          pat = pair[0]
          rep_atlas = pair[1]
          out_dir = os.path.join(*[args.run_dir, pat, "tu", str(args.sz), rep_atlas])
          f.write("results_dir_" + str(gpu_idx) + "=" + out_dir + "\n")
          cmd = par.runcmd(run_params)
          f.write("CUDA_VISIBLE_DEVICES=" + str(gpu_idx) + " " + cmd + "${bin_path} -config ${results_dir_" + str(gpu_idx) + "}/solver_config.txt > ${results_dir_" + str(gpu_idx) + "}/solver_log.txt 2>&1 &\n")
          gpu_idx += 1
        f.write("\n")
        f.write("wait\n")

    
