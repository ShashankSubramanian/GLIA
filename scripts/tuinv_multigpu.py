import os, sys
import subprocess

def create_sbatch_header(results_path, idx, compute_sys='longhorn'):
  bash_filename = results_path + "/tuinv-job" + str(idx) + ".sh"
  print("creating job file in ", results_path)

  if compute_sys == 'frontera':
      queue = "normal"
      num_nodes = str(4)
      num_cores = str(128)
  elif compute_sys == 'maverick2':
      queue = "v100"
      num_nodes = str(1)
      num_cores = str(1)
  elif compute_sys == 'longhorn':
      queue = "v100"
      num_nodes = str(1)
      num_cores = str(1)
  elif compute_sys == 'stampede2':
      queue = "skx-normal"
      num_nodes = str(6)
      num_cores = str(128)
  else:
      queue = "normal"
      num_nodes = str(1)
      num_cores = str(1)

  bash_file = open(bash_filename, 'w')
  bash_file.write("#!/bin/bash\n\n");
  bash_file.write("#SBATCH -J tumor-inv\n");
  bash_file.write("#SBATCH -o " + results_path + "/tu_log_" + str(idx) + ".txt\n")
  bash_file.write("#SBATCH -p " + queue + "\n")
  bash_file.write("#SBATCH -N " + num_nodes + "\n")
  bash_file.write("#SBATCH -n " + num_cores + "\n")
  bash_file.write("#SBATCH -t 06:00:00\n\n")
  bash_file.write("source ~/.bashrc\n")

  bash_file.write("\n\n")
  bash_file.close()

  return bash_filename
    
def write_tuinv(invdir, atlist, bin_path, bash_file, idx):
  f = open(bash_file, 'a') 
  f.write("bin_path=" + bin_path + "\n")
  for i in range(1,5):
    f.write("results_dir_" + str(i) + "=" + invdir + atlist[4*idx + i - 1] + "\n")
  f.write("CUDA_VISIBLE_DEVICES=0 ibrun ${bin_path} -config ${results_dir_1}/solver_config.txt > ${results_dir_1}/log &\n")
  f.write("CUDA_VISIBLE_DEVICES=1 ibrun ${bin_path} -config ${results_dir_2}/solver_config.txt > ${results_dir_2}/log &\n")
  f.write("CUDA_VISIBLE_DEVICES=2 ibrun ${bin_path} -config ${results_dir_3}/solver_config.txt > ${results_dir_3}/log &\n")
  f.write("CUDA_VISIBLE_DEVICES=3 ibrun ${bin_path} -config ${results_dir_4}/solver_config.txt > ${results_dir_4}/log &\n")
  f.write("\n")
  f.write("wait\n")

  f.close()
  return bash_file
 
tumor_dir       = os.path.dirname(os.path.realpath(__file__)) + "/../"
bin_path        = tumor_dir + "build/last/tusolver"
pat_names       = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AAP_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1"]
#pat_names       = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1"]
#pat_names       = ["Brats18_CBICA_AAP_1"]
submit_job      = True
suff            = "-noreg"
for patient_name in pat_names:
  invdir   = tumor_dir + "results/" + patient_name + "/tu/"
  atlist   = []
  for atlas in os.listdir(invdir):
    if atlas[0] == "5" and atlas.find(suff) is not -1: ### adni atlas
      atlist.append(atlas)

  numatlas = len(atlist)
  numjobs  = int(numatlas/4)
  
  for i in range(0,numjobs):
    bash_file = create_sbatch_header(invdir, i, compute_sys = "longhorn")
    bash_file = write_tuinv(invdir, atlist, bin_path, bash_file, i)
    if submit_job:
      subprocess.call(['sbatch', bash_file])
     
