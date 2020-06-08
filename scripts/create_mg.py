import os, sys
import subprocess

scripts_path = os.path.dirname(os.path.realpath(__file__)) + "/"
brats_str    = "Brats18_CBICA_"
#names        = ["AMH", "ALU"]
#names        = ["ABO", "AMH"]
names        = ["ABO", "AMH", "ALU", "AAP"]
suff         = ""
submit_job   = False
for patient_name in names:
  res          = brats_str + patient_name + "_1"
  results_path = scripts_path + "../results/" + res
  bin_path     = scripts_path + "../build/last/tusolver"
  suff_2       = ""
  res_suff     = suff_2 + suff
  fname        = results_path + "/multigpu" + suff_2 + ".sh"
  f = open(fname, "w+")
  f.write("#!/bin/bash\n")
  f.write("#SBATCH -J tuinv\n")
  f.write("#SBATCH -o " + results_path + "/log" + res_suff + "\n") 
  f.write("#SBATCH -p v100\n")
  f.write("#SBATCH -N 1\n")
  f.write("#SBATCH -n 1\n")
  f.write("#SBATCH -t 06:00:00\n")
  f.write("source ~/.bashrc\n")
  f.write("export OMP_NUM_THREADS=1\n\n")
  f.write("bin_path=" + bin_path + "\n")
  idx  = ['1', '2', '3', '4']
  idx2 = ['5', '6', '7', '8']
  ct   = 0
  for i in idx:
    if suff_2 == "-2":
      ii = idx2[ct]
    else:
      ii = idx[ct] 
    f.write("results_dir_" + i + "=" + results_path + "/atlas-" + ii + suff + "\n")
    ct += 1
  f.write("\n")
  f.write("CUDA_VISIBLE_DEVICES=0 ibrun ${bin_path} -config ${results_dir_1}/solver_config.txt > ${results_dir_1}/log &\n")
  f.write("CUDA_VISIBLE_DEVICES=1 ibrun ${bin_path} -config ${results_dir_2}/solver_config.txt > ${results_dir_2}/log &\n")
  f.write("CUDA_VISIBLE_DEVICES=2 ibrun ${bin_path} -config ${results_dir_3}/solver_config.txt > ${results_dir_3}/log &\n")
  f.write("CUDA_VISIBLE_DEVICES=3 ibrun ${bin_path} -config ${results_dir_4}/solver_config.txt > ${results_dir_4}/log &\n")
  f.write("\n")
  f.write("wait\n")
  f.close()
  if submit_job:
    subprocess.call(['sbatch', fname])

  suff_2       = "-2"
  res_suff     = suff_2 + suff
  fname        = results_path + "/multigpu" + suff_2 + ".sh"
  f = open(fname, "w+")
  f.write("#!/bin/bash\n")
  f.write("#SBATCH -J tuinv\n")
  f.write("#SBATCH -o " + results_path + "/log" + res_suff + "\n") 
  f.write("#SBATCH -p v100\n")
  f.write("#SBATCH -N 1\n")
  f.write("#SBATCH -n 1\n")
  f.write("#SBATCH -t 06:00:00\n")
  f.write("source ~/.bashrc\n")
  f.write("export OMP_NUM_THREADS=1\n\n")
  f.write("bin_path=" + bin_path + "\n")
  idx  = ['1', '2', '3', '4']
  idx2 = ['5', '6', '7', '8']
  ct   = 0
  for i in idx:
    if suff_2 == "-2":
      ii = idx2[ct]
    else:
      ii = idx[ct] 
    f.write("results_dir_" + i + "=" + results_path + "/atlas-" + ii + suff + "\n")
    ct += 1
  f.write("\n")
  f.write("CUDA_VISIBLE_DEVICES=0 ibrun ${bin_path} -config ${results_dir_1}/solver_config.txt > ${results_dir_1}/log &\n")
  f.write("CUDA_VISIBLE_DEVICES=1 ibrun ${bin_path} -config ${results_dir_2}/solver_config.txt > ${results_dir_2}/log &\n")
  f.write("CUDA_VISIBLE_DEVICES=2 ibrun ${bin_path} -config ${results_dir_3}/solver_config.txt > ${results_dir_3}/log &\n")
  f.write("CUDA_VISIBLE_DEVICES=3 ibrun ${bin_path} -config ${results_dir_4}/solver_config.txt > ${results_dir_4}/log &\n")
  f.write("\n")
  f.write("wait\n")
  f.close()
  if submit_job:
    subprocess.call(['sbatch', fname])
