import os, sys
import numpy as np
from numpy import linalg as LA
import re

res_path = os.path.dirname(os.path.realpath(__file__)) + "/../../results"
#pat_names = ["Brats18_CBICA_ABO_1"]
#pat_names = ["Brats18_CBICA_ABO_1", "Brats18_CBICA_AAP_1", "Brats18_CBICA_AMH_1", "Brats18_CBICA_ALU_1"]
pat_names = ["atlas-2-case2", "atlas-2-case3"]
inv_suff  = "-hp"
g_scale = 1E4
r_scale = 1
k_scale = 1E-2
match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee][+-]?\ *-?\ *[0-9]+)?')
for pat in pat_names:
  print("scrubbing hessian cond for pat {}".format(pat))
  stat_dir = res_path + "/stat-" + pat + "/"
  hess_f   = stat_dir + "hessian_cond_ic.txt"
  hess_f_eig   = stat_dir + "hessian_cond_ic_eig.txt"
  f = open(hess_f, "w+")
  f_eig = open(hess_f_eig, "w+")
  hess_f_a   = stat_dir + "hessian_cond_opt.txt"
  hess_f_a_eig   = stat_dir + "hessian_cond_opt_eig.txt"
  fopt_eig = open(hess_f_a_eig, "w+")
  fopt = open(hess_f_a, "w+")
  for i in range(1,9):
    atlas    = "atlas-" + str(i) + inv_suff
    inv_res  = res_path + "/inv-" + pat + "/" + atlas

    ### scrub log file to find the points
    log_file = inv_res + "/log"
    start_idx = 0
    with open(log_file) as fl:
      lines = fl.readlines()
      for line in lines:
        if "estimated" in line:
          break;
        start_idx += 1

      line = lines[72-1]
      l = re.findall("\d*\.?\d+", line)
      g_ic = float(l[0]) * g_scale
      line = lines[73-1]
      l = re.findall("\d*\.?\d+", line)
      r_ic = float(l[0]) * r_scale
      line = lines[74-1]
      l = re.findall("\d*\.?\d+", line)
      k_ic = float(l[0]) * k_scale
#      line = lines[24]
#      l = re.findall("\d*\.?\d+", line)
#      g_ic = float(l[0]) * g_scale
#      line = lines[25]
#      l = re.findall("\d*\.?\d+", line)
#      r_ic = float(l[0]) * r_scale
#      line = lines[26]
#      l = re.findall("\d*\.?\d+", line)
#      k_ic = float(l[0]) * k_scale
      
      line = lines[start_idx+1]
      l = re.findall("\d*\.?\d+", line)
      gamma = float(l[1])
      line = lines[start_idx+3]
      l = re.findall("\d*\.?\d+", line)
      rho = float(l[2])
      line = lines[start_idx+5]
      l = re.findall("\d*\.?\d+", line)
      kappa = float(l[2])

    raw      = []
    with open(inv_res + "/hessian_IC.txt", "r") as f1:
      for line in f1:
        raw.append(line.split())
    hess = np.array(raw)
    hess = hess.astype(float)
    cond = LA.cond(hess)
    f.write("{},{},{},{}\n".format(g_ic, r_ic, k_ic, cond))
    f_eig.write("Atlas-{}\n".format(i))
    f_eig.write("Point: {},{},{}\n".format(g_ic, r_ic, k_ic))
    f_eig.write("Hessian:\n")
    f_eig.write("{}\n".format(hess))
    f_eig.write("Condition number: {}\n".format(cond))
    eigval, eigvec = LA.eig(hess)
    f_eig.write("eigvals: {}\n".format(eigval))
    f_eig.write("eigvecs: \n")
    f_eig.write("{}\n".format(eigvec))
    f_eig.write("##########\n\n")
    raw      = []
    with open(inv_res + "/hessian_opt.txt", "r") as f1:
      for line in f1:
        raw.append(line.split())
    hess = np.array(raw)
    hess = hess.astype(float)
    cond = LA.cond(hess)
    fopt.write("{},{},{},{}\n".format(gamma, rho, kappa, cond))
    fopt_eig.write("Atlas-{}\n".format(i))
    fopt_eig.write("Point: {},{},{}\n".format(gamma, rho, kappa))
    fopt_eig.write("Hessian:\n")
    fopt_eig.write("{}\n".format(hess))
    fopt_eig.write("Condition number: {}\n".format(cond))
    eigval, eigvec = LA.eig(hess)
    fopt_eig.write("eigvals: {}\n".format(eigval))
    fopt_eig.write("eigvecs: \n")
    fopt_eig.write("{}\n".format(eigvec))
    fopt_eig.write("##########\n\n")
  f_eig.close()
  fopt_eig.close()
  f.close()
  fopt.close()
