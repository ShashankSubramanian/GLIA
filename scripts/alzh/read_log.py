import numpy as np
import os, sys
import argparse

parser = argparse.ArgumentParser(description='read objective')
parser.add_argument ('-x',           type = str,          help = 'path to the results folder');
args = parser.parse_args();


DIRS = os.listdir(args.x)
for dir in DIRS:
    if not "inv-" in dir:
        continue;

    n_it = 0
    rho_inv = 0
    k_inv = 0
    eps_k = 0
    eps_rho = 0
    miss = 0
    time = 0
    success=False
    with open(os.path.join(os.path.join(args.x, dir), 'tumor_solver_log.txt')) as f:
        for line in f.readlines():

            if "optimization done:" in line: 
                n_it = int(line.split("#N-it:")[-1].split(",")[0])
                time = float(line.split("exec time:")[-1].split()[0])
                success=True
            if "r1: " in line:
                rho_inv = float(line.split("r1: ")[-1].split(",")[0])
            if "k1: " in line:
                k_inv = float(line.split("k1: ")[-1].split(",")[0])
            if "rel. l2-error at observation" in line:
                miss = float(line.split("at observation points: ")[-1].split()[0])

            if k_inv > 1:
                eps_k = np.abs(k_inv - 1);
            else:
                eps_k = np.abs(k_inv - 0.1) / 0.1
            eps_rho = np.abs(rho_inv - 12) / 12.
    if success:
       print("{0:s} & {1:f} & \\num{{ {2:e} }} & \\num{{ {3:e} }} & \\num{{ {4:e} }} & \\num{{ {5:e} }} & {6:f} & \\num{{ {7:e} }}  ".format(dir, rho_inv, k_inv, eps_rho, eps_k, miss, n_it, time))
    else: 
        print("{} ERROR".format(dir))

