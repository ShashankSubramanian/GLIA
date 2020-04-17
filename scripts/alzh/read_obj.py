import numpy as np
import os, sys
import argparse

parser = argparse.ArgumentParser(description='read objective')
parser.add_argument ('-path',           type = str,          help = 'path to the results folder');
args = parser.parse_args();


obj_nit = []
rho_nit = []
kap_nit = []
obj_ls  = []
rho_ls  = []
kap_ls  = []
read=False
with open(os.path.join(args.path, 'tumor_solver_log.txt')) as f:
    for line in f.readlines():
        if "objective (abs)" in line: 
            read=True
            continue
        if read:
            if ("[1;31m[" in line) and (not "converged" in line) and (not "maximum number" in line):
                obj_nit.append(float(line.split()[2]))
                rho_nit.append(float(line.split()[6]))
                kap_nit.append(float(line.split()[7].split(']')[0]))
            #if "J(p) = Dc(c1)" in line:i
            #    obj_ls.append(float(line.split('=')[2]))
            #if "Reaction guess" in line:
            #    rho_ls.append(float(line.split('= (')[-1].split(',')[0]))
            #if "Diffusivity guess" in line:
            #    kap_ls.append(float(line.split('= (')[-1].split(',')[0]))


print("\nobjective: \n{}".format(obj_nit))
print("\nrho: \n{}".format(rho_nit))
print("\nkappa: \n{}".format(kap_nit))
