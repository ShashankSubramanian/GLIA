import numpy as np
import os, sys
import argparse

parser = argparse.ArgumentParser(description='read objective')
parser.add_argument ('-x',           type = str,          help = 'path to the results folder');
args = parser.parse_args();


TODO = [
'inv-noise-sp10-iguess[r-8-k-0.01]-fd-lbfgs-3-bounds',
'inv-noise-lres-iguess[r-0-k-0]-fd-lbfgs-3-bounds-scale',
'inv-adv-noise-sp04-iguess[r-0-k-0]-fd-lbfgs-3-bounds',
'inv-noise-sp10-iguess[r-8-k-0.01]-fd-lbfgs-3-bounds-scale',
'inv-noise-sp04-iguess[r-0-k-0]-fd-lbfgs-3-bounds',
'inv-noise-sp10-iguess[r-8-k-0.01]-fd-lbfgs-3-bounds-scale2',
'inv-noise-lres-iguess[r-8-k-0.01]-fd-lbfgs-3-bounds-scale2',
'inv-adv-noise-lres-iguess[r-0-k-0]-fd-lbfgs-3-bounds',
'inv-noise-sp20-iguess[r-0-k-0]-fd-lbfgs-3-bounds',
]

lres_noise = 14
sp04_noise = 22
sp05_noise = 32
sp08_noise = 76
sp09_noise = 96
sp10_noise = 118

DIRS = os.listdir(args.x)
TAB = []

for dir in DIRS:
    if not "inv-" in dir:
        continue;
    #if dir not in TODO:
    #    continue;


    n_it = 0
    rho_inv = 0
    k_inv = 0
    eps_k = 0
    eps_rho = 0
    miss = 0
    time = 0
    descr = ''
    if 'nonoise' in dir or 'noise-no' in dir:
        descr = 'no noise'
    elif 'noise-lres' in dir:
        descr = 'low res (' + str(lres_noise) + '\\%)'
    elif 'noise-sp04' in dir:
        descr = 'sp noise (' + str(sp04_noise) + '\\%)'
    elif 'noise-sp05' in dir:
        descr = 'sp noise (' + str(sp05_noise) + '\\%)'
    elif 'noise-sp08' in dir:
        descr = 'sp noise (' + str(sp08_noise) + '\\%)'
    elif 'noise-sp09' in dir:
        descr = 'sp noise (' + str(sp09_noise) + '\\%)'
    elif 'noise-sp10' in dir:
        descr = 'sp noise (' + str(sp10_noise) + '\\%)'
    if 'adv' in dir:
        descr = 'adv; ' + descr
    meth = dir.split(']-')[-1].split('-')[0] 
    if 'scale2' in dir:
        meth = meth + '-scaled-1e2';
    elif 'scale' in dir:
        meth = meth + '-scaled-1e1';
    init_r = float(dir.split('iguess[r-')[-1].split('-')[0])
    init_k = float(dir.split(']-')[0].split('k-')[-1])
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

            #if k_inv > 1:
            eps_k = np.abs(k_inv - 1);
            #else:
            #    eps_k = np.abs(k_inv - 0.1) / 0.1
            eps_rho = np.abs(rho_inv - 12) / 12.
    if success:
       TAB.append("\\textit{{{0:22s}}}  & \\textit{{{1:14s}}} & \\num{{{2:1.0f}}}  & \\num{{{3:e}}}  & {4:4.2f} & \\num{{{5:e}}} & \\num{{{6:e}}} & \\num{{{7:e}}} & \\num{{{8:e}}} & {9:2d} & \\num{{{10:e}}}  ".format(descr, meth, init_r, init_k,  rho_inv, k_inv, eps_rho, eps_k, miss, n_it, time))
       #print("\\textit{{{0:22s}}}  & \\textit{{{1:14s}}} & \\num{{{2:1.0f}}}  & \\num{{{3:e}}}  & {4:4.2f} & \\num{{{5:e}}} & \\num{{{6:e}}} & \\num{{{7:e}}} & \\num{{{8:e}}} & {9:2d} & \\num{{{10:e}}}  ".format(descr, meth, init_r, init_k,  rho_inv, k_inv, eps_rho, eps_k, miss, n_it, time))
    else: 
        print("{} ERROR".format(dir))


TAB.sort()
for t in TAB:
    print(t, "  \\\ ");
