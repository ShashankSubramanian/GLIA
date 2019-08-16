import matplotlib as mpl
import os
import numpy as np
import argparse
import shutil
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import scipy


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='process BRATS results')
    parser.add_argument ('-x',  '--dir', type = str, help = 'path to the results folder');
    args = parser.parse_args();

    level = int(args.dir.split("nx")[-1].split("/")[0])
    np_    = 0;

    logfile        = 'tumor_solver_log_nx' + str(level) + '.txt'
    grad_file      = 'g_it.dat'
    sol_file       = 'x_it.dat'
    glob_grad_file = 'glob_g_it.dat'
    SUPP           = {}
    INIT_SUPP      = []
    INIT_COMP      = []
    COMP           = {}
    COLOR          = ['g', 'r', 'm', 'y']
    GRAD_SOLVE_IT      = {}
    GRAD_DIFF_SOLVE_IT = {}
    SOL_SOLVE_IT       = {}
    SOL_DIFF_SOLVE_IT  = {}
    sol_diff_norms_pi  = {}
    sol_diff_norms_k   = {}
    grad_diff_norms_pi = {}
    grad_diff_norms_k  = {}
    # init
    for i in range(3):
        GRAD_SOLVE_IT[i] = {}
        GRAD_DIFF_SOLVE_IT[i] = {}
        SOL_SOLVE_IT[i] = {}
        SOL_DIFF_SOLVE_IT[i] = {}

    j=0
    if os.path.exists(os.path.join(args.dir, logfile)):
        with open(os.path.join(args.dir, logfile), 'r') as f:
            lines = f.readlines();
            no = 0;
            for line in lines:
                if "----- np:" in line:
                    np_ = int(line.split("----- np:")[-1].split("------")[0])
                if "starting CoSaMP solver with initial support:" in line:
                    supp = line.split("starting CoSaMP solver with initial support: [")[-1].split("]")[0].split()
                    for s in supp:
                        INIT_SUPP.append(int(s))
                if "component label of initial support :" in line:
                    comp = line. split("component label of initial support : [")[-1].split("]")[0].split()
                    for c in comp:
                        INIT_COMP.append(int(c))
                if "support for corrective L2 solve :" in line:
                    supp = line.split("support for corrective L2 solve : [")[-1].split("]")[0].split()
                    SUPP[j] = []
                    for s in supp:
                        SUPP[j].append(int(s))
                    comp = lines[no+1]. split("component label of support : [")[-1].split("]")[0].split()
                    COMP[j] = []
                    for c in comp:
                        COMP[j].append(int(c))
                    j += 1
                no += 1;

    # fetch global gradient
    print("np:",np_)
    if os.path.exists(os.path.join(args.dir, glob_grad_file)):
        with open(os.path.join(args.dir, glob_grad_file), 'r') as f:
            lines = f.readlines();
            G = {};
            j = 0;
            for line in lines:
                if not line.strip():
                    continue;
                L = line.split(";")[0].split(",")
                nl = len(L);
                G[j] = []
                # print("number entries:", nl, ", number glob_grads: ", int(nl/(np_-1)));
                for i in range(0, nl):
                    G[j].append(float(L[i]))
                j += 1;

    # fetch gradient of subspace
    nb_solves = -1;
    j=0;
    if os.path.exists(os.path.join(args.dir, grad_file)):
        with open(os.path.join(args.dir, grad_file), 'r') as f:
            lines = f.readlines();
            it = 0;
            for line in lines:
                if not line.strip():
                    continue;
                if "## ----- ##" in line:
                    nb_solves += 1;
                    it = 0;
                    continue;

                L = line.split(";")[0].split(",")
                GRAD_SOLVE_IT[nb_solves][it] = []
                for l in L:
                    GRAD_SOLVE_IT[nb_solves][it].append(float(l))
                GRAD_SOLVE_IT[nb_solves][it] = np.asarray(GRAD_SOLVE_IT[nb_solves][it]);
                it +=1


    # fetch sol of subspace
    nb_solves = -1;
    if os.path.exists(os.path.join(args.dir, sol_file)):
        with open(os.path.join(args.dir, sol_file), 'r') as f:
            lines = f.readlines();
            it = 0;
            for line in lines:
                if not line.strip():
                    continue;
                if "## ----- ##" in line:
                    nb_solves += 1;
                    it = 0;
                    continue;

                L = line.split(";")[0].split(",")
                SOL_SOLVE_IT[nb_solves][it] = []
                for l in L:
                    SOL_SOLVE_IT[nb_solves][it].append(float(l))
                SOL_SOLVE_IT[nb_solves][it] = np.asarray(SOL_SOLVE_IT[nb_solves][it]);
                it +=1


    # compute differences
    print(bcolors.OKBLUE + " computing gradient and iterate differences:" + bcolors.ENDC);
    for nb_solves in range(3):
        print("\n -- L2 #%d (%d iterations) --" % ((nb_solves+1), len(GRAD_SOLVE_IT[nb_solves])))
        sol_diff_norms_pi[nb_solves]  = []
        sol_diff_norms_k[nb_solves]   = []
        grad_diff_norms_pi[nb_solves] = []
        grad_diff_norms_k[nb_solves]  = []
        for it in range(1,len(GRAD_SOLVE_IT[nb_solves])):
            GRAD_DIFF_SOLVE_IT[nb_solves][it-1] = GRAD_SOLVE_IT[nb_solves][it] - GRAD_SOLVE_IT[nb_solves][it-1]
            SOL_DIFF_SOLVE_IT[nb_solves][it-1]  = SOL_SOLVE_IT[nb_solves][it]  - SOL_SOLVE_IT[nb_solves][it-1]
            sol_diff_norms_pi[nb_solves].append(np.linalg.norm(SOL_DIFF_SOLVE_IT[nb_solves][it-1][:-1]))
            sol_diff_norms_k[nb_solves].append(np.linalg.norm(SOL_DIFF_SOLVE_IT[nb_solves][it-1][-1]))
            grad_diff_norms_pi[nb_solves].append(np.linalg.norm(GRAD_DIFF_SOLVE_IT[nb_solves][it-1][:-1]))
            grad_diff_norms_k[nb_solves].append(np.linalg.norm(GRAD_DIFF_SOLVE_IT[nb_solves][it-1][-1]))


        print("l2norm diff p_it:    ", sol_diff_norms_pi[nb_solves]);
        print("l2norm diff k_it:    ", sol_diff_norms_k[nb_solves]);
        print("l2norm diff g(p_it): ", grad_diff_norms_pi[nb_solves]);
        print("l2norm diff g(k_it): ", grad_diff_norms_k[nb_solves]);

    mpl.style.use('seaborn')
    sns.set(style="ticks")
    f = plt.figure(figsize=(8,8))
    grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.4, height_ratios=[2,2,2])

    kappa          = []
    kappa_grad     = []
    kappa_diff     = []
    sol_norm       = []
    sol_diff_norm  = []
    grad_norm      = []
    grad_diff_norm = []
    for i in range(3):
        kappa_diff.extend([SOL_DIFF_SOLVE_IT[i][it][-1] for it in range(len(SOL_DIFF_SOLVE_IT[i]))])
        kappa.extend([SOL_SOLVE_IT[i][it][-1] for it in range(len(SOL_SOLVE_IT[i]))])
        sol_norm.extend([np.linalg.norm(SOL_SOLVE_IT[i][it][:-1]) for it in range(len(SOL_SOLVE_IT[i]))])
        sol_diff_norm.extend(sol_diff_norms_pi[i])
        grad_norm.extend([np.linalg.norm(GRAD_SOLVE_IT[i][it][:-1]) for it in range(len(GRAD_SOLVE_IT[i]))])
        kappa_grad.extend([GRAD_SOLVE_IT[i][it][-1] for it in range(len(GRAD_SOLVE_IT[i]))])
        grad_diff_norm.extend(grad_diff_norms_pi[i])
    xx = np.linspace(0,len(kappa),len(kappa))


    ax = plt.subplot(grid[0,0])
    ax.step(np.linspace(0,len(kappa),len(kappa)), kappa, label='$\\|\\kappa_{i}\\|$');
    ax.set_yscale('log')
    ax.legend()

    ax = plt.subplot(grid[0,1])
    ax.step(np.linspace(0,len(kappa_diff),len(kappa_diff)), [abs(x) for x in kappa_diff], label='$\\|\\kappa_{i}-\\kappa_{i-1}\\|$');
    ax.set_yscale('log')
    ax.legend()


    ax = plt.subplot(grid[1,0])
    ax.step(np.linspace(0,len(grad_norm),len(grad_norm)), grad_norm, label='$\\|g_i\\|$');
    ax.step(np.linspace(0,len(sol_norm),len(sol_norm)), sol_norm, label='$\\|x_{i}\\|$');
    ax.set_yscale('log')
    ax.legend()

    ax = plt.subplot(grid[1,1])
    ax.step(np.linspace(0,len(grad_diff_norm),len(grad_diff_norm)), grad_diff_norm, label='$\\|g_{i}-g_{i-1}\\|$');
    ax.step(np.linspace(0,len(sol_diff_norm),len(sol_diff_norm)), sol_diff_norm, label='$\\|x_{i}-x_{i-1}\\|$');
    ax.set_yscale('log')
    ax.legend()

    ax = plt.subplot(grid[2,1])
    ax.step(np.linspace(0,len(kappa_diff),len(kappa_diff)), [abs(x) for x in kappa_diff], label='$\\|\\kappa_i-\\kappa_{i-1}\\|$');
    ax.step(np.linspace(0,len(sol_diff_norm),len(sol_diff_norm)), [abs(x) for x in sol_diff_norm], label='$\\|x_{i}-x_{i-1}\\|$');
    ax.set_yscale('log')
    ax.legend()

    ax = plt.subplot(grid[2,0])
    ax.step(np.linspace(0,len(grad_norm),len(grad_norm)), grad_norm, label='$\\|g_i(p_i)\\|$');
    ax.step(np.linspace(0,len(kappa_grad),len(kappa_grad)), [abs(x) for x in kappa_grad], label='$\\|g_i(\\kappa_i)\\|$');
    ax.set_yscale('log')
    ax.legend()

    l2solves = [ len(SOL_SOLVE_IT[i]) for i in range(3) ]
    print("Iterations per L2 solve:", l2solves)
    xxx = 0
    for xc in l2solves:
        xxx += xc
        for i in range(3):
            ax = plt.subplot(grid[i,0])
            ax.axvline(x=xxx, color='k', linestyle='--')
            ax = plt.subplot(grid[i,1])
            ax.axvline(x=xxx-1, color='k', linestyle='--')



    f = plt.figure(figsize=(8,8))
    grid = plt.GridSpec(2, 1, wspace=0.2, hspace=0.4, height_ratios=[2,2])


    ax = plt.subplot(grid[0,0])
    ax.plot(np.linspace(0,np_,np_), G[0])
    for ss, c in zip(SUPP[0], COMP[0]):
        ax.scatter(ss,G[0][ss], s=30, color=COLOR[c])
    for ss, c in zip(INIT_SUPP, INIT_COMP):
        ax.scatter(ss,G[0][ss], s=40, marker='s', color=COLOR[c])

    ax = plt.subplot(grid[1,0])
    ax.plot(np.linspace(0,np_,np_), G[1])
    for ss, c in zip(SUPP[1], COMP[1]):
        ax.scatter(ss,G[1][ss], s=30, color=COLOR[c])
    for ss, c in zip(INIT_SUPP, INIT_COMP):
        ax.scatter(ss,G[1][ss], s=40, marker='s', color=COLOR[c])

    plt.show()
