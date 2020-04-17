import numpy as np
import os, sys
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='plot convergence')
parser.add_argument ('-x',           type = str,          help = 'path to the results folder');
args = parser.parse_args();




# noise 
if True:
    spec = args.x.split("fd")
    d1 = spec[0] + 'fd' + spec[1]
    d2 = spec[0] + 'adj' + spec[1]
    d3 = spec[0] + 'fd' + spec[1] + '-scale'
    d4 = spec[0] + 'adj' + spec[1] + '-scale'
    spec = args.x.split("inv")
    d5 = 'inv-' + 'adv' + spec[1]
    DIR = [d1,d2,d3,d4,d5]

# noise 
if False:
    spec = args.x.split("nonoise")
    d1 = spec[0] + 'nonoise' + spec[1]
    d2 = spec[0] + 'noise-lres' + spec[1]
    d3 = spec[0] + 'noise-sp10' + spec[1]
    #d4 = spec[0] + 'noise-sp20' + spec[1]
    DIR = [d1,d2,d3]

# filter dirs
DIRS = []
for dir in DIR:
    if (not os.path.isdir(dir)) or (not os.path.exists(os.path.join(dir, 'tumor_solver_log.txt'))):
       continue;
    DIRS.append(dir)
 

print("DIRS: {}".format(DIRS))

fig = plt.figure(1, figsize=(8,2*len(DIRS)))

sp = 1
rows = len(DIRS)
TERM = False;
for dir in DIRS:

    obj_nit = {}
    rho_nit = {}
    kap_nit = {}
    obj_ls  = {}
    rho_ls  = {}
    kap_ls  = {}
    read=False
    x = 0
    print(" .. processing dir {}".format(dir))
    with open(os.path.join(dir, 'tumor_solver_log.txt')) as f:
        LINES = f.readlines()
        for k in range(len(LINES)):
            if "Error" in LINES[k] or "ERROR" in LINES[k]:
                print(" .. error detected. skipping.")
                TERM = True;
                break;
        if TERM:
            rows -= 1;
            continue;
        for k in range(len(LINES)):
            line = LINES[k];
            if "objective (abs)" in line: 
                read=True
                continue
            if read:
                if ("[1;31m[" in line) and (not "converged" in line) and (not "maximum number" in line):
                    obj_nit[x] = float(line.split()[2])
                    #obj_ls[] = float(line.split()[2])
                    rho_nit[x] = float(line.split()[6])
                    #rho_ls[] = float(line.split()[6])
                    kap_nit[x] = float(line.split()[7].split(']')[0])
                    #kap_ls[] = float(line.split()[7].split(']')[0])
                    l = 5 if "linesearch: successful" in LINES[k+1] else 0
                    x += 1
                    if "converged" in LINES[k+1+l]:
                        break;
                    kap_ls[x] = float(LINES[k+1+l].split("Diffusivity guess = (")[-1].split(",")[0])
                    rho_ls[x] = float(LINES[k+2+l].split("Reaction  guess   = (")[-1].split(",")[0])
                    obj_ls[x] = float(LINES[k+4+l].split("J(p) = Dc(c1) + Dc(c0) + S(c0) =")[-1].split("=")[0])
                    #kap_nit.append(float('nan'))
                    #rho_nit.append(float('nan'))
                    #obj_nit.append(float('nan'))
    

    ax = fig.add_subplot(rows,1,sp)

    lists = sorted(rho_ls.items())
    x, y = zip(*lists)
    ax.scatter(x, y, label="rho_ls", marker='o', c='r')
    #ax.scatter(np.arange(0,len(rho_ls)), rho_ls, label="rho_ls", marker='o', c='r')
    #ax.scatter(np.arange(0,len(rho_nit)), rho_ls, label="rho_nit", c='r')
    lists = sorted(rho_nit.items())
    x, y = zip(*lists)
    ax.plot(x, y, label="rho_nit", c='r')
    #ax.plot(np.arange(0,len(obj_ls), 2), rho_nit, label="rho_nit", c='r')
    ax.set_ylim([3,15])
    ax.axhline(y=12, color='gray', linestyle='--')
    plt.legend()
    ax = ax.twinx()
    lists = sorted(obj_ls.items())
    x, y = zip(*lists)
    ax.scatter(x, y, label="obj_ls", marker='o', c='b')
    #ax.scatter(np.arange(0,len(obj_ls)), obj_ls, label="obj_ls", marker='o', c='b')
    #ax.scatter(np.arange(0,len(obj_nit)), obj_nit, label="obj_nit", c='b')
    lists = sorted(obj_nit.items())
    x, y = zip(*lists)
    ax.plot(x, y, label="obj_nit", c='b')
    #ax.plot(np.arange(0,len(obj_ls), 2), obj_nit, label="obj_nit", c='b')
    lists = sorted(kap_ls.items())
    x, y = zip(*lists)
    ax.scatter(x, y, label="kap_ls", marker='o', c='g')
    #ax.scatter(np.arange(0,len(kap_ls)), kap_ls, label="kap_ls", marker='o', c='g')
    #ax.scatter(np.arange(0,len(kap_nit)), kap_nit, label="kap_nit",  c='g')
    lists = sorted(kap_nit.items())
    x, y = zip(*lists)
    ax.plot(x, y, label="kap_nit", c='g')
    #ax.plot(np.arange(0,len(obj_ls), 2), kap_nit, label="kap_nit", c='g')
    if not "scale" in dir:
        ax.axhline(y=0.1, color='gray', linestyle='-.')
        ax.set_ylim([1E-3,2])
    else:
        ax.axhline(y=10, color='gray', linestyle='-.')
        ax.set_ylim([1E-2,20])
    ax.set_yscale('log')
    plt.legend()
    plt.title(dir)
    sp += 1

#plt.tight_layout()
plt.show()

#print("\nobjective: \n{}".format(obj_nit))
#print("\nrho: \n{}".format(rho_nit))
#print("\nkappa: \n{}".format(kap_nit))
