import argparse
import nibabel as nib
import os, sys
import ntpath
import numpy as np
import netCDF4
from netCDF4 import Dataset
import scipy.ndimage as ndimage

### ------------------------------------------------------------------------ ###
def createNetCDF(filename,dimensions,variable):
    '''
    function to write a netcdf image file and return its contents
    '''
    imgfile = Dataset(filename,mode='w',format="NETCDF3_CLASSIC");
    x = imgfile.createDimension("x",dimensions[0]);
    y = imgfile.createDimension("y",dimensions[1]);
    z = imgfile.createDimension("z",dimensions[2]);
    data = imgfile.createVariable("data","f8",("x","y","z",));
    data[:,:,:] = variable[:,:,:];
    imgfile.close();

###
### ------------------------------------------------------------------------ ###
def readNetCDF(filename):
    '''
    function to read a netcdf image file and return its contents
    '''
    imgfile = Dataset(filename);
    img = imgfile.variables['data'][:]
    imgfile.close();
    return img

###
### ------------------------------------------------------------------------ ###
def writeNII(img, filename, affine=None, ref_image=None):
    '''
    function to write a nifti image, creates a new nifti object
    '''
    if ref_image is not None:
        data = nib.Nifti1Image(img, affine=ref_image.affine, header=ref_image.header);
        data.header['datatype'] = 64
        data.header['glmax'] = np.max(img)
        data.header['glmin'] = np.min(img)
    elif affine is not None:
        data = nib.Nifti1Image(img, affine=affine);
    else:
        data = nib.Nifti1Image(img, np.eye(4))

    nib.save(data, filename);



###
### ------------------------------------------------------------------------ ###
def resizeImage(img, new_size, interp_order):
    '''
    resize image to new_size
    '''
    factor = tuple([float(x)/float(y) for x,y in zip(list(new_size), list(np.shape(img)))]);
    #print(factor);
    return ndimage.zoom(img, factor, order=interp_order);


###
### ------------------------------------------------------------------------ ###
def resizeNIIImage(img, new_size, interp_order=0):
    '''
    uses nifti img object to resize it to new_size
    '''
    # load the image
    old_voxel_size = img.header['pixdim'][1:4];
    old_size = img.header['dim'][1:4];
    new_size = np.asarray(new_size);
    print("new size: {}, old size: {}".format(new_size, old_size))
    new_voxel_size = np.multiply(old_voxel_size, np.divide(old_size, new_size));
    return nib.processing.resample_to_output(img, new_voxel_size, order=interp_order, mode='wrap');



parser = argparse.ArgumentParser(description='read objective')
parser.add_argument ('-x',           type = str,          help = 'path to the results folder');
args = parser.parse_args();


NOISE = {
0  : {'0.1' : {'d1' : 8,  'd0' : 45},
      '0.5' : {'d1' : 14, 'd0' : 45},
      '1'   : {'d1' : 24, 'd0' : 45},
      '1.5' : {'d1' : 35, 'd0' : 45},
     },
20 : {'0.1' : {'d1' : 8,  'd0' : 8},
      '0.5' : {'d1' : 12, 'd0' : 12},
      '1'   : {'d1' : 19, 'd0' : 19},
      '1.5' : {'d1' : 27, 'd0' : 27},
     },
40 : {'0.1' : {'d1' : 8,  'd0' : 8},
      '0.5' : {'d1' : 12, 'd0' : 12},
      '1'   : {'d1' : 20, 'd0' : 20},
      '1.5' : {'d1' : 29, 'd0' : 29},
     },
60 : {'0.1' : {'d1' : 8,  'd0' : 8},
      '0.5' : {'d1' : 13, 'd0' : 13},
      '1'   : {'d1' : 21, 'd0' : 21},
      '1.5' : {'d1' : 30, 'd0' : 31},
     },
80 : {'0.1' : {'d1' : 8,  'd0' : 8},
      '0.5' : {'d1' : 13, 'd0' : 13},
      '1'   : {'d1' : 23, 'd0' : 22},
      '1.5' : {'d1' : 32, 'd0' : 32},
     }
}

rho_star = 8.
k_star   = 0.18

DIRS = os.listdir(args.x)
TAB = []

for dir in DIRS:
    if not "inv-adv-" in dir:
        continue;
    if not "scale" in dir:
        continue;
    if "lres" in dir: 
        continue
    tpoint = int(dir.split('d_nc-')[-1].split('-corr'))
    nlvl   = dir.split('-sp')[-1].split('-iguess')[0]

    n_it = 0
    rho_inv = 0
    k_inv = 0
    eps_k = 0
    eps_rho = 0
    miss = 0
    time = 0
    descr = ''
    d1 = ''
    d0 = ''
    if 'nonoise' in dir or 'noise-no' in dir:
        d1 = '0\\%'
        d0 = '0\\%'
    d1 = str(NOISE[tpoint][nlvl]['d1']) + '\\%'
    d0 = str(NOISE[tpoint][nlvl]['d0']) + '\\%'
     
    if 'adv' in dir:
        descr = 'adv; ' + descr
    if 'corr' in dir:
        descr = 'cor; ' + descr
    meth = dir.split(']-')[-1].split('-')[0] 
    if 'scale2' in dir:
        meth = meth + '-sc-1e2';
    elif 'scale' in dir:
        meth = meth + '-sc-1e1';
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

            eps_k = np.abs(k_inv - k_star) / k_star
            eps_rho = np.abs(rho_inv - rho_star) / rho_star

    # compute prediction error
    d_path    = os.path.join(os.path.join(args.x, 'tc'), 'd_nc-'+str(tpoint));
    d1_true   = readNetCDF(os.path.join(d_path, 'dataBeforeObservation.nc'))
    d_path    = os.path.join(os.path.join(args.x, 'tc'), 't=1.2');
    d12_true  = readNetCDF(os.path.join(d_path, 'dataBeforeObservation.nc'))
    d_path    = os.path.join(os.path.join(args.x, 'tc'), 't=1.5');
    d15_true  = readNetCDF(os.path.join(d_path, 'dataBeforeObservation.nc'))
    c1_rec    = readNetCDF(os.path.join(os.path.join(args.x, dir),'cPrediction_[t=1.0].nc'))
    c12_rec   = readNetCDF(os.path.join(os.path.join(args.x, dir),'cPrediction_[t=1.2].nc'))
    c15_rec   = readNetCDF(os.path.join(os.path.join(args.x, dir),'cPrediction_[t=1.5].nc'))

    miss_t1   = np.linalg.norm(c1_rec - d1_true) / np.linalg.norm(d1_true); 
    miss_t12  = np.linalg.norm(c12_rec - d12_true) / np.linalg.norm(d12_true); 
    miss_t15  = np.linalg.norm(c15_rec - d15_true) / np.linalg.norm(d15_true); 

    out_p = os.path.join(os.path.join(args.x, 'res_summary'), 't-'+str(tpoint))
    if not os.path.exists(out_p):
        os.makedirs(out_p)
    out_p_curr = os.path.join(out_p, dir.split('noise-')[-1].split('-')[0]+str('Pc0'))
    if not os.path.exists(out_p_curr):
        os.makedirs(out_p_curr)

    ref  = nib.load('0368Y02_cbq_n3.nii.gz');
    seg  = nib.load('0368Y02_segmented.nii.gz').get_fdata();
    seg2 = nib.load('0368Y01_segmented.nii.gz').get_fdata();
    header = ref.header;
    affine = ref.affine;
    sz = ref.header['dim'][1:4]
    writeNII(seg,  os.path.join(out_p, '0368Y02_segmented.nii.gz'), affine);
    writeNII(seg2, os.path.join(out_p, '0368Y01_segmented.nii.gz'), affine);

    c1_rec  = resizeImage(np.swapaxes(c1_rec, 0, 2), sz, 1) 
    c12_rec = resizeImage(np.swapaxes(c12_rec, 0, 2), sz, 1) 
    c15_rec = resizeImage(np.swapaxes(c15_rec, 0, 2), sz, 1) 
    writeNII(c1_rec,  os.path.join(out_p_curr, 'c1_rec.nii.gz'), affine);
    writeNII(c12_rec, os.path.join(out_p_curr, 'c12_rec.nii.gz'), affine);
    writeNII(c15_rec, os.path.join(out_p_curr, 'c15_rec.nii.gz'), affine);

    d0_true  = resizeImage(np.swapaxes(readNetCDF(os.path.join('tc/d_nc-'+str(tpoint), 'c0True.nc')), 0, 2), sz, 1) 
    d1_true  = resizeImage(np.swapaxes(d1_true, 0, 2), sz, 1) 
    d12_true = resizeImage(np.swapaxes(d12_true, 0, 2), sz, 1) 
    d15_true = resizeImage(np.swapaxes(d15_true, 0, 2), sz, 1) 
    writeNII(d1_true,   os.path.join(out_p, 'd1_true.nii.gz'), affine);
    writeNII(d0_true,   os.path.join(out_p, 'd0_true.nii.gz'), affine);
    writeNII(d12_true,  os.path.join(out_p, 'd12_true.nii.gz'), affine);
    writeNII(d15_true,  os.path.join(out_p, 'd15_true.nii.gz'), affine);

    nn =''
    if "noise-no" in dir:
        writeNII(d1_true,   os.path.join(out_p_curr, 'd1_true.nii.gz'), affine);
        writeNII(d0_true,   os.path.join(out_p_curr, 'd0_true.nii.gz'), affine);
    else:
        nn = dir.split('noise-sp')[-1].split('-')[0]
        d1_noise  = resizeImage(np.swapaxes(readNetCDF(os.path.join('tc/d_nc-'+str(tpoint), 'data_t1_noise-'+str(nn)+'.nc')), 0, 2), sz, 1) 
        d0_noise  = resizeImage(np.swapaxes(readNetCDF(os.path.join('tc/d_nc-'+str(tpoint), 'data_t0_noise-'+str(nn)+'.nc')), 0, 2), sz, 1) 
        writeNII(d1_noise, os.path.join(out_p_curr, 'd1_noise.nii.gz'), affine);
        writeNII(d0_noise, os.path.join(out_p_curr, 'd0_noise.nii.gz'), affine);

    if success:
       TAB.append("\\textit{{{0:22s}}} & \\textit{{{1:4d}}}  & {2:6s} & {3:6s}  & \\textit{{{4:14s}}} & \\num{{{5:1.0f}}}  & \\num{{{6:e}}}  & {7:4.2f} & \\num{{{8:e}}}& \\num{{{9:e}}} & \\num{{{10:e}}} & \\num{{{11:e}}} & \\num{{{12:e}}} & \\num{{{13:e}}} & \\num{{{14:e}}}  & {15:2d} & \\num{{{16:e}}}  ".format(descr,str(tpoint), d1, d0, meth, init_r, init_k,  rho_inv, k_inv, eps_rho, eps_k, miss, miss_t1, miss_t12, miss_t15,  n_it, time))
       #print("\\textit{{{0:22s}}}  & \\textit{{{1:14s}}} & \\num{{{2:1.0f}}}  & \\num{{{3:e}}}  & {4:4.2f} & \\num{{{5:e}}} & \\num{{{6:e}}} & \\num{{{7:e}}} & \\num{{{8:e}}} & {9:2d} & \\num{{{10:e}}}  ".format(descr, meth, init_r, init_k,  rho_inv, k_inv, eps_rho, eps_k, miss, n_it, time))
    else: 
        print("{} ERROR".format(dir))


TAB.sort()
for t in TAB:
    print(t, "  \\\ ");
