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

TF = {
       '022_S_6013' :  {'t1' : 0.40,  't2' : 0.1, 't02' : 0.2},
       '023_S_1190' :  {'t1' : 0.47,  't2' : 0.1, 't02' : 0.2},
       '127_S_4301' :  {'t1' : 0.47,  't2' : 0.63, 't02' : 1.63},
       '127_S_2234' :  {'t1' : 0.66,  't2' : 0.63, 't02' : 1.63},
       '012_S_6073' :  {'t1' : 0.35,  't2' : 0.63, 't02' : 1.63},
       '033_S_4179' :  {'t1' : 0.37,  't2' : 0.63, 't02' : 1.63},
       '941_S_4036' :  {'t1' : 0.46,  't2' : 0.63, 't02' : 1.63},
       '032_S_5289' :  {'t1' : 0.74,  't2' : 0.63, 't02' : 1.63},
       '035_S_4114' :  {'t1' : 0.42,  't2' : 0.63, 't02' : 1.63},
    }


DIRS = os.listdir(args.x)
TAB = []
TAB2 = []

th=0.2
tol=''

for case_dir in DIRS:
    if not "CASE_" in case_dir:
        continue;
    if "_CASE" in case_dir:
        continue;

    for dir in os.listdir(os.path.join(args.x, case_dir)):
        tol = '1E-4'
        if not "inv-" in dir: 
            continue;
        if not "th-"+str(th) in dir:
            continue;
        if 'tol-' in dir:
            tol = str(dir.split('tol-')[-1])
        elif 'tol' in dir:
            tol = str(dir.split('tol')[-1])
        if 'sol-' in dir:
            tol = str(dir.split('sol-')[-1])
        elif 'sol' in dir:
            tol = str(dir.split('sol')[-1])
        print(tol)
        if not  tol == '0.01':
            print(" ...skipping")
            continue;


        n_it = 0
        rho_inv = 0
        k_inv = 0
        eps_k = 0
        eps_rho = 0
        miss = 0
        time = 0
        descr = ''
     
        print("[] processing {}/{}".format(case_dir,dir))
        
        if 'adv' in dir:
            descr = 'adv; ' + descr
        descr += '; ' + tol + '; '
        if 'rho_lb' in dir:
            descr = 'rlb-' + str(dir.split('rho-lb-')[-1].split('-fd')[0]) + '; ' + descr
        init_r = float(dir.split('iguess[r-')[-1].split('-')[0])
        init_k = float(dir.split(']-')[0].split('k-')[-1])
        success=False
        dir = os.path.join(case_dir, dir)
        case = case_dir.split('CASE_')[-1]
        descr = case + descr
        try:
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
                    if "rel. l2-error at observation points:" in line:
                        miss_obs = float(line.split("rel. l2-error at observation points:")[-1].split()[0])
                    if "l2-error in reconstruction:" in line:
                        miss = float(line.split("l2-error in reconstruction:")[-1].split()[0])
                    if "rel. l2-error (everywhere) (T=1.0) :" in line:
                        mu_1 = float(line.split('rel. l2-error (everywhere) (T=1.0) :')[-1].split()[0])
                    if "rel. l2-error at observation points (T=1.0) :" in line:
                        mu_1_obs = float(line.split('rel. l2-error at observation points (T=1.0) :')[-1].split()[0])
                    if "rel. l2-error at observation points (T=1.2) :" in line:
                        mu_12 = float(line.split('rel. l2-error at observation points (T=1.2) :')[-1].split()[0])
                    if "rel. l2-error at observation points (T=1.5) :" in line:
                        mu_15 = float(line.split('rel. l2-error at observation points (T=1.5) :')[-1].split()[0])
        except:
            print("Error reading log file");
            continue;
        try:
            # compute prediction error
            d_path    = os.path.join(os.path.join(args.x, case_dir), 'data');
            o_path    = os.path.join(args.x, dir);
            tc_path    = os.path.join(os.path.join(args.x, case_dir), 'tc');
            d1_true   = readNetCDF(os.path.join(d_path, 'time_point_1_tau.nc'))

            ref  = nib.load(os.path.join(d_path, 'time_point_0_tau.nii.gz'));
            header = ref.header;
            affine = ref.affine;
            sz = ref.header['dim'][1:4]
      
            c1_rec    = readNetCDF(os.path.join(o_path,'cPrediction0_[t='+str(TF[case]['t1'])+'].nc'))
            c0        = readNetCDF(os.path.join(o_path,'c0Recon.nc'))
            d1s       = readNetCDF(os.path.join(o_path,'d1_s.nc'))
            d0s       = readNetCDF(os.path.join(o_path,'d0_s.nc'))
            d0s_obs   = readNetCDF(os.path.join(o_path,'d0_s_obs.nc'))
            res1      = readNetCDF(os.path.join(o_path,'res1.nc'))
            res1_obs  = readNetCDF(os.path.join(o_path,'res1_obs.nc'))
         
            np.clip(c1_rec,  0, 1, out=c1_rec)
            np.clip(d1_true, 0, 1, out=d1_true)

            d1_true_obs = np.where(d1_true > th, d1_true, 0);
    
            miss_t1     = np.linalg.norm(c1_rec - d1_true) / np.linalg.norm(d1_true); 
            miss_t1_obs = np.linalg.norm(np.where(d1_true > th, c1_rec, 0) - d1_true_obs) / np.linalg.norm(d1_true_obs); 
    
            out_p = os.path.join(args.x, 'res_summary_th-'+str(th))
            if not os.path.exists(out_p):
                os.makedirs(out_p)
            out_p_curr = os.path.join(out_p, dir)
            if not os.path.exists(out_p_curr):
                os.makedirs(out_p_curr)
   
    
            writeNII( resizeImage(np.swapaxes(np.abs(res1),  0,2), sz, 1),  os.path.join(out_p_curr, 'res_1.nii.gz'), ref_image=ref);
            writeNII( resizeImage(np.swapaxes(np.abs(res1_obs),  0,2), sz, 1),  os.path.join(out_p_curr, 'res_obs_1.nii.gz'), ref_image=ref);
            #writeNII( resizeImage(np.swapaxes(np.abs(c1_rec  - d1_true),  0,2), sz, 1),  os.path.join(out_p_curr, 'res_1.nii.gz'), ref_image=ref);
            #writeNII( resizeImage(np.swapaxes(np.abs(np.where(d1_true > th, c1_rec, 0)  - d1_true_obs),  0,2), sz, 1),  os.path.join(out_p_curr, 'res_obs_1.nii.gz'), ref_image=ref);
            c1_rec   = resizeImage(np.swapaxes(c1_rec,    0, 2), sz, 1) 
            c0       = resizeImage(np.swapaxes(c0,        0, 2), sz, 1) 
            d1s      = resizeImage(np.swapaxes(d1s,       0, 2), sz, 1) 
            d0s      = resizeImage(np.swapaxes(d0s,       0, 2), sz, 1) 
            d0s_obs  = resizeImage(np.swapaxes(d0s_obs,   0, 2), sz, 1) 
            writeNII(c1_rec,  os.path.join(out_p_curr, 'c1_rec.nii.gz'), ref_image=ref);
            d0_true  = resizeImage(np.swapaxes(readNetCDF(os.path.join(d_path, 'time_point_0_tau.nc')), 0, 2), sz, 1) 
            d1_true  = resizeImage(np.swapaxes(d1_true, 0, 2), sz, 1) 
            d1_true_obs  = resizeImage(np.swapaxes(d1_true_obs, 0, 2), sz, 1) 
            #d12_true = resizeImage(np.swapaxes(d12_true, 0, 2), sz, 1) 
            #d15_true = resizeImage(np.swapaxes(d15_true, 0, 2), sz, 1) 
            writeNII(d1_true,     os.path.join(out_p_curr, 'd1.nii.gz'), ref_image=ref);
            writeNII(d1s,         os.path.join(out_p_curr, 'd1_smooth.nii.gz'), ref_image=ref);
            writeNII(d1_true_obs, os.path.join(out_p_curr, 'd1_obs.nii.gz'), ref_image=ref);
            writeNII(d0_true,     os.path.join(out_p_curr, 'd0.nii.gz'), ref_image=ref);
            writeNII(c0,          os.path.join(out_p_curr, 'c0.nii.gz'), ref_image=ref);
            writeNII(d0s_obs,     os.path.join(out_p_curr, 'd0_smooth_obs.nii.gz'), ref_image=ref);
    
        except Exception as e:
            print("Error during loading images: ", e)
            success = False;
        if success:
           TAB.append("\\textit{{{0:22s}}}  & \\num{{{1:4.2f}}} &  \\num{{{2:e}}} & \\num{{{3:e}}}  & \\num{{{4:e}}} ".format(descr,  rho_inv, k_inv, mu_1, mu_1_obs))
        else: 
            print("{} ERROR".format(dir))
    

TAB.sort()
TAB2.sort()
for t in TAB:
    print(t, "  \\\ ");
print()
#print()
#for t in TAB2:
#    print(t, "  \\\ ");
