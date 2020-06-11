from tools import *
import argparse

code_dir = os.path.dirname(os.path.realpath(__file__)) + '/../'
prefix = 'CASE_035_S_4114'
data_path = code_dir + 'real_data/' + '/data/'
results_dir = code_dir + 'inverse_RD_finali1/' 
template = nib.load(os.path.join(data_path, 'time_point_1_seg_wm.nii.gz'))

def convert(d):
    for dir in os.listdir(d):
        if not '.nc' in dir:
           continue
        print("   ... converting {}".format(dir))
        dat = readNetCDF(os.path.join(d, dir));
        dat = np.swapaxes(dat,0,2);
        output_size = tuple(dat.shape)
        filename = ntpath.basename(dir);
        filename = filename.split('.nc')[0]
        newfilename = filename + '.nii.gz';
        writeNII(dat, os.path.join(d,newfilename), template.affine);




parser = argparse.ArgumentParser(description='reading files')
parser.add_argument ('-x', type = str, help = 'path to the results folder')
args = parser.parse_args()

DIRS = os.listdir(results_dir + '/.')
TAB = []

for dir in DIRS:

    # Converting the nc files to nii 
    print(' [] converting ddir {}'.format(results_dir + dir))
    convert(results_dir + dir)

