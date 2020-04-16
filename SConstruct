import os
import subprocess
import sys

##################################################################### FUNCTIONS
def uniqueCheckLib(conf, lib):
    """ Checks for a library and appends it to env if not already appended. """
    if conf.CheckLib(lib, autoadd=0, language="C++"):
        conf.env.AppendUnique(LIBS = [lib])
        return True
    else:
        print ("ERROR: Library '" + lib + "' not found!")
        Exit(1)

def errorMissingHeader(header, usage):
    print ("ERROR: Header '" + header + "' (needed for " + usage + ") not found or does not compile!")
    Exit(1)

def print_options(vars):
    """ Print all build option and if they have been modified from their default value. """
    for opt in vars.options:
        try:
            is_default = vars.args[opt.key] == opt.default
        except KeyError:
            is_default = True
        vprint(opt.key, env[opt.key], is_default, opt.help)

def vprint(name, value, default=True, description = None):
    """ Pretty prints an environment variabe with value and modified or not. """
    mod = "(default)" if default else "(modified)"
    desc = "   " + description if description else ""
    print ("{0:10} {1:25} = {2!s:8}{3}".format(mod, name, value, desc))

def checkset_var(varname, default):
    """ Checks if environment variable is set, use default otherwise and print the value. """
    var = os.getenv(varname)
    if not var:
        var = default
        vprint(varname, var)
    else:
        vprint(varname, var, False)
    return var

def get_real_compiler(compiler):
    """ Gets the compiler behind the MPI compiler wrapper. """
    if compiler.startswith("mpi"):
        try:
            output = subprocess.check_output("%s -show" % compiler, shell=True)
        except (OSError, subprocess.CalledProcessError) as e:
            print ("Error getting wrapped compiler from MPI compiler")
            print ("Command was:", e.cmd, "Output was:", e.output)
        else:
            return output.split()[0]
    else:
        return compiler



########################################################################## MAIN

vars = Variables(None, ARGUMENTS)

vars.Add(PathVariable("builddir", "Directory holding build files.", "build", PathVariable.PathAccept))
vars.Add(EnumVariable('build', 'Build type, either release or debug', "debug", allowed_values=('release', 'debug')))
vars.Add("compiler", "Compiler to use.", "mpicxx")
vars.Add("platform", "Specify platform.", "local")
vars.Add(BoolVariable("niftiio", "enable/disable nifti.", False))
vars.Add(BoolVariable("gpu", "Enables build for GPU support.", False))
vars.Add(BoolVariable("multi_gpu", "Enables build for multi-GPU support.", False))
vars.Add(BoolVariable("single_precision", "Enables single precision computation.", False))

env = Environment(variables = vars, ENV = os.environ)   # For configuring build variables
conf = Configure(env) # For checking libraries, headers, ...

Help(vars.GenerateHelpText(env))
env.Append(CPPPATH = ['#include'])
env.Append(CPPPATH = [os.path.join( "3rdparty")])
env.Append(CPPPATH = [os.path.join('3rdparty', 'timings')])
env.Append(CPPPATH = [os.path.join('include', 'test')])
env.Append(CPPPATH = [os.path.join('include', 'opt')])
print
print_options(vars)

if env["gpu"] == True:
    env.Tool('nvcc', toolpath = [os.getcwd()])

buildpath = os.path.join(env["builddir"], "") # Ensures to have a trailing slash

print
env.Append(LIBPATH = [('#' + buildpath)])
env.Append(CCFLAGS= ['-std=c++11'])
#env.Append(CCFLAGS= ['-Wall', '-std=c++11'])

WARN='-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused'
#env.Append(CCFLAGS= [WARN])

#if not conf.CheckHeader('mpi.h'):
#  errorMissingHeader('mpi.h', 'MPI')

# ====== Compiler Settings ======

# Produce position independent code for dynamic linking
env.Append(CCFLAGS = ['-fPIC'])
#env.Append(CCFLAGS = ['-fPIE'])
#env.Append(LINKFLAGS = ["-fpie"])
#env.Append(LINKFLAGS = ["-fpie", "-Bstatic"])

real_compiler = get_real_compiler(env["compiler"])
if real_compiler == 'icc':
    env.AppendUnique(LIBPATH = ['/usr/lib/'])
    env.Append(LIBS = ['stdc++'])
    if env["build"] == 'debug':
        env.Append(CCFLAGS = ['-align'])
    elif env["build"] == 'release':
        env.Append(CCFLAGS = ['-w', '-fast', '-align', '-ansi-alias'])
elif real_compiler == 'g++':
    pass
elif real_compiler == "clang++":
    env.Append(CCFLAGS= ['-Wsign-compare']) # sign-compare not enabled in Wall with clang.
elif real_compiler == "g++-mp-4.9":
    # Some special treatment that seems to be necessary for Mac OS.
    # See https://github.com/precice/precice/issues/2
    env.Append(LIBS = ['libstdc++.6'])
    env.AppendUnique(LIBPATH = ['/opt/local/lib/'])

#env.Append(LINKFLAGS = ["-lf2clapack", "-lf2cblas"])
#env.Append(LINKFLAGS = ['-O3', '-std=c++11', '-fopenmp',  '-DREG_HAS_PNETCDF',  '-DGAUSS_NEWTON', '-march=native'])

# set compiler variables
env.Replace(CXX = env["compiler"])
env.Replace(CC = env["compiler"])

# ====== Build Directories ======
if env["build"] == 'debug':
    # The Assert define does not actually switches asserts on/off, these are controlled by NDEBUG.
    # It's kept in place for some legacy code.
    env.Append(CPPDEFINES = ['Debug', 'Asserts'])
    env.Append(CCFLAGS = ['-g3', '-O0'])
    env.Append(LINKFLAGS = ["-rdynamic"]) # Gives more informative backtraces
    buildpath += "debug"
elif env["build"] == 'release':
    env.Append(CPPDEFINES = ['NDEBUG']) # Standard C++ macro which disables all asserts, also used by Eigen
    env.Append(CCFLAGS = ['-O3'])
    buildpath += "release"

# openMP
env.Append(CCFLAGS = ['-fopenmp'])
env.Append(LINKFLAGS = ["-fopenmp"])


# ====== preprocessor defines, #ifdefs ========
env.Append(CCFLAGS = ['-DPVFMM_MEMDEBUG'])

if env["gpu"] == True:
    env.Append(CCFLAGS = ['-DCUDA'])

if env["single_precision"] == True:
    env.Append(CCFLAGS = ['-DSINGLE'])

if env["multi_gpu"] == True:
    env.Append(CCFLAGS = ['-DMPICUDA'])

if env["niftiio"] == True:
    env.Append(CCFLAGS = ['-DNIFTIIO'])

# avx
if env["platform"] != "frontera":
    env.Append(CCFLAGS = ['-march=native'])
elif env["platform"] == "frontera" and env["gpu"] == False:
    env.Append(CCFLAGS = ['-march=native'])


# ====== CUDA =======
if env["gpu"] == True:
    CUDA_DIR = checkset_var("CUDA_DIR", "")
    env.Append(CPPPATH = [os.path.join( CUDA_DIR, "include")])
    env.Append(LIBPATH = [os.path.join( CUDA_DIR, "lib64")])
    uniqueCheckLib(conf, "cusparse")
    uniqueCheckLib(conf, "cufft")
    uniqueCheckLib(conf, "cublas")
    uniqueCheckLib(conf, "cudart")
# MPI
MPI_DIR = checkset_var("MPI_DIR", "")
env.Append(CPPPATH = [os.path.join( MPI_DIR, "include")])
env.Append(LIBPATH = [os.path.join( MPI_DIR, "lib")])

# ====== ACCFFT =======
ACCFFT_DIR = checkset_var("ACCFFT_DIR", "")
env.Append(CPPPATH = [os.path.join( ACCFFT_DIR, "include")])
env.Append(LIBPATH = [os.path.join( ACCFFT_DIR, "lib")])
uniqueCheckLib(conf, "accfft")
uniqueCheckLib(conf, "accfft_utils")
if env["gpu"] == True:
    uniqueCheckLib(conf, "accfft_gpu")
    uniqueCheckLib(conf, "accfft_utils_gpu")

# ====== PNETCDF ======
PNETCDF_DIR = checkset_var("PNETCDF_DIR", "")
env.Append(CPPPATH = [os.path.join( PNETCDF_DIR, "include")])
env.Append(LIBPATH = [os.path.join( PNETCDF_DIR, "lib")])
uniqueCheckLib(conf, "pnetcdf")

# ====== NIFTI ======
if env["niftiio"] == True:
    NIFTI_DIR = checkset_var("NIFTI_DIR", "")
    env.Append(CPPPATH = [os.path.join( NIFTI_DIR, "include/nifti")])
    env.Append(LIBPATH = [os.path.join( NIFTI_DIR, "lib")])
    uniqueCheckLib(conf, "niftiio")
    uniqueCheckLib(conf, "nifticdf")
    uniqueCheckLib(conf, "znz")

    # ====== ZLIB ======
    ZLIB_DIR = checkset_var("ZLIB_DIR", "")
    env.Append(CPPPATH = [os.path.join( ZLIB_DIR, "include")])
    env.Append(LIBPATH = [os.path.join( ZLIB_DIR, "lib")])
    uniqueCheckLib(conf, "libz")

# ====== FFTW =========
FFTW_DIR = checkset_var("FFTW_DIR", "")
env.Append(CPPPATH = [os.path.join( FFTW_DIR, "include")])
env.Append(LIBPATH = [os.path.join( FFTW_DIR, "lib")])
uniqueCheckLib(conf, "fftw3_threads")
uniqueCheckLib(conf, "fftw3")
uniqueCheckLib(conf, "fftw3f_threads")
uniqueCheckLib(conf, "fftw3f")
uniqueCheckLib(conf, "fftw3_omp")
uniqueCheckLib(conf, "fftw3f_omp")

# ====== PETSc ========
PETSC_DIR = checkset_var("PETSC_DIR", "")
PETSC_ARCH = checkset_var("PETSC_ARCH", "")
env.Append(CPPPATH = [os.path.join( PETSC_DIR, "include"),
                      os.path.join( PETSC_DIR, PETSC_ARCH, "include")])
env.Append(LIBPATH = [os.path.join( PETSC_DIR, "lib"),
                      os.path.join( PETSC_DIR, PETSC_ARCH, "lib")])

if env["platform"] == "hazelhen":
  # do nothing
  pass
elif env["platform"] == "lonestar" or env["platform"] == "stampede2" or env["platform"] == "maverick2" or env["platform"] == "frontera":
  uniqueCheckLib(conf, "petsc")
else:
  uniqueCheckLib(conf, "petsc")
  uniqueCheckLib(conf, "f2clapack")
  uniqueCheckLib(conf, "f2cblas")

# ====== 3rdParty ======
env.Append(CPPPATH = [os.path.join( "3rdparty")])
env.Append(CPPPATH = [os.path.join( "3rdparty", "pvfmm", "include")])

print
env = conf.Finish() # Used to check libraries


#--------------------------------------------- Define sources and build targets

(sourcesPGLISTR, sourcesPGLISTRGPU, sourcesSIBIAapp) = SConscript (
    'SConscript',
    variant_dir = buildpath,
    duplicate = 0
)


if env["gpu"] == True:
    bininv = env.Program (
    target = buildpath + '/tusolver',
    source = [sourcesPGLISTRGPU, './app/tusolver.cpp']
    )
    bininv = env.Program (
        target = buildpath + '/test',
        source = [sourcesPGLISTR, './app/test.cpp']
    ) 
    env.Alias("bin", bininv)
    staticlib = env.StaticLibrary (
        target = buildpath + '/pglistr',
        source = [sourcesPGLISTRGPU],
    )
    env.Alias("staticlib", staticlib)
else:
    bininv = env.Program (
        target = buildpath + '/tusolver',
        source = [sourcesPGLISTR, './app/tusolver.cpp']
    ) 
    bininv = env.Program (
        target = buildpath + '/test',
        source = [sourcesPGLISTR, './app/test.cpp']
    ) 
    env.Alias("bin", bininv)
    # solib = env.SharedLibrary (
    # target = buildpath + '/pglistr',
    #     source = [sourcesPGLISTR],
    # )
    # env.Alias("solib", solib)
    staticlib = env.StaticLibrary (
        target = buildpath + '/pglistr',
        source = [sourcesPGLISTR],
    )
    env.Alias("staticlib", staticlib)


# Creates a symlink that always points to the latest build
symlink = env.Command(
    target = "Symlink",
    source = None,
    action = "ln -fns {0} {1}".format(os.path.split(buildpath)[-1], os.path.join(os.path.split(buildpath)[0], "last"))
)

# Default(staticlib, bin, symlink)
#Default(bininv, solib, symlink)
Default(bininv, symlink)
# Default(bininv, symlink)
AlwaysBuild(symlink)

print ("Targets:   " + ", ".join([str(i) for i in BUILD_TARGETS]))
print ("Buildpath: " + buildpath)
print
