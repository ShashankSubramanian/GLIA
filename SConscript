# This is file Sconscript

sourcesPGLISTR = [
    f for f in Glob('./src/*.cpp')
    #if 'opt/' not in f.path
]

sourcesTHIRDPARTY = [
    Glob('./3rdparty/timings/*.cpp'),
]

sourcesDrivers = [
    Glob('app/tusolver.cpp')
]

sourcesAllNoMain = [
    sourcesPGLISTR,
    sourcesTHIRDPARTY,
]

sourcesAllNoMainGPU = [
    sourcesPGLISTR,
    sourcesTHIRDPARTY,
    Glob('src/*.cu'),
    Glob('src/Cuda/*.cu')
]

sourcesAll = [
    sourcesAllNoMain,
    sourcesDrivers
]

Return ('sourcesAllNoMain', 'sourcesAllNoMainGPU', 'sourcesAll')
