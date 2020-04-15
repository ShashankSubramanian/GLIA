# This is file Sconscript

sourcesPGLISTR = [
    Glob('./src/*.cpp'),
    Glob('./src/test/*.cpp')
    #if '*.cpp' not in f.path
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
    Glob('src/cuda/*.cu')
]

sourcesAll = [
    sourcesAllNoMain,
    sourcesDrivers
]

Return ('sourcesAllNoMain', 'sourcesAllNoMainGPU', 'sourcesAll')
