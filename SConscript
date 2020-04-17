# This is file Sconscript

sourcesPGLISTR = [
    Glob('./src/*.cpp'),
    Glob('./src/test/*.cpp'),
    Glob('./src/opt/*.cpp'),
    Glob('./src/utils/*.cpp'),
    Glob('./src/pde/*.cpp'),
    Glob('./src/mat/*.cpp'),
    Glob('./src/grad/*.cpp'),
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
