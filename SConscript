# This is file Sconscript

sourcesPGLISTR = [
    f for f in Glob('./src/*.cpp')
    #if '*.cpp' not in f.path
]

sourcesTHIRDPARTY = [
    Glob('./3rdparty/timings/*.cpp')
]

sourcesDrivers = [
    Glob('app/forward.cpp'),
    Glob('app/inverse.cpp')
]

sourcesAllNoMain = [
    sourcesPGLISTR,
    sourcesTHIRDPARTY,
    Glob('src/*.cu')
]

sourcesAll = [
    sourcesAllNoMain,
    sourcesDrivers
]

Return ('sourcesAllNoMain', 'sourcesAll')
