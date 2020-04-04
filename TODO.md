# TODOS CLEANUP
---

### inverse.cpp/Solver.cpp

 - in _Solver.cpp_: read data for all solvers, initialize all solvers
 - in _Solver.cpp_: implement run and finalize methods (compute error)
 - in _tusolver.cpp_: config file; parse config file; populate to parameters
 - in _Solver.cpp_: read in mass effect data; healthy patient etc
 - in _Solver.cpp_: read in alzh data
 - in _Solver.cpp_: implement generate synthetic data
 - in _Solver.cpp_: compute error <-- move to finalize
 - in _Solver.cpp_: compute segmentation <-- move to finalize
 - in _Solver.cpp_: apply low frequency noise; add to createSynthetic
 - in _Solver.cpp_: implement readUserCM
 - generate sinusoidal: do we need this? <-- move to test suite
 
---

 ### Parameters.h
  - populate all parameters from utils and n_misc; structure them into groups, see examples
  - change all occurrences of n_misc parameters etc.

---

### MatProb.h
  - can we get rid of bg? clarify what glm is used for; I also introduced ve;
