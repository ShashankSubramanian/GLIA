# TODOS CLEANUP
---

### inverse.cpp/Solver.cpp

**todo: (implement)**
 - in _tusolver.cpp_: config file; parse config file; populate to parameters
 - in _Solver.cpp_: implement read in mass effect data; healthy patient etc; complete ME initialize
 - in _Solver.cpp_: implement read in alzh data
 - in _Solver.cpp_: implement apply low frequency noise <-- add to createSynthetic

**unclear: (dropped)**
 - in _Solver.cpp_: createSynthetic: I dropped generation of MF data; this should be implemented as test in the test suite, since hard coded coordinates; you can also define coordinates in a p.txt and phi.txt file, store the min a test/ dir, and read them in (with hard coded path);
- generate sinusoidal: do we need this? <-- move to test suite
 - inverse.cpp:840-844: no idea what this is good for, not implemented
 - inverse.cpp:877-891: not implemented; outparams not implemented
 - inverse.cpp: 893-920: wL2 norm, not implemented, dropped this section
 - generateSyntheticData: why is c(0) read in in 'generateSynthetic'? I'm dropping this; reading in is done in readData.
 - inverse.cpp:1603-1718: error for p (plus some weighted error) I have dropped this

---

 ### Parameters.h
  - populate all parameters from utils and n_misc; structure them into groups, see examples
  - change all occurrences of n_misc parameters etc.

---

### MatProb.h
- can we get rid of bg? clarify what glm is used for; I also introduced ve;

---

### Obs.h/Obs.cpp
- implement two time points from alzh solver; add time point to apply in all files

---

### IO.h/IO.cpp
- change dataOut to take params_ as argument and std::string instead of char*

---
### PdeOperators
- change forward_flag to time_history_off_

---
### Invsolver
- OptimizerSettings members have a trailing underscore now.
- changed newtonsolver to newton_solver_