#ifndef _PVFMM_COMMON_HPP_
#define _PVFMM_COMMON_HPP_

//Define NULL
#ifndef NULL
#define NULL 0
#endif

//Disable assert checks.
//#ifndef NDEBUG
//#define NDEBUG
//#endif

//Enable memory checks.
#ifndef NDEBUG
#define PVFMM_MEMDEBUG
#endif

#include <stdint.h>
typedef     long  Integer;  // bounded numbers < 32k
typedef  int64_t     Long;  // problem size

#define MEM_ALIGN 64
#define GLOBAL_MEM_BUFF 0LL //in MB

#include <iostream>
#include <cstdio>

#define PVFMM_WARN(msg) \
do { \
  std::cerr<<"\n\033[1;31mWarning:\033[0m "<<msg<<'\n'; \
}while(0)

#define PVFMM_ERROR(msg) \
do { \
  std::cerr<<"\n\033[1;31mError:\033[0m "<<msg<<'\n'; \
  abort(); \
}while(0)

#define PVFMM_ASSERT_MSG(cond, msg) \
do { \
  if (!(cond)) PVFMM_ERROR(msg); \
}while(0)

#define PVFMM_ASSERT(cond) \
do { \
  if (!(cond)) { \
    fprintf (stderr, "\n%s:%d: %s: Assertion `%s' failed.\n",__FILE__,__LINE__,__PRETTY_FUNCTION__,#cond); \
    abort(); \
  } \
}while(0)


//#include <pvfmm/stacktrace.h>
//const int sgh=pvfmm::SetSigHandler();

#endif //_PVFMM_COMMON_HPP_
