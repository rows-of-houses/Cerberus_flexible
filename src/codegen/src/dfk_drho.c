/* This file was automatically generated by CasADi 3.6.5.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) dfk_drho_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[16] = {12, 1, 0, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[7] = {3, 1, 0, 3, 0, 1, 2};

/* dfk_drho:(i0[12],i1[4])->(o0[3],o1[3],o2[3],o3[3]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6;
  a0=arg[0]? arg[0][1] : 0;
  a1=cos(a0);
  a2=arg[0]? arg[0][2] : 0;
  a3=sin(a2);
  a4=(a1*a3);
  a0=sin(a0);
  a2=cos(a2);
  a5=(a0*a2);
  a4=(a4+a5);
  a4=(-a4);
  if (res[0]!=0) res[0][0]=a4;
  a4=arg[0]? arg[0][0] : 0;
  a5=sin(a4);
  a6=(a5*a0);
  a6=(a6*a3);
  a5=(a5*a1);
  a5=(a5*a2);
  a6=(a6-a5);
  a6=(-a6);
  if (res[0]!=0) res[0][1]=a6;
  a4=cos(a4);
  a1=(a4*a1);
  a1=(a1*a2);
  a4=(a4*a0);
  a4=(a4*a3);
  a1=(a1-a4);
  a1=(-a1);
  if (res[0]!=0) res[0][2]=a1;
  a1=arg[0]? arg[0][4] : 0;
  a4=cos(a1);
  a3=arg[0]? arg[0][5] : 0;
  a0=sin(a3);
  a2=(a4*a0);
  a1=sin(a1);
  a3=cos(a3);
  a6=(a1*a3);
  a2=(a2+a6);
  a2=(-a2);
  if (res[1]!=0) res[1][0]=a2;
  a2=arg[0]? arg[0][3] : 0;
  a6=sin(a2);
  a5=(a6*a1);
  a5=(a5*a0);
  a6=(a6*a4);
  a6=(a6*a3);
  a5=(a5-a6);
  a5=(-a5);
  if (res[1]!=0) res[1][1]=a5;
  a2=cos(a2);
  a4=(a2*a4);
  a4=(a4*a3);
  a2=(a2*a1);
  a2=(a2*a0);
  a4=(a4-a2);
  a4=(-a4);
  if (res[1]!=0) res[1][2]=a4;
  a4=arg[0]? arg[0][7] : 0;
  a2=cos(a4);
  a0=arg[0]? arg[0][8] : 0;
  a1=sin(a0);
  a3=(a2*a1);
  a4=sin(a4);
  a0=cos(a0);
  a5=(a4*a0);
  a3=(a3+a5);
  a3=(-a3);
  if (res[2]!=0) res[2][0]=a3;
  a3=arg[0]? arg[0][6] : 0;
  a5=sin(a3);
  a6=(a5*a4);
  a6=(a6*a1);
  a5=(a5*a2);
  a5=(a5*a0);
  a6=(a6-a5);
  a6=(-a6);
  if (res[2]!=0) res[2][1]=a6;
  a3=cos(a3);
  a2=(a3*a2);
  a2=(a2*a0);
  a3=(a3*a4);
  a3=(a3*a1);
  a2=(a2-a3);
  a2=(-a2);
  if (res[2]!=0) res[2][2]=a2;
  a2=arg[0]? arg[0][10] : 0;
  a3=cos(a2);
  a1=arg[0]? arg[0][11] : 0;
  a4=sin(a1);
  a0=(a3*a4);
  a2=sin(a2);
  a1=cos(a1);
  a6=(a2*a1);
  a0=(a0+a6);
  a0=(-a0);
  if (res[3]!=0) res[3][0]=a0;
  a0=arg[0]? arg[0][9] : 0;
  a6=sin(a0);
  a5=(a6*a2);
  a5=(a5*a4);
  a6=(a6*a3);
  a6=(a6*a1);
  a5=(a5-a6);
  a5=(-a5);
  if (res[3]!=0) res[3][1]=a5;
  a0=cos(a0);
  a3=(a0*a3);
  a3=(a3*a1);
  a0=(a0*a2);
  a0=(a0*a4);
  a3=(a3-a0);
  a3=(-a3);
  if (res[3]!=0) res[3][2]=a3;
  return 0;
}

CASADI_SYMBOL_EXPORT int dfk_drho(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int dfk_drho_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int dfk_drho_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void dfk_drho_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int dfk_drho_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void dfk_drho_release(int mem) {
}

CASADI_SYMBOL_EXPORT void dfk_drho_incref(void) {
}

CASADI_SYMBOL_EXPORT void dfk_drho_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int dfk_drho_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int dfk_drho_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real dfk_drho_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* dfk_drho_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* dfk_drho_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* dfk_drho_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* dfk_drho_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s2;
    case 2: return casadi_s2;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int dfk_drho_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int dfk_drho_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 4*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
