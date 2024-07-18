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
  #define CASADI_PREFIX(ID) dJ_dq_ ## ID
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
static const casadi_int casadi_s2[52] = {1, 27, 0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* dJ_dq:(i0[12],i1[4])->(o0[1x27,22nz],o1[1x27,22nz],o2[1x27,22nz],o3[1x27,22nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a3, a4, a5, a6, a7, a8, a9;
  a0=8.3799999999999999e-02;
  a1=arg[0]? arg[0][0] : 0;
  a2=cos(a1);
  a3=(a0*a2);
  a4=-2.0000000000000001e-01;
  a5=arg[0]? arg[0][1] : 0;
  a6=cos(a5);
  a7=sin(a1);
  a8=(a6*a7);
  a9=(a4*a8);
  a3=(a3-a9);
  a9=arg[1]? arg[1][0] : 0;
  a10=arg[0]? arg[0][2] : 0;
  a11=cos(a10);
  a8=(a11*a8);
  a12=sin(a10);
  a13=sin(a5);
  a14=(a13*a7);
  a14=(a12*a14);
  a8=(a8-a14);
  a8=(a9*a8);
  a3=(a3+a8);
  a3=(-a3);
  if (res[0]!=0) res[0][0]=a3;
  a7=(a0*a7);
  a3=(a6*a2);
  a8=(a4*a3);
  a7=(a7+a8);
  a2=(a13*a2);
  a2=(a12*a2);
  a3=(a11*a3);
  a2=(a2-a3);
  a2=(a9*a2);
  a7=(a7+a2);
  a7=(-a7);
  if (res[0]!=0) res[0][1]=a7;
  a7=sin(a5);
  a2=cos(a1);
  a3=(a7*a2);
  a8=(a4*a3);
  a14=cos(a5);
  a15=(a14*a2);
  a15=(a12*a15);
  a3=(a11*a3);
  a15=(a15+a3);
  a15=(a9*a15);
  a8=(a8-a15);
  if (res[0]!=0) res[0][2]=a8;
  a8=sin(a1);
  a15=(a7*a8);
  a3=(a4*a15);
  a15=(a11*a15);
  a16=(a14*a8);
  a16=(a12*a16);
  a15=(a15+a16);
  a15=(a9*a15);
  a3=(a3-a15);
  if (res[0]!=0) res[0][3]=a3;
  a3=cos(a10);
  a15=(a13*a2);
  a15=(a3*a15);
  a16=sin(a10);
  a2=(a6*a2);
  a2=(a16*a2);
  a15=(a15+a2);
  a15=(a9*a15);
  a15=(-a15);
  if (res[0]!=0) res[0][4]=a15;
  a15=(a6*a8);
  a15=(a16*a15);
  a8=(a13*a8);
  a8=(a3*a8);
  a15=(a15+a8);
  a15=(a9*a15);
  a15=(-a15);
  if (res[0]!=0) res[0][5]=a15;
  a15=cos(a1);
  a8=cos(a5);
  a2=(a15*a8);
  a2=(a12*a2);
  a17=sin(a5);
  a18=(a15*a17);
  a19=(a11*a18);
  a2=(a2+a19);
  a2=(a9*a2);
  a18=(a4*a18);
  a2=(a2-a18);
  a2=(-a2);
  if (res[0]!=0) res[0][6]=a2;
  a2=sin(a1);
  a18=(a2*a17);
  a19=(a4*a18);
  a20=(a2*a8);
  a20=(a12*a20);
  a18=(a11*a18);
  a20=(a20+a18);
  a20=(a9*a20);
  a19=(a19-a20);
  if (res[0]!=0) res[0][7]=a19;
  a19=sin(a5);
  a20=(a11*a19);
  a5=cos(a5);
  a18=(a12*a5);
  a20=(a20+a18);
  a20=(a9*a20);
  a18=(a4*a19);
  a20=(a20-a18);
  if (res[0]!=0) res[0][8]=a20;
  a20=sin(a1);
  a18=(a20*a5);
  a21=(a4*a18);
  a18=(a11*a18);
  a22=(a20*a19);
  a22=(a12*a22);
  a18=(a18-a22);
  a18=(a9*a18);
  a21=(a21-a18);
  if (res[0]!=0) res[0][9]=a21;
  a1=cos(a1);
  a5=(a1*a5);
  a11=(a11*a5);
  a19=(a1*a19);
  a12=(a12*a19);
  a11=(a11-a12);
  a11=(a9*a11);
  a5=(a4*a5);
  a11=(a11-a5);
  if (res[0]!=0) res[0][10]=a11;
  a11=(a3*a17);
  a5=(a16*a8);
  a11=(a11+a5);
  a11=(a9*a11);
  if (res[0]!=0) res[0][11]=a11;
  a11=(a20*a8);
  a11=(a3*a11);
  a5=(a20*a17);
  a5=(a16*a5);
  a11=(a11-a5);
  a11=(a9*a11);
  a11=(-a11);
  if (res[0]!=0) res[0][12]=a11;
  a8=(a1*a8);
  a3=(a3*a8);
  a17=(a1*a17);
  a16=(a16*a17);
  a3=(a3-a16);
  a3=(a9*a3);
  if (res[0]!=0) res[0][13]=a3;
  a3=(a13*a15);
  a16=cos(a10);
  a3=(a3*a16);
  a15=(a6*a15);
  a17=sin(a10);
  a15=(a15*a17);
  a3=(a3+a15);
  a3=(a9*a3);
  a3=(-a3);
  if (res[0]!=0) res[0][14]=a3;
  a3=(a13*a2);
  a3=(a3*a16);
  a2=(a6*a2);
  a2=(a2*a17);
  a3=(a3+a2);
  a3=(a9*a3);
  a3=(-a3);
  if (res[0]!=0) res[0][15]=a3;
  a3=(a14*a17);
  a2=(a7*a16);
  a3=(a3+a2);
  a3=(a9*a3);
  if (res[0]!=0) res[0][16]=a3;
  a3=(a20*a14);
  a3=(a3*a16);
  a2=(a20*a7);
  a2=(a2*a17);
  a3=(a3-a2);
  a3=(a9*a3);
  a3=(-a3);
  if (res[0]!=0) res[0][17]=a3;
  a14=(a1*a14);
  a14=(a14*a16);
  a7=(a1*a7);
  a7=(a7*a17);
  a14=(a14-a7);
  a14=(a9*a14);
  if (res[0]!=0) res[0][18]=a14;
  a14=sin(a10);
  a7=(a6*a14);
  a10=cos(a10);
  a17=(a13*a10);
  a7=(a7+a17);
  a7=(a9*a7);
  if (res[0]!=0) res[0][19]=a7;
  a7=(a20*a6);
  a7=(a7*a10);
  a20=(a20*a13);
  a20=(a20*a14);
  a7=(a7-a20);
  a7=(a9*a7);
  a7=(-a7);
  if (res[0]!=0) res[0][20]=a7;
  a6=(a1*a6);
  a6=(a6*a10);
  a1=(a1*a13);
  a1=(a1*a14);
  a6=(a6-a1);
  a9=(a9*a6);
  if (res[0]!=0) res[0][21]=a9;
  a9=-8.3799999999999999e-02;
  a6=arg[0]? arg[0][3] : 0;
  a1=cos(a6);
  a14=(a9*a1);
  a13=arg[0]? arg[0][4] : 0;
  a10=cos(a13);
  a7=sin(a6);
  a20=(a10*a7);
  a17=(a4*a20);
  a14=(a14-a17);
  a17=arg[1]? arg[1][1] : 0;
  a16=arg[0]? arg[0][5] : 0;
  a3=cos(a16);
  a20=(a3*a20);
  a2=sin(a16);
  a15=sin(a13);
  a8=(a15*a7);
  a8=(a2*a8);
  a20=(a20-a8);
  a20=(a17*a20);
  a14=(a14+a20);
  a14=(-a14);
  if (res[1]!=0) res[1][0]=a14;
  a7=(a9*a7);
  a14=(a10*a1);
  a20=(a4*a14);
  a7=(a7+a20);
  a1=(a15*a1);
  a1=(a2*a1);
  a14=(a3*a14);
  a1=(a1-a14);
  a1=(a17*a1);
  a7=(a7+a1);
  a7=(-a7);
  if (res[1]!=0) res[1][1]=a7;
  a7=sin(a13);
  a1=cos(a6);
  a14=(a7*a1);
  a20=(a4*a14);
  a8=cos(a13);
  a11=(a8*a1);
  a11=(a2*a11);
  a14=(a3*a14);
  a11=(a11+a14);
  a11=(a17*a11);
  a20=(a20-a11);
  if (res[1]!=0) res[1][2]=a20;
  a20=sin(a6);
  a11=(a7*a20);
  a14=(a4*a11);
  a11=(a3*a11);
  a5=(a8*a20);
  a5=(a2*a5);
  a11=(a11+a5);
  a11=(a17*a11);
  a14=(a14-a11);
  if (res[1]!=0) res[1][3]=a14;
  a14=cos(a16);
  a11=(a15*a1);
  a11=(a14*a11);
  a5=sin(a16);
  a1=(a10*a1);
  a1=(a5*a1);
  a11=(a11+a1);
  a11=(a17*a11);
  a11=(-a11);
  if (res[1]!=0) res[1][4]=a11;
  a11=(a10*a20);
  a11=(a5*a11);
  a20=(a15*a20);
  a20=(a14*a20);
  a11=(a11+a20);
  a11=(a17*a11);
  a11=(-a11);
  if (res[1]!=0) res[1][5]=a11;
  a11=cos(a6);
  a20=cos(a13);
  a1=(a11*a20);
  a1=(a2*a1);
  a12=sin(a13);
  a19=(a11*a12);
  a21=(a3*a19);
  a1=(a1+a21);
  a1=(a17*a1);
  a19=(a4*a19);
  a1=(a1-a19);
  a1=(-a1);
  if (res[1]!=0) res[1][6]=a1;
  a1=sin(a6);
  a19=(a1*a12);
  a21=(a4*a19);
  a18=(a1*a20);
  a18=(a2*a18);
  a19=(a3*a19);
  a18=(a18+a19);
  a18=(a17*a18);
  a21=(a21-a18);
  if (res[1]!=0) res[1][7]=a21;
  a21=sin(a13);
  a18=(a3*a21);
  a13=cos(a13);
  a19=(a2*a13);
  a18=(a18+a19);
  a18=(a17*a18);
  a19=(a4*a21);
  a18=(a18-a19);
  if (res[1]!=0) res[1][8]=a18;
  a18=sin(a6);
  a19=(a18*a13);
  a22=(a4*a19);
  a19=(a3*a19);
  a23=(a18*a21);
  a23=(a2*a23);
  a19=(a19-a23);
  a19=(a17*a19);
  a22=(a22-a19);
  if (res[1]!=0) res[1][9]=a22;
  a6=cos(a6);
  a13=(a6*a13);
  a3=(a3*a13);
  a21=(a6*a21);
  a2=(a2*a21);
  a3=(a3-a2);
  a3=(a17*a3);
  a13=(a4*a13);
  a3=(a3-a13);
  if (res[1]!=0) res[1][10]=a3;
  a3=(a14*a12);
  a13=(a5*a20);
  a3=(a3+a13);
  a3=(a17*a3);
  if (res[1]!=0) res[1][11]=a3;
  a3=(a18*a20);
  a3=(a14*a3);
  a13=(a18*a12);
  a13=(a5*a13);
  a3=(a3-a13);
  a3=(a17*a3);
  a3=(-a3);
  if (res[1]!=0) res[1][12]=a3;
  a20=(a6*a20);
  a14=(a14*a20);
  a12=(a6*a12);
  a5=(a5*a12);
  a14=(a14-a5);
  a14=(a17*a14);
  if (res[1]!=0) res[1][13]=a14;
  a14=(a15*a11);
  a5=cos(a16);
  a14=(a14*a5);
  a11=(a10*a11);
  a12=sin(a16);
  a11=(a11*a12);
  a14=(a14+a11);
  a14=(a17*a14);
  a14=(-a14);
  if (res[1]!=0) res[1][14]=a14;
  a14=(a15*a1);
  a14=(a14*a5);
  a1=(a10*a1);
  a1=(a1*a12);
  a14=(a14+a1);
  a14=(a17*a14);
  a14=(-a14);
  if (res[1]!=0) res[1][15]=a14;
  a14=(a8*a12);
  a1=(a7*a5);
  a14=(a14+a1);
  a14=(a17*a14);
  if (res[1]!=0) res[1][16]=a14;
  a14=(a18*a8);
  a14=(a14*a5);
  a1=(a18*a7);
  a1=(a1*a12);
  a14=(a14-a1);
  a14=(a17*a14);
  a14=(-a14);
  if (res[1]!=0) res[1][17]=a14;
  a8=(a6*a8);
  a8=(a8*a5);
  a7=(a6*a7);
  a7=(a7*a12);
  a8=(a8-a7);
  a8=(a17*a8);
  if (res[1]!=0) res[1][18]=a8;
  a8=sin(a16);
  a7=(a10*a8);
  a16=cos(a16);
  a12=(a15*a16);
  a7=(a7+a12);
  a7=(a17*a7);
  if (res[1]!=0) res[1][19]=a7;
  a7=(a18*a10);
  a7=(a7*a16);
  a18=(a18*a15);
  a18=(a18*a8);
  a7=(a7-a18);
  a7=(a17*a7);
  a7=(-a7);
  if (res[1]!=0) res[1][20]=a7;
  a10=(a6*a10);
  a10=(a10*a16);
  a6=(a6*a15);
  a6=(a6*a8);
  a10=(a10-a6);
  a17=(a17*a10);
  if (res[1]!=0) res[1][21]=a17;
  a17=arg[0]? arg[0][6] : 0;
  a10=cos(a17);
  a6=(a0*a10);
  a8=arg[0]? arg[0][7] : 0;
  a15=cos(a8);
  a16=sin(a17);
  a7=(a15*a16);
  a18=(a4*a7);
  a6=(a6-a18);
  a18=arg[1]? arg[1][2] : 0;
  a12=arg[0]? arg[0][8] : 0;
  a5=cos(a12);
  a7=(a5*a7);
  a14=sin(a12);
  a1=sin(a8);
  a11=(a1*a16);
  a11=(a14*a11);
  a7=(a7-a11);
  a7=(a18*a7);
  a6=(a6+a7);
  a6=(-a6);
  if (res[2]!=0) res[2][0]=a6;
  a0=(a0*a16);
  a16=(a15*a10);
  a6=(a4*a16);
  a0=(a0+a6);
  a10=(a1*a10);
  a10=(a14*a10);
  a16=(a5*a16);
  a10=(a10-a16);
  a10=(a18*a10);
  a0=(a0+a10);
  a0=(-a0);
  if (res[2]!=0) res[2][1]=a0;
  a0=sin(a8);
  a10=cos(a17);
  a16=(a0*a10);
  a6=(a4*a16);
  a7=cos(a8);
  a11=(a7*a10);
  a11=(a14*a11);
  a16=(a5*a16);
  a11=(a11+a16);
  a11=(a18*a11);
  a6=(a6-a11);
  if (res[2]!=0) res[2][2]=a6;
  a6=sin(a17);
  a11=(a0*a6);
  a16=(a4*a11);
  a11=(a5*a11);
  a20=(a7*a6);
  a20=(a14*a20);
  a11=(a11+a20);
  a11=(a18*a11);
  a16=(a16-a11);
  if (res[2]!=0) res[2][3]=a16;
  a16=cos(a12);
  a11=(a1*a10);
  a11=(a16*a11);
  a20=sin(a12);
  a10=(a15*a10);
  a10=(a20*a10);
  a11=(a11+a10);
  a11=(a18*a11);
  a11=(-a11);
  if (res[2]!=0) res[2][4]=a11;
  a11=(a15*a6);
  a11=(a20*a11);
  a6=(a1*a6);
  a6=(a16*a6);
  a11=(a11+a6);
  a11=(a18*a11);
  a11=(-a11);
  if (res[2]!=0) res[2][5]=a11;
  a11=cos(a17);
  a6=cos(a8);
  a10=(a11*a6);
  a10=(a14*a10);
  a3=sin(a8);
  a13=(a11*a3);
  a2=(a5*a13);
  a10=(a10+a2);
  a10=(a18*a10);
  a13=(a4*a13);
  a10=(a10-a13);
  a10=(-a10);
  if (res[2]!=0) res[2][6]=a10;
  a10=sin(a17);
  a13=(a10*a3);
  a2=(a4*a13);
  a21=(a10*a6);
  a21=(a14*a21);
  a13=(a5*a13);
  a21=(a21+a13);
  a21=(a18*a21);
  a2=(a2-a21);
  if (res[2]!=0) res[2][7]=a2;
  a2=sin(a8);
  a21=(a5*a2);
  a8=cos(a8);
  a13=(a14*a8);
  a21=(a21+a13);
  a21=(a18*a21);
  a13=(a4*a2);
  a21=(a21-a13);
  if (res[2]!=0) res[2][8]=a21;
  a21=sin(a17);
  a13=(a21*a8);
  a22=(a4*a13);
  a13=(a5*a13);
  a19=(a21*a2);
  a19=(a14*a19);
  a13=(a13-a19);
  a13=(a18*a13);
  a22=(a22-a13);
  if (res[2]!=0) res[2][9]=a22;
  a17=cos(a17);
  a8=(a17*a8);
  a5=(a5*a8);
  a2=(a17*a2);
  a14=(a14*a2);
  a5=(a5-a14);
  a5=(a18*a5);
  a8=(a4*a8);
  a5=(a5-a8);
  if (res[2]!=0) res[2][10]=a5;
  a5=(a16*a3);
  a8=(a20*a6);
  a5=(a5+a8);
  a5=(a18*a5);
  if (res[2]!=0) res[2][11]=a5;
  a5=(a21*a6);
  a5=(a16*a5);
  a8=(a21*a3);
  a8=(a20*a8);
  a5=(a5-a8);
  a5=(a18*a5);
  a5=(-a5);
  if (res[2]!=0) res[2][12]=a5;
  a6=(a17*a6);
  a16=(a16*a6);
  a3=(a17*a3);
  a20=(a20*a3);
  a16=(a16-a20);
  a16=(a18*a16);
  if (res[2]!=0) res[2][13]=a16;
  a16=(a1*a11);
  a20=cos(a12);
  a16=(a16*a20);
  a11=(a15*a11);
  a3=sin(a12);
  a11=(a11*a3);
  a16=(a16+a11);
  a16=(a18*a16);
  a16=(-a16);
  if (res[2]!=0) res[2][14]=a16;
  a16=(a1*a10);
  a16=(a16*a20);
  a10=(a15*a10);
  a10=(a10*a3);
  a16=(a16+a10);
  a16=(a18*a16);
  a16=(-a16);
  if (res[2]!=0) res[2][15]=a16;
  a16=(a7*a3);
  a10=(a0*a20);
  a16=(a16+a10);
  a16=(a18*a16);
  if (res[2]!=0) res[2][16]=a16;
  a16=(a21*a7);
  a16=(a16*a20);
  a10=(a21*a0);
  a10=(a10*a3);
  a16=(a16-a10);
  a16=(a18*a16);
  a16=(-a16);
  if (res[2]!=0) res[2][17]=a16;
  a7=(a17*a7);
  a7=(a7*a20);
  a0=(a17*a0);
  a0=(a0*a3);
  a7=(a7-a0);
  a7=(a18*a7);
  if (res[2]!=0) res[2][18]=a7;
  a7=sin(a12);
  a0=(a15*a7);
  a12=cos(a12);
  a3=(a1*a12);
  a0=(a0+a3);
  a0=(a18*a0);
  if (res[2]!=0) res[2][19]=a0;
  a0=(a21*a15);
  a0=(a0*a12);
  a21=(a21*a1);
  a21=(a21*a7);
  a0=(a0-a21);
  a0=(a18*a0);
  a0=(-a0);
  if (res[2]!=0) res[2][20]=a0;
  a15=(a17*a15);
  a15=(a15*a12);
  a17=(a17*a1);
  a17=(a17*a7);
  a15=(a15-a17);
  a18=(a18*a15);
  if (res[2]!=0) res[2][21]=a18;
  a18=arg[0]? arg[0][9] : 0;
  a15=cos(a18);
  a17=(a9*a15);
  a7=arg[0]? arg[0][10] : 0;
  a1=cos(a7);
  a12=sin(a18);
  a0=(a1*a12);
  a21=(a4*a0);
  a17=(a17-a21);
  a21=arg[1]? arg[1][3] : 0;
  a3=arg[0]? arg[0][11] : 0;
  a20=cos(a3);
  a0=(a20*a0);
  a16=sin(a3);
  a10=sin(a7);
  a11=(a10*a12);
  a11=(a16*a11);
  a0=(a0-a11);
  a0=(a21*a0);
  a17=(a17+a0);
  a17=(-a17);
  if (res[3]!=0) res[3][0]=a17;
  a9=(a9*a12);
  a12=(a1*a15);
  a17=(a4*a12);
  a9=(a9+a17);
  a15=(a10*a15);
  a15=(a16*a15);
  a12=(a20*a12);
  a15=(a15-a12);
  a15=(a21*a15);
  a9=(a9+a15);
  a9=(-a9);
  if (res[3]!=0) res[3][1]=a9;
  a9=sin(a7);
  a15=cos(a18);
  a12=(a9*a15);
  a17=(a4*a12);
  a0=cos(a7);
  a11=(a0*a15);
  a11=(a16*a11);
  a12=(a20*a12);
  a11=(a11+a12);
  a11=(a21*a11);
  a17=(a17-a11);
  if (res[3]!=0) res[3][2]=a17;
  a17=sin(a18);
  a11=(a9*a17);
  a12=(a4*a11);
  a11=(a20*a11);
  a6=(a0*a17);
  a6=(a16*a6);
  a11=(a11+a6);
  a11=(a21*a11);
  a12=(a12-a11);
  if (res[3]!=0) res[3][3]=a12;
  a12=cos(a3);
  a11=(a10*a15);
  a11=(a12*a11);
  a6=sin(a3);
  a15=(a1*a15);
  a15=(a6*a15);
  a11=(a11+a15);
  a11=(a21*a11);
  a11=(-a11);
  if (res[3]!=0) res[3][4]=a11;
  a11=(a1*a17);
  a11=(a6*a11);
  a17=(a10*a17);
  a17=(a12*a17);
  a11=(a11+a17);
  a11=(a21*a11);
  a11=(-a11);
  if (res[3]!=0) res[3][5]=a11;
  a11=cos(a18);
  a17=cos(a7);
  a15=(a11*a17);
  a15=(a16*a15);
  a5=sin(a7);
  a8=(a11*a5);
  a14=(a20*a8);
  a15=(a15+a14);
  a15=(a21*a15);
  a8=(a4*a8);
  a15=(a15-a8);
  a15=(-a15);
  if (res[3]!=0) res[3][6]=a15;
  a15=sin(a18);
  a8=(a15*a5);
  a14=(a4*a8);
  a2=(a15*a17);
  a2=(a16*a2);
  a8=(a20*a8);
  a2=(a2+a8);
  a2=(a21*a2);
  a14=(a14-a2);
  if (res[3]!=0) res[3][7]=a14;
  a14=sin(a7);
  a2=(a20*a14);
  a7=cos(a7);
  a8=(a16*a7);
  a2=(a2+a8);
  a2=(a21*a2);
  a8=(a4*a14);
  a2=(a2-a8);
  if (res[3]!=0) res[3][8]=a2;
  a2=sin(a18);
  a8=(a2*a7);
  a22=(a4*a8);
  a8=(a20*a8);
  a13=(a2*a14);
  a13=(a16*a13);
  a8=(a8-a13);
  a8=(a21*a8);
  a22=(a22-a8);
  if (res[3]!=0) res[3][9]=a22;
  a18=cos(a18);
  a7=(a18*a7);
  a20=(a20*a7);
  a14=(a18*a14);
  a16=(a16*a14);
  a20=(a20-a16);
  a20=(a21*a20);
  a4=(a4*a7);
  a20=(a20-a4);
  if (res[3]!=0) res[3][10]=a20;
  a20=(a12*a5);
  a4=(a6*a17);
  a20=(a20+a4);
  a20=(a21*a20);
  if (res[3]!=0) res[3][11]=a20;
  a20=(a2*a17);
  a20=(a12*a20);
  a4=(a2*a5);
  a4=(a6*a4);
  a20=(a20-a4);
  a20=(a21*a20);
  a20=(-a20);
  if (res[3]!=0) res[3][12]=a20;
  a17=(a18*a17);
  a12=(a12*a17);
  a5=(a18*a5);
  a6=(a6*a5);
  a12=(a12-a6);
  a12=(a21*a12);
  if (res[3]!=0) res[3][13]=a12;
  a12=(a10*a11);
  a6=cos(a3);
  a12=(a12*a6);
  a11=(a1*a11);
  a5=sin(a3);
  a11=(a11*a5);
  a12=(a12+a11);
  a12=(a21*a12);
  a12=(-a12);
  if (res[3]!=0) res[3][14]=a12;
  a12=(a10*a15);
  a12=(a12*a6);
  a15=(a1*a15);
  a15=(a15*a5);
  a12=(a12+a15);
  a12=(a21*a12);
  a12=(-a12);
  if (res[3]!=0) res[3][15]=a12;
  a12=(a0*a5);
  a15=(a9*a6);
  a12=(a12+a15);
  a12=(a21*a12);
  if (res[3]!=0) res[3][16]=a12;
  a12=(a2*a0);
  a12=(a12*a6);
  a15=(a2*a9);
  a15=(a15*a5);
  a12=(a12-a15);
  a12=(a21*a12);
  a12=(-a12);
  if (res[3]!=0) res[3][17]=a12;
  a0=(a18*a0);
  a0=(a0*a6);
  a9=(a18*a9);
  a9=(a9*a5);
  a0=(a0-a9);
  a0=(a21*a0);
  if (res[3]!=0) res[3][18]=a0;
  a0=sin(a3);
  a9=(a1*a0);
  a3=cos(a3);
  a5=(a10*a3);
  a9=(a9+a5);
  a9=(a21*a9);
  if (res[3]!=0) res[3][19]=a9;
  a9=(a2*a1);
  a9=(a9*a3);
  a2=(a2*a10);
  a2=(a2*a0);
  a9=(a9-a2);
  a9=(a21*a9);
  a9=(-a9);
  if (res[3]!=0) res[3][20]=a9;
  a1=(a18*a1);
  a1=(a1*a3);
  a18=(a18*a10);
  a18=(a18*a0);
  a1=(a1-a18);
  a21=(a21*a1);
  if (res[3]!=0) res[3][21]=a21;
  return 0;
}

CASADI_SYMBOL_EXPORT int dJ_dq(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int dJ_dq_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int dJ_dq_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void dJ_dq_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int dJ_dq_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void dJ_dq_release(int mem) {
}

CASADI_SYMBOL_EXPORT void dJ_dq_incref(void) {
}

CASADI_SYMBOL_EXPORT void dJ_dq_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int dJ_dq_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int dJ_dq_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real dJ_dq_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* dJ_dq_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* dJ_dq_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* dJ_dq_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* dJ_dq_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s2;
    case 2: return casadi_s2;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int dJ_dq_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int dJ_dq_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 4*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
