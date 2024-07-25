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
  #define CASADI_PREFIX(ID) process_jac_ ## ID
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
#define casadi_s3 CASADI_PREFIX(s3)

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

static const casadi_int casadi_s0[26] = {22, 1, 0, 22, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
static const casadi_int casadi_s1[11] = {7, 1, 0, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s3[70] = {22, 22, 0, 1, 2, 3, 5, 7, 9, 18, 27, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 0, 1, 2, 0, 3, 1, 4, 2, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};

/* process_jac:(i0[22],i1[7],i2[7],i3)->(o0[22x22,45nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a102, a103, a104, a105, a106, a107, a108, a109, a11, a110, a111, a112, a113, a114, a115, a116, a117, a118, a119, a12, a120, a121, a122, a123, a124, a125, a126, a127, a128, a129, a13, a130, a131, a132, a133, a134, a135, a136, a137, a138, a139, a14, a140, a141, a142, a143, a144, a145, a146, a147, a148, a149, a15, a150, a151, a152, a153, a154, a155, a156, a157, a158, a159, a16, a160, a161, a162, a163, a164, a165, a166, a167, a168, a169, a17, a170, a171, a172, a173, a174, a175, a176, a177, a178, a179, a18, a180, a181, a182, a183, a184, a185, a186, a187, a188, a189, a19, a190, a191, a192, a193, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
  a0=1.;
  if (res[0]!=0) res[0][0]=a0;
  if (res[0]!=0) res[0][1]=a0;
  if (res[0]!=0) res[0][2]=a0;
  a1=arg[3]? arg[3][0] : 0;
  if (res[0]!=0) res[0][3]=a1;
  if (res[0]!=0) res[0][4]=a0;
  if (res[0]!=0) res[0][5]=a1;
  if (res[0]!=0) res[0][6]=a0;
  if (res[0]!=0) res[0][7]=a1;
  if (res[0]!=0) res[0][8]=a0;
  a2=1.6666666666666666e-01;
  a2=(a2*a1);
  a3=arg[1]? arg[1][4] : 0;
  a4=arg[0]? arg[0][8] : 0;
  a5=cos(a4);
  a6=arg[0]? arg[0][7] : 0;
  a7=sin(a6);
  a8=(a5*a7);
  a9=arg[0]? arg[0][6] : 0;
  a10=cos(a9);
  a11=(a8*a10);
  a12=sin(a4);
  a13=sin(a9);
  a14=(a12*a13);
  a11=(a11+a14);
  a11=(a3*a11);
  a14=arg[1]? arg[1][5] : 0;
  a15=cos(a9);
  a12=(a12*a15);
  a16=sin(a9);
  a8=(a8*a16);
  a12=(a12-a8);
  a12=(a14*a12);
  a11=(a11+a12);
  a12=(a1*a11);
  a8=arg[1]? arg[1][3] : 0;
  a17=arg[2]? arg[2][3] : 0;
  a18=(a8+a17);
  a19=2.;
  a18=(a18/a19);
  a20=sin(a9);
  a21=cos(a6);
  a20=(a20/a21);
  a22=arg[1]? arg[1][1] : 0;
  a23=(a20*a22);
  a24=cos(a9);
  a25=cos(a6);
  a24=(a24/a25);
  a26=arg[1]? arg[1][2] : 0;
  a27=(a24*a26);
  a23=(a23+a27);
  a23=(a1*a23);
  a23=(a23/a19);
  a23=(a4+a23);
  a27=cos(a23);
  a28=cos(a9);
  a28=(a28*a22);
  a29=sin(a9);
  a29=(a29*a26);
  a28=(a28-a29);
  a28=(a1*a28);
  a28=(a28/a19);
  a28=(a6+a28);
  a29=sin(a28);
  a30=5.0000000000000000e-01;
  a31=sin(a9);
  a31=(a22*a31);
  a32=cos(a9);
  a32=(a26*a32);
  a31=(a31+a32);
  a32=(a1*a31);
  a32=(a30*a32);
  a33=(a29*a32);
  a34=(a27*a33);
  a35=cos(a28);
  a36=sin(a23);
  a37=cos(a9);
  a37=(a37/a21);
  a37=(a22*a37);
  a38=sin(a9);
  a38=(a38/a25);
  a38=(a26*a38);
  a37=(a37-a38);
  a38=(a1*a37);
  a38=(a30*a38);
  a39=(a36*a38);
  a40=(a35*a39);
  a34=(a34-a40);
  a34=(a18*a34);
  a40=arg[2]? arg[2][4] : 0;
  a41=(a3+a40);
  a41=(a41/a19);
  a42=sin(a28);
  a43=(a27*a42);
  a44=arg[1]? arg[1][0] : 0;
  a45=sin(a6);
  a46=sin(a9);
  a47=(a45*a46);
  a48=cos(a6);
  a47=(a47/a48);
  a49=(a47*a22);
  a49=(a44+a49);
  a50=cos(a9);
  a51=sin(a6);
  a52=(a50*a51);
  a53=cos(a6);
  a52=(a52/a53);
  a54=(a52*a26);
  a49=(a49+a54);
  a49=(a1*a49);
  a49=(a49/a19);
  a49=(a9+a49);
  a54=cos(a49);
  a55=cos(a9);
  a45=(a45*a55);
  a45=(a45/a48);
  a45=(a22*a45);
  a55=sin(a9);
  a51=(a51*a55);
  a51=(a51/a53);
  a51=(a26*a51);
  a45=(a45-a51);
  a51=(a1*a45);
  a51=(a30*a51);
  a51=(a0+a51);
  a55=(a54*a51);
  a56=(a43*a55);
  a57=sin(a49);
  a39=(a42*a39);
  a58=cos(a28);
  a59=(a58*a32);
  a60=(a27*a59);
  a39=(a39+a60);
  a60=(a57*a39);
  a56=(a56-a60);
  a60=cos(a49);
  a61=cos(a23);
  a62=(a61*a38);
  a63=(a60*a62);
  a64=sin(a23);
  a65=sin(a49);
  a66=(a65*a51);
  a67=(a64*a66);
  a63=(a63-a67);
  a56=(a56-a63);
  a56=(a41*a56);
  a34=(a34+a56);
  a56=arg[2]? arg[2][5] : 0;
  a63=(a14+a56);
  a63=(a63/a19);
  a67=sin(a49);
  a62=(a67*a62);
  a68=cos(a49);
  a69=(a68*a51);
  a70=(a64*a69);
  a62=(a62+a70);
  a70=cos(a49);
  a39=(a70*a39);
  a71=sin(a49);
  a72=(a71*a51);
  a73=(a43*a72);
  a39=(a39+a73);
  a62=(a62-a39);
  a62=(a63*a62);
  a34=(a34+a62);
  a62=(a1*a34);
  a12=(a12+a62);
  a62=(a8+a17);
  a62=(a62/a19);
  a39=sin(a49);
  a73=cos(a28);
  a39=(a39/a73);
  a74=arg[2]? arg[2][1] : 0;
  a75=(a22+a74);
  a75=(a75/a19);
  a76=(a39*a75);
  a77=cos(a49);
  a78=cos(a28);
  a77=(a77/a78);
  a79=arg[2]? arg[2][2] : 0;
  a80=(a26+a79);
  a80=(a80/a19);
  a81=(a77*a80);
  a76=(a76+a81);
  a76=(a1*a76);
  a76=(a76/a19);
  a76=(a4+a76);
  a81=cos(a76);
  a82=cos(a49);
  a82=(a82*a75);
  a83=sin(a49);
  a83=(a83*a80);
  a82=(a82-a83);
  a82=(a1*a82);
  a82=(a82/a19);
  a82=(a6+a82);
  a83=sin(a82);
  a84=sin(a49);
  a85=(a84*a51);
  a85=(a75*a85);
  a86=cos(a49);
  a87=(a86*a51);
  a87=(a80*a87);
  a85=(a85+a87);
  a87=(a1*a85);
  a87=(a30*a87);
  a88=(a83*a87);
  a89=(a81*a88);
  a90=cos(a82);
  a91=sin(a76);
  a92=cos(a49);
  a93=(a92*a51);
  a93=(a93/a73);
  a39=(a39/a73);
  a94=sin(a28);
  a95=(a94*a32);
  a95=(a39*a95);
  a93=(a93-a95);
  a93=(a75*a93);
  a95=sin(a49);
  a96=(a95*a51);
  a96=(a96/a78);
  a77=(a77/a78);
  a97=sin(a28);
  a98=(a97*a32);
  a98=(a77*a98);
  a96=(a96+a98);
  a96=(a80*a96);
  a93=(a93-a96);
  a96=(a1*a93);
  a96=(a30*a96);
  a98=(a91*a96);
  a99=(a90*a98);
  a89=(a89-a99);
  a89=(a62*a89);
  a99=(a3+a40);
  a99=(a99/a19);
  a100=sin(a82);
  a101=(a81*a100);
  a102=arg[2]? arg[2][0] : 0;
  a103=(a44+a102);
  a103=(a103/a19);
  a104=sin(a28);
  a105=sin(a49);
  a106=(a104*a105);
  a107=cos(a28);
  a106=(a106/a107);
  a108=(a106*a75);
  a103=(a103+a108);
  a108=cos(a49);
  a109=sin(a28);
  a110=(a108*a109);
  a111=cos(a28);
  a110=(a110/a111);
  a112=(a110*a80);
  a103=(a103+a112);
  a103=(a1*a103);
  a103=(a103/a19);
  a103=(a9+a103);
  a112=cos(a103);
  a113=cos(a49);
  a114=(a113*a51);
  a114=(a104*a114);
  a115=cos(a28);
  a116=(a115*a32);
  a116=(a105*a116);
  a114=(a114-a116);
  a114=(a114/a107);
  a106=(a106/a107);
  a116=sin(a28);
  a117=(a116*a32);
  a117=(a106*a117);
  a114=(a114-a117);
  a114=(a75*a114);
  a49=sin(a49);
  a51=(a49*a51);
  a51=(a109*a51);
  a117=cos(a28);
  a118=(a117*a32);
  a118=(a108*a118);
  a51=(a51+a118);
  a51=(a51/a111);
  a110=(a110/a111);
  a118=sin(a28);
  a119=(a118*a32);
  a119=(a110*a119);
  a51=(a51+a119);
  a51=(a80*a51);
  a114=(a114-a51);
  a51=(a1*a114);
  a51=(a30*a51);
  a51=(a0+a51);
  a119=(a112*a51);
  a120=(a101*a119);
  a121=sin(a103);
  a98=(a100*a98);
  a122=cos(a82);
  a123=(a122*a87);
  a124=(a81*a123);
  a98=(a98+a124);
  a124=(a121*a98);
  a120=(a120-a124);
  a124=cos(a103);
  a125=cos(a76);
  a126=(a125*a96);
  a127=(a124*a126);
  a128=sin(a76);
  a129=sin(a103);
  a130=(a129*a51);
  a131=(a128*a130);
  a127=(a127-a131);
  a120=(a120-a127);
  a120=(a99*a120);
  a89=(a89+a120);
  a120=(a14+a56);
  a120=(a120/a19);
  a127=sin(a103);
  a126=(a127*a126);
  a131=cos(a103);
  a132=(a131*a51);
  a133=(a128*a132);
  a126=(a126+a133);
  a133=cos(a103);
  a98=(a133*a98);
  a134=sin(a103);
  a135=(a134*a51);
  a136=(a101*a135);
  a98=(a98+a136);
  a126=(a126-a98);
  a126=(a120*a126);
  a89=(a89+a126);
  a126=(a1*a89);
  a12=(a12+a126);
  a12=(a2*a12);
  if (res[0]!=0) res[0][9]=a12;
  a12=sin(a4);
  a126=(a12*a7);
  a98=(a126*a10);
  a136=cos(a4);
  a13=(a136*a13);
  a98=(a98-a13);
  a98=(a3*a98);
  a126=(a126*a16);
  a136=(a136*a15);
  a126=(a126+a136);
  a126=(a14*a126);
  a98=(a98-a126);
  a126=(a1*a98);
  a136=cos(a23);
  a15=(a136*a38);
  a13=(a35*a15);
  a137=sin(a23);
  a33=(a137*a33);
  a13=(a13+a33);
  a13=(a18*a13);
  a15=(a42*a15);
  a59=(a137*a59);
  a15=(a15-a59);
  a59=(a57*a15);
  a33=(a137*a42);
  a138=(a33*a55);
  a59=(a59+a138);
  a138=sin(a23);
  a38=(a138*a38);
  a139=(a60*a38);
  a23=cos(a23);
  a66=(a23*a66);
  a139=(a139+a66);
  a59=(a59-a139);
  a59=(a41*a59);
  a13=(a13+a59);
  a15=(a70*a15);
  a59=(a33*a72);
  a15=(a15-a59);
  a69=(a23*a69);
  a38=(a67*a38);
  a69=(a69-a38);
  a15=(a15-a69);
  a15=(a63*a15);
  a13=(a13+a15);
  a15=(a1*a13);
  a126=(a126+a15);
  a15=cos(a76);
  a69=(a15*a96);
  a38=(a90*a69);
  a59=sin(a76);
  a88=(a59*a88);
  a38=(a38+a88);
  a38=(a62*a38);
  a69=(a100*a69);
  a123=(a59*a123);
  a69=(a69-a123);
  a123=(a121*a69);
  a88=(a59*a100);
  a139=(a88*a119);
  a123=(a123+a139);
  a139=sin(a76);
  a96=(a139*a96);
  a66=(a124*a96);
  a76=cos(a76);
  a130=(a76*a130);
  a66=(a66+a130);
  a123=(a123-a66);
  a123=(a99*a123);
  a38=(a38+a123);
  a69=(a133*a69);
  a123=(a88*a135);
  a69=(a69-a123);
  a132=(a76*a132);
  a96=(a127*a96);
  a132=(a132-a96);
  a69=(a69-a132);
  a69=(a120*a69);
  a38=(a38+a69);
  a69=(a1*a38);
  a126=(a126+a69);
  a126=(a2*a126);
  if (res[0]!=0) res[0][10]=a126;
  a126=cos(a6);
  a10=(a126*a10);
  a10=(a3*a10);
  a126=(a126*a16);
  a126=(a14*a126);
  a10=(a10-a126);
  a126=(a1*a10);
  a16=sin(a28);
  a69=(a16*a32);
  a132=(a57*a69);
  a96=cos(a28);
  a55=(a96*a55);
  a132=(a132+a55);
  a132=(a41*a132);
  a28=cos(a28);
  a32=(a28*a32);
  a32=(a18*a32);
  a132=(a132+a32);
  a69=(a70*a69);
  a72=(a96*a72);
  a69=(a69-a72);
  a69=(a63*a69);
  a132=(a132+a69);
  a69=(a1*a132);
  a126=(a126+a69);
  a69=sin(a82);
  a72=(a69*a87);
  a32=(a121*a72);
  a55=cos(a82);
  a119=(a55*a119);
  a32=(a32+a119);
  a32=(a99*a32);
  a119=cos(a82);
  a123=(a119*a87);
  a123=(a62*a123);
  a32=(a32+a123);
  a72=(a133*a72);
  a135=(a55*a135);
  a72=(a72-a135);
  a72=(a120*a72);
  a32=(a32+a72);
  a72=(a1*a32);
  a126=(a126+a72);
  a126=(a2*a126);
  if (res[0]!=0) res[0][11]=a126;
  a34=(a19*a34);
  a11=(a11+a34);
  a89=(a19*a89);
  a11=(a11+a89);
  a89=sin(a103);
  a34=cos(a82);
  a89=(a89/a34);
  a126=(a22+a74);
  a126=(a126/a19);
  a72=(a89*a126);
  a135=cos(a103);
  a123=cos(a82);
  a135=(a135/a123);
  a66=(a26+a79);
  a66=(a66/a19);
  a130=(a135*a66);
  a72=(a72+a130);
  a72=(a1*a72);
  a72=(a4+a72);
  a130=cos(a72);
  a140=cos(a103);
  a140=(a140*a126);
  a141=sin(a103);
  a141=(a141*a66);
  a140=(a140-a141);
  a140=(a1*a140);
  a140=(a6+a140);
  a141=sin(a140);
  a142=sin(a103);
  a143=(a142*a51);
  a143=(a126*a143);
  a144=cos(a103);
  a145=(a144*a51);
  a145=(a66*a145);
  a143=(a143+a145);
  a145=(a1*a143);
  a146=(a141*a145);
  a147=(a130*a146);
  a148=cos(a140);
  a149=sin(a72);
  a150=cos(a103);
  a151=(a150*a51);
  a151=(a151/a34);
  a89=(a89/a34);
  a152=sin(a82);
  a153=(a152*a87);
  a153=(a89*a153);
  a151=(a151-a153);
  a151=(a126*a151);
  a153=sin(a103);
  a154=(a153*a51);
  a154=(a154/a123);
  a135=(a135/a123);
  a155=sin(a82);
  a156=(a155*a87);
  a156=(a135*a156);
  a154=(a154+a156);
  a154=(a66*a154);
  a151=(a151-a154);
  a154=(a1*a151);
  a156=(a149*a154);
  a157=(a148*a156);
  a147=(a147-a157);
  a147=(a17*a147);
  a157=sin(a140);
  a158=(a130*a157);
  a44=(a44+a102);
  a44=(a44/a19);
  a102=sin(a82);
  a159=sin(a103);
  a160=(a102*a159);
  a161=cos(a82);
  a160=(a160/a161);
  a162=(a160*a126);
  a44=(a44+a162);
  a162=cos(a103);
  a163=sin(a82);
  a164=(a162*a163);
  a165=cos(a82);
  a164=(a164/a165);
  a166=(a164*a66);
  a44=(a44+a166);
  a44=(a1*a44);
  a44=(a9+a44);
  a166=cos(a44);
  a167=cos(a103);
  a168=(a167*a51);
  a168=(a102*a168);
  a169=cos(a82);
  a170=(a169*a87);
  a170=(a159*a170);
  a168=(a168-a170);
  a168=(a168/a161);
  a160=(a160/a161);
  a170=sin(a82);
  a171=(a170*a87);
  a171=(a160*a171);
  a168=(a168-a171);
  a168=(a126*a168);
  a103=sin(a103);
  a51=(a103*a51);
  a51=(a163*a51);
  a171=cos(a82);
  a172=(a171*a87);
  a172=(a162*a172);
  a51=(a51+a172);
  a51=(a51/a165);
  a164=(a164/a165);
  a82=sin(a82);
  a87=(a82*a87);
  a87=(a164*a87);
  a51=(a51+a87);
  a51=(a66*a51);
  a168=(a168-a51);
  a51=(a1*a168);
  a51=(a0+a51);
  a87=(a166*a51);
  a172=(a158*a87);
  a173=sin(a44);
  a156=(a157*a156);
  a174=cos(a140);
  a175=(a174*a145);
  a176=(a130*a175);
  a156=(a156+a176);
  a176=(a173*a156);
  a172=(a172-a176);
  a176=cos(a44);
  a177=cos(a72);
  a178=(a177*a154);
  a179=(a176*a178);
  a180=sin(a72);
  a181=sin(a44);
  a182=(a181*a51);
  a183=(a180*a182);
  a179=(a179-a183);
  a172=(a172-a179);
  a172=(a40*a172);
  a147=(a147+a172);
  a172=sin(a44);
  a178=(a172*a178);
  a179=cos(a44);
  a183=(a179*a51);
  a184=(a180*a183);
  a178=(a178+a184);
  a184=cos(a44);
  a156=(a184*a156);
  a185=sin(a44);
  a186=(a185*a51);
  a187=(a158*a186);
  a156=(a156+a187);
  a178=(a178-a156);
  a178=(a56*a178);
  a147=(a147+a178);
  a11=(a11+a147);
  a11=(a2*a11);
  if (res[0]!=0) res[0][12]=a11;
  a13=(a19*a13);
  a98=(a98+a13);
  a38=(a19*a38);
  a98=(a98+a38);
  a38=cos(a72);
  a13=(a38*a154);
  a11=(a148*a13);
  a147=sin(a72);
  a146=(a147*a146);
  a11=(a11+a146);
  a11=(a17*a11);
  a13=(a157*a13);
  a175=(a147*a175);
  a13=(a13-a175);
  a175=(a173*a13);
  a146=(a147*a157);
  a178=(a146*a87);
  a175=(a175+a178);
  a178=sin(a72);
  a154=(a178*a154);
  a156=(a176*a154);
  a72=cos(a72);
  a182=(a72*a182);
  a156=(a156+a182);
  a175=(a175-a156);
  a175=(a40*a175);
  a11=(a11+a175);
  a13=(a184*a13);
  a175=(a146*a186);
  a13=(a13-a175);
  a183=(a72*a183);
  a154=(a172*a154);
  a183=(a183-a154);
  a13=(a13-a183);
  a13=(a56*a13);
  a11=(a11+a13);
  a98=(a98+a11);
  a98=(a2*a98);
  if (res[0]!=0) res[0][13]=a98;
  a132=(a19*a132);
  a10=(a10+a132);
  a32=(a19*a32);
  a10=(a10+a32);
  a32=sin(a140);
  a132=(a32*a145);
  a98=(a173*a132);
  a11=cos(a140);
  a87=(a11*a87);
  a98=(a98+a87);
  a98=(a40*a98);
  a87=cos(a140);
  a13=(a87*a145);
  a13=(a17*a13);
  a98=(a98+a13);
  a132=(a184*a132);
  a186=(a11*a186);
  a132=(a132-a186);
  a132=(a56*a132);
  a98=(a98+a132);
  a10=(a10+a98);
  a10=(a2*a10);
  if (res[0]!=0) res[0][14]=a10;
  a114=(a19*a114);
  a45=(a45+a114);
  a168=(a19*a168);
  a45=(a45+a168);
  a168=sin(a140);
  a114=cos(a44);
  a10=(a114*a51);
  a10=(a168*a10);
  a98=sin(a44);
  a132=cos(a140);
  a186=(a132*a145);
  a186=(a98*a186);
  a10=(a10-a186);
  a186=cos(a140);
  a10=(a10/a186);
  a13=(a168*a98);
  a13=(a13/a186);
  a13=(a13/a186);
  a183=sin(a140);
  a154=(a183*a145);
  a154=(a13*a154);
  a10=(a10-a154);
  a10=(a74*a10);
  a154=sin(a140);
  a175=sin(a44);
  a156=(a175*a51);
  a156=(a154*a156);
  a182=cos(a44);
  a187=cos(a140);
  a188=(a187*a145);
  a188=(a182*a188);
  a156=(a156+a188);
  a188=cos(a140);
  a156=(a156/a188);
  a189=(a182*a154);
  a189=(a189/a188);
  a189=(a189/a188);
  a190=sin(a140);
  a191=(a190*a145);
  a191=(a189*a191);
  a156=(a156+a191);
  a156=(a79*a156);
  a10=(a10-a156);
  a45=(a45+a10);
  a45=(a2*a45);
  a45=(a0+a45);
  if (res[0]!=0) res[0][15]=a45;
  a85=(a19*a85);
  a31=(a31+a85);
  a143=(a19*a143);
  a31=(a31+a143);
  a143=sin(a44);
  a85=(a143*a51);
  a85=(a74*a85);
  a45=cos(a44);
  a10=(a45*a51);
  a10=(a79*a10);
  a85=(a85+a10);
  a31=(a31+a85);
  a31=(a2*a31);
  a31=(-a31);
  if (res[0]!=0) res[0][16]=a31;
  a93=(a19*a93);
  a37=(a37+a93);
  a151=(a19*a151);
  a37=(a37+a151);
  a151=cos(a44);
  a93=(a151*a51);
  a31=cos(a140);
  a93=(a93/a31);
  a85=sin(a44);
  a85=(a85/a31);
  a85=(a85/a31);
  a10=sin(a140);
  a156=(a10*a145);
  a156=(a85*a156);
  a93=(a93-a156);
  a93=(a74*a93);
  a156=sin(a44);
  a51=(a156*a51);
  a191=cos(a140);
  a51=(a51/a191);
  a44=cos(a44);
  a44=(a44/a191);
  a44=(a44/a191);
  a140=sin(a140);
  a145=(a140*a145);
  a145=(a44*a145);
  a51=(a51+a145);
  a51=(a79*a51);
  a93=(a93-a51);
  a37=(a37+a93);
  a37=(a2*a37);
  if (res[0]!=0) res[0][17]=a37;
  a37=sin(a9);
  a93=cos(a6);
  a51=(a5*a93);
  a145=(a37*a51);
  a145=(a3*a145);
  a192=sin(a6);
  a5=(a5*a192);
  a5=(a8*a5);
  a145=(a145-a5);
  a5=cos(a9);
  a51=(a5*a51);
  a51=(a14*a51);
  a145=(a145+a51);
  a51=(a1*a145);
  a193=(a27*a58);
  a20=(a20/a21);
  a21=sin(a6);
  a20=(a20*a21);
  a20=(a22*a20);
  a24=(a24/a25);
  a25=sin(a6);
  a24=(a24*a25);
  a24=(a26*a24);
  a20=(a20+a24);
  a24=(a1*a20);
  a24=(a30*a24);
  a25=(a36*a24);
  a21=(a42*a25);
  a193=(a193-a21);
  a21=(a57*a193);
  a47=(a47/a48);
  a48=sin(a6);
  a47=(a47*a48);
  a46=(a46+a47);
  a22=(a22*a46);
  a52=(a52/a53);
  a53=sin(a6);
  a52=(a52*a53);
  a50=(a50+a52);
  a26=(a26*a50);
  a22=(a22+a26);
  a26=(a1*a22);
  a26=(a30*a26);
  a54=(a54*a26);
  a50=(a43*a54);
  a21=(a21+a50);
  a50=(a61*a24);
  a52=(a60*a50);
  a65=(a65*a26);
  a53=(a64*a65);
  a52=(a52-a53);
  a21=(a21-a52);
  a21=(a41*a21);
  a25=(a35*a25);
  a27=(a27*a29);
  a25=(a25+a27);
  a25=(a18*a25);
  a21=(a21-a25);
  a50=(a67*a50);
  a68=(a68*a26);
  a64=(a64*a68);
  a50=(a50+a64);
  a193=(a70*a193);
  a71=(a71*a26);
  a43=(a43*a71);
  a193=(a193-a43);
  a50=(a50+a193);
  a50=(a63*a50);
  a21=(a21+a50);
  a50=(a1*a21);
  a51=(a51+a50);
  a84=(a84*a26);
  a84=(a75*a84);
  a86=(a86*a26);
  a86=(a80*a86);
  a84=(a84+a86);
  a86=(a1*a84);
  a86=(a30*a86);
  a86=(a0-a86);
  a122=(a122*a86);
  a50=(a81*a122);
  a92=(a92*a26);
  a92=(a92/a73);
  a39=(a39*a94);
  a92=(a92+a39);
  a92=(a75*a92);
  a77=(a77*a97);
  a95=(a95*a26);
  a95=(a95/a78);
  a77=(a77-a95);
  a77=(a80*a77);
  a92=(a92+a77);
  a77=(a1*a92);
  a77=(a30*a77);
  a95=(a91*a77);
  a78=(a100*a95);
  a50=(a50-a78);
  a78=(a121*a50);
  a105=(a105*a115);
  a113=(a113*a26);
  a104=(a104*a113);
  a105=(a105+a104);
  a105=(a105/a107);
  a106=(a106*a116);
  a105=(a105+a106);
  a75=(a75*a105);
  a108=(a108*a117);
  a49=(a49*a26);
  a109=(a109*a49);
  a108=(a108-a109);
  a108=(a108/a111);
  a110=(a110*a118);
  a108=(a108+a110);
  a80=(a80*a108);
  a75=(a75+a80);
  a80=(a1*a75);
  a30=(a30*a80);
  a112=(a112*a30);
  a80=(a101*a112);
  a78=(a78+a80);
  a80=(a125*a77);
  a108=(a124*a80);
  a129=(a129*a30);
  a110=(a128*a129);
  a108=(a108-a110);
  a78=(a78-a108);
  a78=(a99*a78);
  a95=(a90*a95);
  a83=(a83*a86);
  a81=(a81*a83);
  a95=(a95+a81);
  a95=(a62*a95);
  a78=(a78-a95);
  a80=(a127*a80);
  a131=(a131*a30);
  a128=(a128*a131);
  a80=(a80+a128);
  a50=(a133*a50);
  a134=(a134*a30);
  a101=(a101*a134);
  a50=(a50-a101);
  a80=(a80+a50);
  a80=(a120*a80);
  a78=(a78+a80);
  a80=(a1*a78);
  a51=(a51+a80);
  a51=(a2*a51);
  if (res[0]!=0) res[0][18]=a51;
  a93=(a12*a93);
  a51=(a37*a93);
  a51=(a3*a51);
  a12=(a12*a192);
  a12=(a8*a12);
  a51=(a51-a12);
  a93=(a5*a93);
  a93=(a14*a93);
  a51=(a51+a93);
  a93=(a1*a51);
  a12=(a136*a24);
  a192=(a35*a12);
  a29=(a137*a29);
  a192=(a192-a29);
  a192=(a18*a192);
  a12=(a42*a12);
  a137=(a137*a58);
  a12=(a12+a137);
  a137=(a57*a12);
  a58=(a33*a54);
  a137=(a137+a58);
  a24=(a138*a24);
  a58=(a60*a24);
  a65=(a23*a65);
  a58=(a58+a65);
  a137=(a137-a58);
  a137=(a41*a137);
  a192=(a192+a137);
  a12=(a70*a12);
  a33=(a33*a71);
  a12=(a12-a33);
  a23=(a23*a68);
  a24=(a67*a24);
  a23=(a23-a24);
  a12=(a12-a23);
  a12=(a63*a12);
  a192=(a192+a12);
  a12=(a1*a192);
  a93=(a93+a12);
  a12=(a15*a77);
  a23=(a90*a12);
  a83=(a59*a83);
  a23=(a23-a83);
  a23=(a62*a23);
  a12=(a100*a12);
  a59=(a59*a122);
  a12=(a12+a59);
  a59=(a121*a12);
  a122=(a88*a112);
  a59=(a59+a122);
  a77=(a139*a77);
  a122=(a124*a77);
  a129=(a76*a129);
  a122=(a122+a129);
  a59=(a59-a122);
  a59=(a99*a59);
  a23=(a23+a59);
  a12=(a133*a12);
  a88=(a88*a134);
  a12=(a12-a88);
  a76=(a76*a131);
  a77=(a127*a77);
  a76=(a76-a77);
  a12=(a12-a76);
  a12=(a120*a12);
  a23=(a23+a12);
  a12=(a1*a23);
  a93=(a93+a12);
  a93=(a2*a93);
  if (res[0]!=0) res[0][19]=a93;
  a54=(a96*a54);
  a93=(a57*a16);
  a54=(a54-a93);
  a54=(a41*a54);
  a28=(a18*a28);
  a54=(a54-a28);
  a16=(a70*a16);
  a96=(a96*a71);
  a16=(a16+a96);
  a16=(a63*a16);
  a54=(a54-a16);
  a16=(a1*a54);
  a96=sin(a6);
  a71=(a37*a96);
  a71=(a3*a71);
  a28=cos(a6);
  a28=(a8*a28);
  a71=(a71+a28);
  a96=(a5*a96);
  a96=(a14*a96);
  a71=(a71+a96);
  a96=(a1*a71);
  a16=(a16-a96);
  a112=(a55*a112);
  a69=(a69*a86);
  a96=(a121*a69);
  a112=(a112-a96);
  a112=(a99*a112);
  a119=(a119*a86);
  a119=(a62*a119);
  a112=(a112-a119);
  a69=(a133*a69);
  a55=(a55*a134);
  a69=(a69+a55);
  a69=(a120*a69);
  a112=(a112-a69);
  a69=(a1*a112);
  a16=(a16+a69);
  a16=(a2*a16);
  if (res[0]!=0) res[0][20]=a16;
  a21=(a19*a21);
  a145=(a145+a21);
  a78=(a19*a78);
  a145=(a145+a78);
  a142=(a142*a30);
  a142=(a126*a142);
  a144=(a144*a30);
  a144=(a66*a144);
  a142=(a142+a144);
  a144=(a1*a142);
  a144=(a0-a144);
  a174=(a174*a144);
  a78=(a130*a174);
  a150=(a150*a30);
  a150=(a150/a34);
  a152=(a152*a86);
  a89=(a89*a152);
  a150=(a150+a89);
  a150=(a126*a150);
  a155=(a155*a86);
  a135=(a135*a155);
  a153=(a153*a30);
  a153=(a153/a123);
  a135=(a135-a153);
  a135=(a66*a135);
  a150=(a150+a135);
  a135=(a1*a150);
  a153=(a149*a135);
  a123=(a157*a153);
  a78=(a78-a123);
  a123=(a173*a78);
  a169=(a169*a86);
  a159=(a159*a169);
  a167=(a167*a30);
  a102=(a102*a167);
  a159=(a159+a102);
  a159=(a159/a161);
  a170=(a170*a86);
  a160=(a160*a170);
  a159=(a159+a160);
  a126=(a126*a159);
  a171=(a171*a86);
  a162=(a162*a171);
  a103=(a103*a30);
  a163=(a163*a103);
  a162=(a162-a163);
  a162=(a162/a165);
  a82=(a82*a86);
  a164=(a164*a82);
  a162=(a162+a164);
  a66=(a66*a162);
  a126=(a126+a66);
  a66=(a1*a126);
  a166=(a166*a66);
  a162=(a158*a166);
  a123=(a123+a162);
  a162=(a177*a135);
  a164=(a176*a162);
  a181=(a181*a66);
  a82=(a180*a181);
  a164=(a164-a82);
  a123=(a123-a164);
  a123=(a40*a123);
  a153=(a148*a153);
  a141=(a141*a144);
  a130=(a130*a141);
  a153=(a153+a130);
  a153=(a17*a153);
  a123=(a123-a153);
  a162=(a172*a162);
  a179=(a179*a66);
  a180=(a180*a179);
  a162=(a162+a180);
  a78=(a184*a78);
  a185=(a185*a66);
  a158=(a158*a185);
  a78=(a78-a158);
  a162=(a162+a78);
  a162=(a56*a162);
  a123=(a123+a162);
  a145=(a145+a123);
  a145=(a2*a145);
  if (res[0]!=0) res[0][21]=a145;
  a192=(a19*a192);
  a51=(a51+a192);
  a23=(a19*a23);
  a51=(a51+a23);
  a23=(a38*a135);
  a192=(a148*a23);
  a141=(a147*a141);
  a192=(a192-a141);
  a192=(a17*a192);
  a23=(a157*a23);
  a147=(a147*a174);
  a23=(a23+a147);
  a147=(a173*a23);
  a174=(a146*a166);
  a147=(a147+a174);
  a135=(a178*a135);
  a174=(a176*a135);
  a181=(a72*a181);
  a174=(a174+a181);
  a147=(a147-a174);
  a147=(a40*a147);
  a192=(a192+a147);
  a23=(a184*a23);
  a146=(a146*a185);
  a23=(a23-a146);
  a72=(a72*a179);
  a135=(a172*a135);
  a72=(a72-a135);
  a23=(a23-a72);
  a23=(a56*a23);
  a192=(a192+a23);
  a51=(a51+a192);
  a51=(a2*a51);
  if (res[0]!=0) res[0][22]=a51;
  a54=(a19*a54);
  a54=(a54-a71);
  a112=(a19*a112);
  a54=(a54+a112);
  a166=(a11*a166);
  a32=(a32*a144);
  a112=(a173*a32);
  a166=(a166-a112);
  a166=(a40*a166);
  a87=(a87*a144);
  a87=(a17*a87);
  a166=(a166-a87);
  a32=(a184*a32);
  a11=(a11*a185);
  a32=(a32+a11);
  a32=(a56*a32);
  a166=(a166-a32);
  a54=(a54+a166);
  a54=(a2*a54);
  if (res[0]!=0) res[0][23]=a54;
  a75=(a19*a75);
  a22=(a22+a75);
  a126=(a19*a126);
  a22=(a22+a126);
  a132=(a132*a144);
  a98=(a98*a132);
  a114=(a114*a66);
  a168=(a168*a114);
  a98=(a98+a168);
  a98=(a98/a186);
  a183=(a183*a144);
  a13=(a13*a183);
  a98=(a98+a13);
  a98=(a74*a98);
  a187=(a187*a144);
  a182=(a182*a187);
  a175=(a175*a66);
  a154=(a154*a175);
  a182=(a182-a154);
  a182=(a182/a188);
  a190=(a190*a144);
  a189=(a189*a190);
  a182=(a182+a189);
  a182=(a79*a182);
  a98=(a98+a182);
  a22=(a22+a98);
  a22=(a2*a22);
  if (res[0]!=0) res[0][24]=a22;
  a84=(a19*a84);
  a142=(a19*a142);
  a84=(a84+a142);
  a143=(a143*a66);
  a143=(a74*a143);
  a45=(a45*a66);
  a45=(a79*a45);
  a143=(a143+a45);
  a84=(a84+a143);
  a84=(a2*a84);
  a84=(a0-a84);
  if (res[0]!=0) res[0][25]=a84;
  a92=(a19*a92);
  a20=(a20+a92);
  a150=(a19*a150);
  a20=(a20+a150);
  a151=(a151*a66);
  a151=(a151/a31);
  a10=(a10*a144);
  a85=(a85*a10);
  a151=(a151+a85);
  a74=(a74*a151);
  a140=(a140*a144);
  a44=(a44*a140);
  a156=(a156*a66);
  a156=(a156/a191);
  a44=(a44-a156);
  a79=(a79*a44);
  a74=(a74+a79);
  a20=(a20+a74);
  a20=(a2*a20);
  if (res[0]!=0) res[0][26]=a20;
  a20=sin(a9);
  a74=cos(a4);
  a79=(a20*a74);
  a44=sin(a4);
  a156=(a7*a44);
  a191=(a5*a156);
  a79=(a79-a191);
  a79=(a14*a79);
  a6=cos(a6);
  a44=(a6*a44);
  a44=(a8*a44);
  a156=(a37*a156);
  a9=cos(a9);
  a74=(a9*a74);
  a156=(a156+a74);
  a156=(a3*a156);
  a44=(a44+a156);
  a79=(a79-a44);
  a44=(a1*a79);
  a156=(a67*a61);
  a74=(a42*a36);
  a191=(a70*a74);
  a156=(a156-a191);
  a156=(a63*a156);
  a36=(a35*a36);
  a36=(a18*a36);
  a74=(a57*a74);
  a61=(a60*a61);
  a74=(a74+a61);
  a74=(a41*a74);
  a36=(a36+a74);
  a156=(a156-a36);
  a36=(a1*a156);
  a44=(a44+a36);
  a36=(a127*a125);
  a74=(a100*a91);
  a61=(a133*a74);
  a36=(a36-a61);
  a36=(a120*a36);
  a91=(a90*a91);
  a91=(a62*a91);
  a74=(a121*a74);
  a125=(a124*a125);
  a74=(a74+a125);
  a74=(a99*a74);
  a91=(a91+a74);
  a36=(a36-a91);
  a91=(a1*a36);
  a44=(a44+a91);
  a44=(a2*a44);
  if (res[0]!=0) res[0][27]=a44;
  a44=cos(a4);
  a6=(a6*a44);
  a8=(a8*a6);
  a7=(a7*a44);
  a37=(a37*a7);
  a4=sin(a4);
  a9=(a9*a4);
  a37=(a37-a9);
  a3=(a3*a37);
  a8=(a8+a3);
  a5=(a5*a7);
  a20=(a20*a4);
  a5=(a5+a20);
  a14=(a14*a5);
  a8=(a8+a14);
  a14=(a1*a8);
  a35=(a35*a136);
  a18=(a18*a35);
  a42=(a42*a136);
  a57=(a57*a42);
  a60=(a60*a138);
  a57=(a57-a60);
  a41=(a41*a57);
  a18=(a18+a41);
  a70=(a70*a42);
  a67=(a67*a138);
  a70=(a70+a67);
  a63=(a63*a70);
  a18=(a18+a63);
  a63=(a1*a18);
  a14=(a14+a63);
  a90=(a90*a15);
  a62=(a62*a90);
  a100=(a100*a15);
  a121=(a121*a100);
  a124=(a124*a139);
  a121=(a121-a124);
  a99=(a99*a121);
  a62=(a62+a99);
  a133=(a133*a100);
  a127=(a127*a139);
  a133=(a133+a127);
  a120=(a120*a133);
  a62=(a62+a120);
  a1=(a1*a62);
  a14=(a14+a1);
  a14=(a2*a14);
  if (res[0]!=0) res[0][28]=a14;
  a156=(a19*a156);
  a79=(a79+a156);
  a36=(a19*a36);
  a79=(a79+a36);
  a36=(a172*a177);
  a156=(a157*a149);
  a14=(a184*a156);
  a36=(a36-a14);
  a36=(a56*a36);
  a149=(a148*a149);
  a149=(a17*a149);
  a156=(a173*a156);
  a177=(a176*a177);
  a156=(a156+a177);
  a156=(a40*a156);
  a149=(a149+a156);
  a36=(a36-a149);
  a79=(a79+a36);
  a79=(a2*a79);
  if (res[0]!=0) res[0][29]=a79;
  a18=(a19*a18);
  a8=(a8+a18);
  a19=(a19*a62);
  a8=(a8+a19);
  a148=(a148*a38);
  a17=(a17*a148);
  a157=(a157*a38);
  a173=(a173*a157);
  a176=(a176*a178);
  a173=(a173-a176);
  a40=(a40*a173);
  a17=(a17+a40);
  a184=(a184*a157);
  a172=(a172*a178);
  a184=(a184+a172);
  a56=(a56*a184);
  a17=(a17+a56);
  a8=(a8+a17);
  a2=(a2*a8);
  if (res[0]!=0) res[0][30]=a2;
  if (res[0]!=0) res[0][31]=a0;
  if (res[0]!=0) res[0][32]=a0;
  if (res[0]!=0) res[0][33]=a0;
  if (res[0]!=0) res[0][34]=a0;
  if (res[0]!=0) res[0][35]=a0;
  if (res[0]!=0) res[0][36]=a0;
  if (res[0]!=0) res[0][37]=a0;
  if (res[0]!=0) res[0][38]=a0;
  if (res[0]!=0) res[0][39]=a0;
  if (res[0]!=0) res[0][40]=a0;
  if (res[0]!=0) res[0][41]=a0;
  if (res[0]!=0) res[0][42]=a0;
  if (res[0]!=0) res[0][43]=a0;
  if (res[0]!=0) res[0][44]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int process_jac(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int process_jac_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int process_jac_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void process_jac_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int process_jac_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void process_jac_release(int mem) {
}

CASADI_SYMBOL_EXPORT void process_jac_incref(void) {
}

CASADI_SYMBOL_EXPORT void process_jac_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int process_jac_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int process_jac_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real process_jac_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* process_jac_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* process_jac_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* process_jac_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* process_jac_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int process_jac_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int process_jac_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 1*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif