#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pycuda-complex.hpp>
#define Mz 91.19 // from 2016 PDG
#define Mw 80.385 // from 2016 PDG
#define w 0.5017107831701941 // from 2016 PDG
#define MZ2 8315.
#define mt 162.6
#define sinsqtw 0.23119 // Added by Miriam, check if it's repeated

__device__ void Dagger(pycuda::complex<double> A[2][2],pycuda::complex<double> Adag[2][2]){
  int ie,je;
  for(ie=0;ie<2;ie++){
    for(je=0;je<2;je++){
      Adag[ie][je]=pycuda::conj(A[je][ie]);
    }
  }
  return;
}

__device__ void mult2x2(double out[2][2], double a[2][2], double b[2][2])
{
  double c11 = a[0][0]*b[0][0] + a[0][1]*b[1][0];
  double c12 = a[0][0]*b[0][1] + a[0][1]*b[1][1];
  double c21 = a[1][0]*b[0][0] + a[1][1]*b[1][0];
  double c22 = a[1][0]*b[0][1] + a[1][1]*b[1][1];

  out[0][0] = c11;
  out[0][1] = c12;
  out[1][0] = c21; 
  out[1][1] = c22;               
  return;
  
}

__device__ void MatrixProduct(pycuda::complex<double> a[2][2],pycuda::complex<double> b[2][2],pycuda::complex<double> out[2][2]){
  // TODO: generalize

  out[0][0] =  a[0][0]*b[0][0] + a[0][1]*b[1][0];
  out[0][1] =  a[0][0]*b[0][1] + a[0][1]*b[1][1];
  out[1][0] =  a[1][0]*b[0][0] + a[1][1]*b[1][0];
  out[1][1] =  a[1][0]*b[0][1] + a[1][1]*b[1][1];
  return;
}


__device__ void get_coefficients_neutralino(double M1,double M2,
double mu,double b,pycuda::complex<double> a[5],pycuda::complex<double> MN[4][4])
{
  
  // Returns coefficients a4,a3,a2,a1,a0, where : a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0
  // From the neutralino matrix 

  a[0] = -M1*M2*mu*mu + 12755.5474708557*M1*mu*sin(b)*cos(b) + 3868.3903291443*M2*mu*sin(b)*cos(b);
  a[1] = (M1*mu*mu + 6377.77373542785*M1*sin(b)*sin(b) + 6377.77373542785*M1*cos(b)*cos(b) + M2*mu*mu 
          + 1934.19516457215*M2*sin(b)*sin(b) + 1934.19516457215*M2*cos(b)*cos(b) - 16623.9378*mu*sin(b)*cos(b));
  a[2] = (M1*M2 - mu*mu - 8311.9689*sin(b)*sin(b) - 8311.9689*cos(b)*cos(b));
  a[3] = (-M1 - M2);
  a[4] = 1.0;

  // Returns the matrix coefficients for the neutralino matrix:
  double Mss,Mcs,Msc,Mcc;
  Mss = Mz*sin(b)*sin(w);
  Mcs = Mz*cos(b)*sin(w);
  Msc = Mz*sin(b)*cos(w);
  Mcc = Mz*cos(b)*cos(w);
  MN[0][0] = M1;
  MN[0][1] = 0.0;
  MN[0][2] = -Mcs;
  MN[0][3] = Mss;
  MN[1][0] = 0.0;
  MN[1][1] = M2;
  MN[1][2] = Mcc;
  MN[1][3] = -Msc;
  MN[2][0] = -Mcs;
  MN[2][1] = Mcc;
  MN[2][2] = 0.0;
  MN[2][3] = -mu;
  MN[3][0] = Mss;
  MN[3][1] = -Msc;
  MN[3][2] = -mu;
  MN[3][3] = 0.0;
  return ;

}

__device__ void get_coefficients_chargino(double M2,double mu,double b,pycuda::complex<double> a[3])
{
  // Returns the matrix coefficients for the chargino matrix (9.7 in Theory and Phenomenology of Sparticles)
  
  pycuda::complex<double> Mch[2][2], Mchdag[2][2], MchMchdag[2][2], MchdagMch[2][2];
  
  Mch[0][0] = M2;
  Mch[0][1] = sqrt(2.)*Mw*sin(b);
  Mch[1][0] = sqrt(2.)*Mw*cos(b);
  Mch[1][1] = mu;

  Dagger(Mch,Mchdag);
  MatrixProduct(Mch,Mchdag,MchMchdag);
  MatrixProduct(Mchdag,Mch,MchdagMch); //WR
  // Returns coefficients, where : a[3]*x**2 + a[2]*x + a[1]
  // a11*a22 - a11*x - a12*a21 - a22*x + x**2
  a[0] = MchMchdag[0][0]*MchMchdag[1][1]  - MchMchdag[0][1]*MchMchdag[1][0];
  a[1] = - MchMchdag[0][0] - MchMchdag[1][1];
  a[2] = 1.0;
  
  return ;

}

__device__ void get_coefficients_stop(double At,double mtL2,double mtR2,double b,double mu,pycuda::complex<double> a[3])
{
  // Returns the matrix coefficients for the stop matrix (SLHA1 paper)
  pycuda::complex<double> Mstop[2][2];
  
  Mstop[0][0] = mtL2 + mt*mt + (0.5 - 2.*sinsqtw/3)*MZ2*cos(2*b);
  Mstop[0][1] = mt*(At - mu/tan(b));
  Mstop[1][0] = mt*(At + mu/tan(b));
  Mstop[1][1] = mtR2 + mt*mt + 2.*sinsqtw/3*MZ2*cos(2*b);
  // Returns coefficients, where : a[2]*x**2 + a[1]*x + a[0]
  // a11*a22 - a11*x - a12*a21 - a22*x + x**2
  a[0] = Mstop[0][0]*Mstop[1][1]  - Mstop[0][1]*Mstop[1][0];
  a[1] = - Mstop[0][0] - Mstop[1][1];
  a[2] = 1.0;
  
  return ;

}

__device__ void inv2x2(double out[2][2], double a[2][2])
{
  double a11 = 1.0/a[0][0] + a[0][1]*a[1][0]/(pow(a[0][0], 2)*(a[1][1] - a[0][1]
*a[1][0]/a[0][0]));
  double a12 = - a[1][0]/(a[0][0]*(a[1][1] - a[0][1]*a[1][0]/a[0][0]));
  double a21 = - a[1][0]/(a[0][0]*(a[1][1] - a[0][1]*a[1][0]/a[0][0]));
  double a22 = 1.0/(a[1][1] - a[0][1]*a[1][0]/a[0][0]);

  out[0][0] = a11;
  out[0][1] = a12;
  out[1][0] = a21;
  out[1][1] = a22;
  return;
  
}

__device__ void get_inv_2d(pycuda::complex<double> a[2][2],pycuda::complex<double> out[2][2])
{
  out[0][0] = 1.0/a[0][0] + a[0][1]*a[1][0]/(pow(a[0][0], 2)*(a[1][1] - a[0][1]
*a[1][0]/a[0][0]));
  out[0][1] = - a[1][0]/(a[0][0]*(a[1][1] - a[0][1]*a[1][0]/a[0][0]));
  out[1][0] = - a[1][0]/(a[0][0]*(a[1][1] - a[0][1]*a[1][0]/a[0][0]));
  out[1][1] = 1.0/(a[1][1] - a[0][1]*a[1][0]/a[0][0]);
  return;
}

__device__ void get_inv_4d(pycuda::complex<double> a[4][4],pycuda::complex<double> ainv[4][4])
{

  // Returns inverse of a 4x4 matrix
  pycuda::complex<double> a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44;
  a11 = a[0][0];
  a12 = a[0][1];
  a13 = a[0][2];
  a14 = a[0][3];
  a21 = a[1][0];
  a22 = a[1][1];
  a23 = a[1][2];
  a24 = a[1][3];
  a31 = a[2][0];
  a32 = a[2][1];
  a33 = a[2][2];
  a34 = a[2][3];
  a41 = a[3][0];
  a42 = a[3][1];
  a43 = a[3][2];
  a44 = a[3][3];

  ainv[0][0] = -(a11*a22 - a12*a21)*(-a12*(a23 - a13*a21/a11)/(a11*(a22 - a12*a21/a11)) + a13/a11)*(a21*(a32 - a12*a31/a11)/(a11*(a22 - a12*a21/a11)) - a31/a11)/(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31) - (-(a11*a22 - a12*a21)*(-a12*(a23 - a13*a21/a11)/(a11*(a22 - a12*a21/a11)) + a13/a11)*(a34 - (a24 - a14*a21/a11)*(a32 - a12*a31/a11)/(a22 - a12*a21/a11) - a14*a31/a11)/(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31) - a12*(a24 - a14*a21/a11)/(a11*(a22 - a12*a21/a11)) + a14/a11)*(-(a11*a22 - a12*a21)*(a21*(a32 - a12*a31/a11)/(a11*(a22 - a12*a21/a11)) - a31/a11)*(a43 - (a23 - a13*a21/a11)*(a42 - a12*a41/a11)/(a22 - a12*a21/a11) - a13*a41/a11)/(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31) + a21*(a42 - a12*a41/a11)/(a11*(a22 - a12*a21/a11)) - a41/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41) + 1.0/a11 + a12*a21/(pycuda::pow(a11,2)*(a22 - a12*a21/a11));
  ainv[0][1] = -((a32 - a12*a31/a11)*(a11*a22 - a12*a21)*(a43 - (a23 - a13*a21/a11)*(a42 - a12*a41/a11)/(a22 - a12*a21/a11) - a13*a41/a11)/((a22 - a12*a21/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)) - (a42 - a12*a41/a11)/(a22 - a12*a21/a11))*(-(a11*a22 - a12*a21)*(-a12*(a23 - a13*a21/a11)/(a11*(a22 - a12*a21/a11)) + a13/a11)*(a34 - (a24 - a14*a21/a11)*(a32 - a12*a31/a11)/(a22 - a12*a21/a11) - a14*a31/a11)/(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31) - a12*(a24 - a14*a21/a11)/(a11*(a22 - a12*a21/a11)) + a14/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41) + (a32 - a12*a31/a11)*(a11*a22 - a12*a21)*(-a12*(a23 - a13*a21/a11)/(a11*(a22 - a12*a21/a11)) + a13/a11)/((a22 - a12*a21/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)) - a12/(a11*(a22 - a12*a21/a11));
  ainv[0][2] = -(a11*a22 - a12*a21)*(-a12*(a23 - a13*a21/a11)/(a11*(a22 - a12*a21/a11)) + a13/a11)/(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31) + (a11*a22 - a12*a21)*(a43 - (a23 - a13*a21/a11)*(a42 - a12*a41/a11)/(a22 - a12*a21/a11) - a13*a41/a11)*(-(a11*a22 - a12*a21)*(-a12*(a23 - a13*a21/a11)/(a11*(a22 - a12*a21/a11)) + a13/a11)*(a34 - (a24 - a14*a21/a11)*(a32 - a12*a31/a11)/(a22 - a12*a21/a11) - a14*a31/a11)/(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31) - a12*(a24 - a14*a21/a11)/(a11*(a22 - a12*a21/a11)) + a14/a11)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41);
  ainv[0][3] = -(-(a11*a22 - a12*a21)*(-a12*(a23 - a13*a21/a11)/(a11*(a22 - a12*a21/a11)) + a13/a11)*(a34 - (a24 - a14*a21/a11)*(a32 - a12*a31/a11)/(a22 - a12*a21/a11) - a14*a31/a11)/(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31) - a12*(a24 - a14*a21/a11)/(a11*(a22 - a12*a21/a11)) + a14/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41);
  ainv[1][0] = -(-(a23 - a13*a21/a11)*(a11*a22 - a12*a21)*(a34 - (a24 - a14*a21/a11)*(a32 - a12*a31/a11)/(a22 - a12*a21/a11) - a14*a31/a11)/((a22 - a12*a21/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)) + (a24 - a14*a21/a11)/(a22 - a12*a21/a11))*(-(a11*a22 - a12*a21)*(a21*(a32 - a12*a31/a11)/(a11*(a22 - a12*a21/a11)) - a31/a11)*(a43 - (a23 - a13*a21/a11)*(a42 - a12*a41/a11)/(a22 - a12*a21/a11) - a13*a41/a11)/(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31) + a21*(a42 - a12*a41/a11)/(a11*(a22 - a12*a21/a11)) - a41/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41) - (a23 - a13*a21/a11)*(a11*a22 - a12*a21)*(a21*(a32 - a12*a31/a11)/(a11*(a22 - a12*a21/a11)) - a31/a11)/((a22 - a12*a21/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)) - a21/(a11*(a22 - a12*a21/a11));
  ainv[1][1] = -(-(a23 - a13*a21/a11)*(a11*a22 - a12*a21)*(a34 - (a24 - a14*a21/a11)*(a32 - a12*a31/a11)/(a22 - a12*a21/a11) - a14*a31/a11)/((a22 - a12*a21/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)) + (a24 - a14*a21/a11)/(a22 - a12*a21/a11))*((a32 - a12*a31/a11)*(a11*a22 - a12*a21)*(a43 - (a23 - a13*a21/a11)*(a42 - a12*a41/a11)/(a22 - a12*a21/a11) - a13*a41/a11)/((a22 - a12*a21/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)) - (a42 - a12*a41/a11)/(a22 - a12*a21/a11))*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41) + 1.0/(a22 - a12*a21/a11) + (a23 - a13*a21/a11)*(a32 - a12*a31/a11)*(a11*a22 - a12*a21)/(pycuda::pow(a22 - a12*a21/a11,2)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31));
  ainv[1][2] = (a11*a22 - a12*a21)*(-(a23 - a13*a21/a11)*(a11*a22 - a12*a21)*(a34 - (a24 - a14*a21/a11)*(a32 - a12*a31/a11)/(a22 - a12*a21/a11) - a14*a31/a11)/((a22 - a12*a21/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)) + (a24 - a14*a21/a11)/(a22 - a12*a21/a11))*(a43 - (a23 - a13*a21/a11)*(a42 - a12*a41/a11)/(a22 - a12*a21/a11) - a13*a41/a11)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41) - (a23 - a13*a21/a11)*(a11*a22 - a12*a21)/((a22 - a12*a21/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31));
  ainv[1][3] = -(-(a23 - a13*a21/a11)*(a11*a22 - a12*a21)*(a34 - (a24 - a14*a21/a11)*(a32 - a12*a31/a11)/(a22 - a12*a21/a11) - a14*a31/a11)/((a22 - a12*a21/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)) + (a24 - a14*a21/a11)/(a22 - a12*a21/a11))*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41);
  ainv[2][0] = (a11*a22 - a12*a21)*(a21*(a32 - a12*a31/a11)/(a11*(a22 - a12*a21/a11)) - a31/a11)/(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31) - (a11*a22 - a12*a21)*(a34 - (a24 - a14*a21/a11)*(a32 - a12*a31/a11)/(a22 - a12*a21/a11) - a14*a31/a11)*(-(a11*a22 - a12*a21)*(a21*(a32 - a12*a31/a11)/(a11*(a22 - a12*a21/a11)) - a31/a11)*(a43 - (a23 - a13*a21/a11)*(a42 - a12*a41/a11)/(a22 - a12*a21/a11) - a13*a41/a11)/(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31) + a21*(a42 - a12*a41/a11)/(a11*(a22 - a12*a21/a11)) - a41/a11)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41);
  ainv[2][1] = -(a11*a22 - a12*a21)*((a32 - a12*a31/a11)*(a11*a22 - a12*a21)*(a43 - (a23 - a13*a21/a11)*(a42 - a12*a41/a11)/(a22 - a12*a21/a11) - a13*a41/a11)/((a22 - a12*a21/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)) - (a42 - a12*a41/a11)/(a22 - a12*a21/a11))*(a34 - (a24 - a14*a21/a11)*(a32 - a12*a31/a11)/(a22 - a12*a21/a11) - a14*a31/a11)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41) - (a32 - a12*a31/a11)*(a11*a22 - a12*a21)/((a22 - a12*a21/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31));
  ainv[2][2] = pycuda::pow(a11*a22 - a12*a21,2)*(a34 - (a24 - a14*a21/a11)*(a32 - a12*a31/a11)/(a22 - a12*a21/a11) - a14*a31/a11)*(a43 - (a23 - a13*a21/a11)*(a42 - a12*a41/a11)/(a22 - a12*a21/a11) - a13*a41/a11)/((a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)*(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41)) + (a11*a22 - a12*a21)/(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31);
  ainv[2][3] = -(a11*a22 - a12*a21)*(a34 - (a24 - a14*a21/a11)*(a32 - a12*a31/a11)/(a22 - a12*a21/a11) - a14*a31/a11)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41);
  ainv[3][0] = (-(a11*a22 - a12*a21)*(a21*(a32 - a12*a31/a11)/(a11*(a22 - a12*a21/a11)) - a31/a11)*(a43 - (a23 - a13*a21/a11)*(a42 - a12*a41/a11)/(a22 - a12*a21/a11) - a13*a41/a11)/(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31) + a21*(a42 - a12*a41/a11)/(a11*(a22 - a12*a21/a11)) - a41/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41);
  ainv[3][1] = ((a32 - a12*a31/a11)*(a11*a22 - a12*a21)*(a43 - (a23 - a13*a21/a11)*(a42 - a12*a41/a11)/(a22 - a12*a21/a11) - a13*a41/a11)/((a22 - a12*a21/a11)*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)) - (a42 - a12*a41/a11)/(a22 - a12*a21/a11))*(a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41);
  ainv[3][2] = -(a11*a22 - a12*a21)*(a43 - (a23 - a13*a21/a11)*(a42 - a12*a41/a11)/(a22 - a12*a21/a11) - a13*a41/a11)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41);
  ainv[3][3] = (a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31)/(a11*a22*a33*a44 - a11*a22*a34*a43 - a11*a23*a32*a44 + a11*a23*a34*a42 + a11*a24*a32*a43 - a11*a24*a33*a42 - a12*a21*a33*a44 + a12*a21*a34*a43 + a12*a23*a31*a44 - a12*a23*a34*a41 - a12*a24*a31*a43 + a12*a24*a33*a41 + a13*a21*a32*a44 - a13*a21*a34*a42 - a13*a22*a31*a44 + a13*a22*a34*a41 + a13*a24*a31*a42 - a13*a24*a32*a41 - a14*a21*a32*a43 + a14*a21*a33*a42 + a14*a22*a31*a43 - a14*a22*a33*a41 - a14*a23*a31*a42 + a14*a23*a32*a41);

  return;
  
}


__device__ void  get_eigvals_4d(pycuda::complex<double> a,pycuda::complex<double> b,pycuda::complex<double> c,
                             pycuda::complex<double> d,pycuda::complex<double> e,pycuda::complex<double> x[4])
{
  // Returns roots of a 4th order polynomial, as in https://en.wikipedia.org/wiki/Quartic_function
  // Equation as : a*x**4 + b*x**3 + c*x**2 + d*x + e = 0
  // TODO : implement exceptions 
  pycuda::complex<double> p,q,D0,D1,Q,S,DISC;
  p = (8.0*a*c - 3.0*b*b)/(8.0*a*a);
  q = (b*b*b - 4.0*a*b*c + 8.0*a*a*d)/(8.0*a*a*a);
  D0 = c*c - 3.0*b*d + 12.0*a*e;
  D1 = 2.0*c*c*c - 9.0*b*c*d + 27.0*b*b*e + 27.0*a*d*d - 72.0*a*c*e;
  Q = pycuda::pow(0.5*(D1 + sqrt(pycuda::pow(D1,2) - 4.0*pycuda::pow(D0,3))),1./3);
  S = 0.5*pycuda::sqrt((-2.0*p/3.0) + (Q + D0/Q)/(3.0*a));
  DISC = D1*D1 - 4.0*D0*D0*D0;

  pycuda::complex<double> x1,x2,x3,x4;
  x1 = -(b/(4.0*a)) - S + 0.5*pycuda::sqrt(-4.0*pycuda::pow(S,2) -2.0*p + q/S);
  x2 = -(b/(4.0*a)) - S - 0.5*pycuda::sqrt(-4.0*pycuda::pow(S,2) -2.0*p + q/S);
  x3 = -(b/(4.0*a)) + S + 0.5*pycuda::sqrt(-4.0*pycuda::pow(S,2) -2.0*p -q/S);
  x4 = -(b/(4.0*a)) + S - 0.5*pycuda::sqrt(-4.0*pycuda::pow(S,2) -2.0*p -q/S);
  
  x[0] = x1;
  x[1] = x2;
  x[2] = x3;
  x[3] = x4;

  return;
}

__device__ void  get_eigvals_2d(pycuda::complex<double> a,pycuda::complex<double> b,pycuda::complex<double> c,
                                pycuda::complex<double> x[2])
{
  // Returns roots of a 2th order polynomial (sqrt because it is the chargino case)
  // Equation as : a*x**2 + b*x + c = 0
  // TODO : implement exceptions
  x[0] = pycuda::sqrt((-b + pycuda::sqrt(b*b - 4.0*a*c))/(2.0*a));
  x[1] = pycuda::sqrt((-b - pycuda::sqrt(b*b - 4.0*a*c))/(2.0*a));
  return;
}



__device__ void get_eigvects_4d(pycuda::complex<double> A[4][4],
                             pycuda::complex<double> bk[4],pycuda::complex<double> lamda)
{
  // Returns the eigenvectors for a given matrix, A (4x4), and the corresponding eigenvalue
  pycuda::complex<double> Ck[4],modCk,b0[4];
  A[0][0] +=  - lamda;
  A[1][1] +=  - lamda;
  A[2][2] +=  - lamda;
  A[3][3] +=  - lamda;

  pycuda::complex<double> Ainv[4][4];
  get_inv_4d(A,Ainv);
  int iterations = 10000.0; // ATTENTION: should be enough iterations to get the proper eigenvector
  int ie;
  for(ie=0;ie<4;ie++){ //initialize b0
    b0[ie] = 0.0001;
  }
  for(ie=1;ie<=iterations;ie++){
    Ck[0] = Ainv[0][0]*b0[0] + Ainv[0][1]*b0[1] + Ainv[0][2]*b0[2] + Ainv[0][3]*b0[3];
    Ck[1] = Ainv[1][0]*b0[0] + Ainv[1][1]*b0[1] + Ainv[1][2]*b0[2] + Ainv[1][3]*b0[3];
    Ck[2] = Ainv[2][0]*b0[0] + Ainv[2][1]*b0[1] + Ainv[2][2]*b0[2] + Ainv[2][3]*b0[3];
    Ck[3] = Ainv[3][0]*b0[0] + Ainv[3][1]*b0[1] + Ainv[3][2]*b0[2] + Ainv[3][3]*b0[3];
    
    modCk = sqrt(Ck[0]*Ck[0] + Ck[1]*Ck[1] + Ck[2]*Ck[2] + Ck[3]*Ck[3]);
    bk[0] = Ck[0]/modCk;
    bk[1] = Ck[1]/modCk;
    bk[2] = Ck[2]/modCk;
    bk[3] = Ck[3]/modCk;

    b0[0] = bk[0];
    b0[1] = bk[1];
    b0[2] = bk[2];
    b0[3] = bk[3];
  }

  if(pycuda::real(bk[0]) < 0 && pycuda::real(bk[1]) < 0 && pycuda::real(bk[2]) < 0 && pycuda::real(bk[3]) < 0){ 
    // revert sign if all of them are negative 
    bk[0] = -1.0*bk[0];
    bk[1] = -1.0*bk[1];
    bk[2] = -1.0*bk[2];
    bk[3] = -1.0*bk[3];
  }

  return;
  
}

__device__ void get_eigvects_2d(pycuda::complex<double> A[2][2],
                             pycuda::complex<double> bk[2],pycuda::complex<double> lamda)
{
  // Returns the eigenvectors for a given (2x2) matrix, A, and the corresponding eigenvalue 
  pycuda::complex<double> mod;
  A[0][0] +=  - lamda;
  A[1][1] +=  - lamda;

  if(pycuda::abs(A[0][1]) != 0){
    mod = pycuda::sqrt(1.0 + (A[0][0]/A[0][1])*(A[0][0]/A[0][1]));
    bk[0] = 1.0/mod;
    bk[1] = -A[0][0]/(A[0][1]*mod);
  }
  else {
    mod = pycuda::sqrt(1.0 + (A[1][0]/A[1][1])*(A[1][0]/A[1][1]));
    bk[0] = 1.0/mod;
    bk[1] = -A[1][0]/(A[1][1]*mod);    
  }
  return;
  
}

__device__ int get_min(int length, pycuda::complex<double> x[4], pycuda::complex<double> y[4])
{
  double min_val = 10000000000.0;
  int ie,je,position;
  position = 500;
  for(ie=0;ie < length;ie ++){
    if(pycuda::abs(x[ie]) <= min_val){
      min_val = pycuda::abs(x[ie]);
      position = ie;
    }
  }
  je = 0;

  for(ie=0;ie < length;ie++){
    if(ie != position){
      y[je] = 1.0*x[ie];
      je++;
    }
  }
  return position;
}

__device__ void sort_list(pycuda::complex<double> x[4],pycuda::complex<double> y[4])
{
  // Returns x sorted in increasing absolute values => y
  int ie, ie1,ie2,ie3,ie4;
  double min_val = 1e9;
  double max_val = 0.;
  
 
  for(ie=0;ie < 4;ie ++){
    if(pycuda::abs(x[ie]) <= min_val){
      min_val = pycuda::abs(x[ie]);
      ie1 = ie;
    }
    if(pycuda::abs(x[ie]) >= max_val){
      max_val = pycuda::abs(x[ie]);
      ie4 = ie;
    }
  }
  
  min_val = 1e9;
  max_val = 0;
  for(ie=0;ie < 4;ie ++){
    if (ie == ie1) continue;
    if (ie == ie4) continue;
    
    if(pycuda::abs(x[ie]) <= min_val){
      min_val = pycuda::abs(x[ie]);
      ie2 = ie;
    }
    if(pycuda::abs(x[ie]) >= max_val){
      max_val = pycuda::abs(x[ie]);
      ie3 = ie;
    }
  }
  y[0] = x[ie1];
  y[1] = x[ie2];
  y[2] = x[ie3];
  y[3] = x[ie4];
    
/*
  ie1 = get_min(4,x,p); // gets the minimum between (x1,x2,x3,x4), leaves the unwanted (p1,p2,p3)
  y[0] = 1.*x[ie1];
  ie2 = get_min(3,p,q); // gets the minimum between (p1,p2,p3), leaves the unwanted (q1,q2)
  y[1] = 1.*p[ie2];
  ie3 = get_min(2,q,r); // gets the minimum between (q1,q2), leaves the unwanted r1
  y[2] = 1.*q[ie3];
  y[3] = 1.*r[0];
  
  
  y[0] = x[0];
  y[1] = x[1];
  y[2] = x[2];
  y[3] = x[3];
  */
  
  return;
}

__device__ void neutralino_guts(double M1,double M2,double mu,double b,
                                pycuda::complex<double> XMN[4])
{
  // Returns eigenvalues (sorted in increasing absolute value) + matrix of eigenvectors
  pycuda::complex<double> MN[4][4];
  
  pycuda::complex<double> coeff[5],x[4];
  get_coefficients_neutralino(M1,M2,mu,b,coeff,MN); //get coefficients to get eigenvects, and eigenvects
  get_eigvals_4d(coeff[4],coeff[3],coeff[2],coeff[1],coeff[0],x);
  // sort the eigenvals 
  //pycuda::complex<double> bk1[4],bk2[4],bk3[4],bk4[4];
 
  sort_list(x,XMN);
  /*
  // Get the eigenvectors:
  get_eigvects_4d(MN,bk1,XMN[0]);
  get_eigvects_4d(MN,bk2,XMN[1]);
  get_eigvects_4d(MN,bk3,XMN[2]);
  get_eigvects_4d(MN,bk4,XMN[3]);
  // Fill the matrix:
  int ie;
  for(ie=0;ie<4;ie++){
    MN[ie][0] = bk1[ie];
    MN[ie][1] = bk2[ie];
    MN[ie][2] = bk3[ie];
    MN[ie][3] = bk4[ie];
  }
  */
  return;
}

__device__ void chargino_guts(double M2,double mu,double b
                             ,pycuda::complex<double> XMch[2])
{
  // Returns eigenvalues (sorted in increasing absolute value) + matrix of eigenvectors
  //pycuda::complex<double> U[2][2], V[2][2];
  
  pycuda::complex<double> coeff[3],x[2];
  
  get_coefficients_chargino(M2,mu,b,coeff); //get coefficients to get eigenvects, and eigenvects
  get_eigvals_2d(coeff[2],coeff[1],coeff[0],x);
  //int ie;
  // sort the eigenvals
  if(pycuda::abs(x[0]) < pycuda::abs(x[1]))
  {
    XMch[0] = x[0];
    XMch[1] = x[1];
  }
  else
  {
    XMch[0] = x[1];
    XMch[1] = x[0];
  }
  /*
  // Get the eigenvectors:
  // V(MMDag)VDag
  pycuda::complex<double> Vbk1[2],Vbk2[2];
  get_eigvects_2d(MchdagMch,Vbk1,XMch[0]*XMch[0]);
  get_eigvects_2d(MchdagMch,Vbk2,XMch[0]*XMch[0]);
  // Fill the matrix:
  for(ie=0;ie<4;ie++){
    V[0][ie] = pycuda::conj(Vbk1[ie]);
    V[1][ie] = pycuda::conj(Vbk2[ie]);
  }
  // U*(MMDag)U
  pycuda::complex<double> Ubk1[2],Ubk2[2];
  get_eigvects_2d(MchdagMch,Ubk1,XMch[0]*XMch[0]);
  get_eigvects_2d(MchdagMch,Ubk2,XMch[0]*XMch[0]);
  // Fill the matrix:
  for(ie=0;ie<4;ie++){
    U[0][ie] = Ubk1[ie];
    U[1][ie] = Ubk2[ie];
  }
  */
  return;
}


//RR -> Charginos Matrix, for the moment separate function (as i think previous
// one is buggy).
//TODO: Check everything works.
//TODO: Merge functions. 
__device__ void chargino_matrix(double M2, double mu, double b
                             ,pycuda::complex<double> XMch[2]
                             ,pycuda::complex<double> U[2][2] 
                             ,pycuda::complex<double> V[2][2])
{
  //This function returns eigenvalues (ordered)
  //U Matrix
  //V matrix

  // pycuda::complex<double> U[2][2], V[2][2];
  pycuda::complex<double> Mch[2][2], Mchdag[2][2], MchMchdag[2][2], MchdagMch[2][2];
  
  Mch[0][0] = M2;
  Mch[0][1] = sqrt(2.)*Mw*sin(b);
  Mch[1][0] = sqrt(2.)*Mw*cos(b);
  Mch[1][1] = mu;

  // for(int i=0; i<2; i++){
    // for(int j=0; j<2; j++){
      // printf("Mch[%d][%d]=%.4f\n, division=%4.f", i, j, Mch[i][j], pycuda::real(Mch[j][i]/Mch[i][j]));
    // }
  // }
  Dagger(Mch,Mchdag);
  MatrixProduct(Mch,Mchdag,MchMchdag);
  MatrixProduct(Mchdag,Mch,MchdagMch); //WR
  
  pycuda::complex<double> coeff[3],x[2];
  
  get_coefficients_chargino(M2,mu,b,coeff); //get coefficients to get eigenvects, and eigenvects
  get_eigvals_2d(coeff[2],coeff[1],coeff[0],x);
  int ie;
  // sort the eigenvals
  if(pycuda::abs(x[0]) < pycuda::abs(x[1]))
  {
    XMch[0] = x[0];
    XMch[1] = x[1];
  }
  else
  {
    XMch[0] = x[1];
    XMch[1] = x[0];
  }
  // Get the eigenvectors:
  // V(MDagM)V**-1
  pycuda::complex<double> Vbk1[2],Vbk2[2];
  get_eigvects_2d(MchdagMch,Vbk1,XMch[0]*XMch[0]);
  get_eigvects_2d(MchdagMch,Vbk2,XMch[1]*XMch[1]);
  // Fill the matrix:
  //TODO: Check conjugates (?)
  for(ie=0;ie<2;ie++){
    // WARNING: I have doubts about this conjugation
    V[0][ie] = pycuda::conj(Vbk1[ie]);
    V[1][ie] = pycuda::conj(Vbk2[ie]);
  }
  // U*(MMDag)(U*)**-1
  pycuda::complex<double> Ubk1[2],Ubk2[2];
  get_eigvects_2d(MchMchdag,Ubk1,XMch[0]*XMch[0]);
  get_eigvects_2d(MchMchdag,Ubk2,XMch[1]*XMch[1]);
  // Fill the matrix:
  for(ie=0;ie<2;ie++){
    U[0][ie] = Ubk1[ie];
    U[1][ie] = Ubk2[ie];
  }
  // for(int i=0; i<2; i++){
    // for(int j=0; j<2; j++){
      // printf("real U [%d][%d]=%.4f\n", i, j, pycuda::real(U[i][j]));
      // printf("imag U [%d][%d]=%.4f\n", i, j, pycuda::imag(U[i][j]));
      // printf("real V [%d][%d]=%.4f\n", i, j, pycuda::real(V[i][j]));
      // printf("imag V [%d][%d]=%.4f\n", i, j, pycuda::imag(V[i][j]));
    // }
  // }
  return;
}




__device__ void stop_guts(double At,double mtL2,double mtR2,double b,double mu,
                              pycuda::complex<double> XMstop[2])
{
  // Returns eigenvalues (sorted in increasing absolute value)   
  pycuda::complex<double> coeff[3],x[2];
  get_coefficients_stop(At,mtL2,mtR2,b,mu,coeff); //get coefficients to get eigenvects, and eigenvects
  get_eigvals_2d(coeff[2],coeff[1],coeff[0],x);
  //printf("coeff2,coeff1,coeff0 %.5e,%.5e,%.5e\n",pycuda::real(coeff[2]),pycuda::real(coeff[1]),pycuda::real(coeff[0]));
  // sort the eigenvals
  if(pycuda::abs(x[0]) < pycuda::abs(x[1]))
  {
    XMstop[0] = x[0];
    XMstop[1] = x[1];
  }
  else
  {
    XMstop[0] = x[1];
    XMstop[1] = x[0];
  }
  //printf("XMstop[0] = %.3e + %.3e i \n",pycuda::real(XMstop[0]),pycuda::imag(XMstop[0]));
  //printf("XMstop[1] = %.3e + %.3e i \n",pycuda::real(XMstop[1]),pycuda::imag(XMstop[1]));
  return;
}
