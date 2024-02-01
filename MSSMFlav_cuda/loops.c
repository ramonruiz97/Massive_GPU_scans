// Diego Martinez Santos, Miriam Lucio, Isabel Suarez, Ramón Ángel Ruiz
// Fernández
// Based mostly on Teppei Kitahara's notes, plus 0909.1333
/*-------------------------------------------------------------------*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pycuda-complex.hpp>
#define loop_w 1e-5
#define loop_wg 0.005
#define EPS_PREC 1E-10
#define EPS 1E-10
// #define M_GAMMAl 0.5772156649015328606065120900824024L
#define M_EULER  0.57721566490153286060651209008 
#define MAXNUM 1.79769313486231570815E308
#define ETHRESH 1.0e-12
#define MACHEP 1.38777878078144567553E-17
#define MU_0 120.0


//Delta kronecker
__device__ int Krodelta(int i,int j){ 
        if(i == j) return 1 ;
        else return 0;
}

__device__ double h3(double x)
{
	if(fabs(x-1.)<loop_w) return -1./4;
	return -0.5/(1.-x)-0.5*x*log(x)*pow(1.-x,-2.);
}

__device__ double h1(double x)
{
	if(fabs(x-1.)<loop_w) return 4./9;
	return 4.*(1.+x)/(3.*pow(1.-x,2.)) + 8.*x*log(x)/(3.*pow((1.-x),3.));
}

__device__ double h2(double x)
{
if(fabs(x-1.)<loop_wg) return -2./9.;
return -8./3.*x*log(x)/pow(-x + 1, 4) + (1./9.)*(4*pow(x, 2) - 20*x - 8)/pow(-x + 1, 3);
}


__device__ double h4(double x,double y){
	if((fabs(1.-x)<loop_w)&&(fabs(1.-y)<loop_w)) return 1./6;
	if(fabs(1.-x)<loop_w) return h4(0.9999,y);
	if(fabs(1.-y)<loop_w) return h4(x,0.9999);
	if(fabs(1.-x/y)<loop_w) return h4(y*0.9998,y);
	return -1./((1.-x)*(1.-y))+x*log(x)/((y-x)*pow((1.-x),2.))+y*log(y)/((x-y)*pow((1.-y),2.));
}

__device__ double g1(double x){
	if(fabs(1.-x)<loop_wg) return -1./216;
	double g1_1 = -(11.+144.*x+27.*pow(x,2.)-2*pow(x,3.))/(108.*pow(1.-x,4.));
	double g1_2 = x*(13.+17.*x)*log(1.*x)/(18.*pow(1.-x,5.));
	return g1_1 - g1_2; 
}

__device__ double g4(double x){
	if(fabs(1.-x)<loop_wg) return 23./180; // ML: widen mass window, otherwise we get crazy values for g4
	double g4_1 = (2.-99.*x-54.*pow(x,2.)+7.*pow(x,3.))/(18.*pow(1.-x,4.));
	double g4_2 = -x*(5.+19.*x)*log(x)/(3.*pow(1.-x,5.));
	return g4_1 + g4_2;
}

__device__ double g5(double x){
	if(fabs(1.-x)<loop_wg) return -7./540;
	double g5_1 = -(10.+117*x+18*pow(x,2)-pow(x,3))/(54*pow(1.-x,4));
	double g5_2 = -x*(11.+13*x)*log(x)/(9*pow(1.-x,5));
	return g5_1 + g5_2;
}

__device__ double f(double x)
{
	if(fabs(x-1.)<loop_w) return 0.5;
	return 1./(1.-x)+x*log(x)/pow(1.-x,2.);
}

__device__ double f1(double x)
{
	if(fabs(x-1.)<loop_w) return -1./12;
  return -1*(x+1.)*1./(4*pow((1.-x),2.)) - x*log(x)/(2*pow(1.-x,3.)); 
}

__device__ double f3(double x)
{
	if(fabs(x-1.)<loop_w) return 1./20;
	return (pow(x,2.) - 8*x - 17)*1./(6*pow((1.-x),4.)) - (3*x + 1)*log(x)/pow(1.-x,5.); 
}

/*--------------------------------eps_K'/eps_K functions--------------------------------------------*/
__device__ double B1(double x)
{
  double x2 = x*x;
  //double x3 = x2*x;
  double w = x-1;
  if (fabs(w) < loop_wg) return 1.0/48.0;
  return (1 + 4*x -5*x2 + 4*x*log(x) + 2*x2*log(x))/(8*w*w*w*w);
}
__device__ double B2(double x)
{
  double x2 = x*x;
  //double x3 = x2*x;
  double w = x-1;
  if (fabs(w) < loop_wg) return -1.0/12.0;

  return x*(5-4*x-x2+2*log(x)+4*x*log(x))/(2*w*w*w*w);
}

__device__ double P1(double x)
{
  double x2 = x*x;
  double x3 = x2*x;
  double w = x-1;
  if (fabs(w) < loop_wg) return -1./30.;
  
  return (1-6*x+18*x2 -10*x3 -3*x3*x +12*x3*log(x))/(18*(w*w*w*w*w));
}
__device__ double P2(double x)/// feel like a 5yo
{
  double x2 = x*x;
  double x3 = x2*x;
  double w = x-1;
  if (fabs(w) < loop_wg) return 1./30.;

  return (7-18*x + 9*x2 + 2*x3 + 3*log(x) -9*x2*log(x))/(9*(w*w*w*w*w));
}
__device__ double M1(double x)
{
  return 4*B1(x);
}
__device__ double M2(double x)
{
  //return -x*B2(x);
  return -B2(x)/x; // TK claims should be like this
}
__device__ double M3(double x)
{
  double x2 = x*x;
  double x3 = x2*x;
  double w = x-1;
  if (fabs(w) < loop_wg) return 1./40.;

  return (-1+9*x+9*x2-17*x3+18*x2*log(x)+6*x3*log(x))/(12*(w*w*w*w*w));
}
__device__ double M4(double x)
{
  double x2 = x*x;
  double x3 = x2*x;
  double w = x-1;
  if (fabs(w) < loop_wg) return 1./60.;

  return (-1-9*x+9*x2+x3-6*x*log(x)-6*x2*log(x))/(6*(w*w*w*w*w));
}


__device__ double tep_f(double x, double y)
{
  if (fabs(y-1) < loop_wg && fabs(x-1) < loop_wg) 
      return -1./12.;
  else if (fabs(y-1)<loop_wg) 
      return -0.5*x*(pow(x, 2) + 4.*x - (4.*x + 2)*log(x) - 5.)/pow(x - 1., 4);
  else if (fabs(x-1)<loop_wg) 
      return (1./6.)*(-pow(y, 3) + 6.*pow(y, 2) - 6.*y*log(y) - 3.*y - 2.)/pow(y - 1., 4);
  else if (fabs(x-y)<loop_wg) 
      return 0.5*(5.*pow(y, 2) - 2.*y*(y + 2)*log(y) - 4.*y - 1.)/pow(y - 1., 4);
    
  return -x*y*log(y)/(pow(x - y, 2)*pow(y - 1., 2)) + x*(x - 2.*y + 1.)/
    (pow(x - 1., 2)*(x - y)*(y - 1.)) + x*(2.*pow(x, 2) - y*(x + 1.))*log(x)/(pow(x - 1., 3)*pow(x - y, 2));
}

__device__ double tep_g(double x, double y)
{
  if (fabs(y-1) < loop_wg && fabs(x-1) < loop_wg) 
      return -1./12.;
  else if (fabs(y-1)<loop_wg) 
      return -0.5*x*(-5.*pow(x, 2) + 2.*x*(x + 2)*log(x) + 4.*x + 1.)/pow(x - 1., 4);
  else if (fabs(x-1)<loop_wg) 
      return (1./6.)*(-2.*pow(y, 3) + 6.*pow(y, 2)*log(y) - 3.*pow(y, 2) + 6.*y - 1.)/pow(y - 1., 4);
  else if (fabs(x-y)<loop_wg) 
      return -0.5*y*(pow(y, 2) + 4.*y - (4.*y + 2.)*log(y) - 5.)/pow(y - 1., 4);
    
  return -pow(x, 2)*(x*(x + 1.) - 2.*y)*log(x)/(pow(x - 1., 3)*pow(x - y, 2)) 
         + x*pow(y, 2)*log(y)/(pow(x - y, 2)*pow(y - 1., 2)) 
         + x*(-2.*x + y*(x + 1.))/(pow(x - 1., 2)*(x - y)*(y - 1.));
}

__device__ double tep_l(double x, double y)
{
  if (fabs(y-1) < loop_wg && fabs(x-1) < loop_wg) 
      return -1./12.;
  else if (fabs(y-1)<loop_wg) 
      return 0.5*(5.*pow(x, 2) - 2.*x*(x + 2.)*log(x) - 4.*x - 1)/pow(x - 1., 4);
  else if (fabs(x-1)<loop_wg) 
      return 0.5*(5.*pow(y, 2) - 2.*y*(y + 2.)*log(y) - 4.*y - 1)/pow(y - 1., 4);
  else if (fabs(x-y)<loop_wg) 
      return (1./6.)*(-pow(y, 3) + 6.*pow(y, 2) - 6.*y*log(y) - 3.*y - 2.)/(y*pow(y - 1., 4));
    
  return x*(-pow(x, 2) - y*(x - 2.))*log(x)/(pow(x - 1., 2)*pow(x - y, 3)) 
         + y*(x*(y - 2.) + pow(y, 2))*log(y)/(pow(x - y, 3)*pow(y - 1., 2)) 
         - (-2*x*y + x + y)/((x - 1)*pow(x - y, 2)*(y - 1.));
}

__device__ double tep_I(double x, double y)
{
  if (fabs(y-1) < loop_wg && fabs(x-1) < loop_wg) 
      return 0.05;
  else if (fabs(y-1)<loop_wg) 
      return (1./6.)*(pow(x, 3) - 9.*pow(x, 2) - 9.*x + (18.*x + 6.)*log(x) + 17.)/pow(x - 1., 5);
  else if (fabs(x-1)<loop_wg) 
      return (1./12.)*(pow(y, 4) - 6.*pow(y, 3) + 18.*pow(y, 2) - 12.*y*log(y) - 10.*y - 3.)/pow(y - 1., 5);
  else if (fabs(x-y)<loop_wg) 
      return 0.5*(pow(y, 3) + 9.*pow(y, 2) - 6.*y*(y + 1.)*log(y) - 9.*y - 1.)/(y*pow(y - 1., 5));

  return -y*log(y)/(pow(x - y, 2)*pow(y - 1., 3)) 
         + (x*y*(x + 2.) + x*(x - 5.) - pow(y, 2)*(x + 5.) + 9.*y - 2.)/(pow(x - 1., 3)*(2.*x - 2.*y)*pow(y - 1., 2)) 
         + (3.*pow(x, 2) - 2.*x*y - y)*log(x)/(pow(x - 1., 4)*pow(x - y, 2));
}

__device__ double tep_J(double x, double y)
{
  if (fabs(y-1) < loop_wg && fabs(x-1) < loop_wg) 
      return 1./30.;
  else if (fabs(y-1)<loop_wg) 
      return (1./3.)*(pow(x, 3) + 9.*pow(x, 2) - 6.*x*(x + 1)*log(x) - 9.*x - 1)/pow(x - 1., 5);
  else if (fabs(x-1)<loop_wg) 
      return (1./12.)*(pow(y, 4) - 8.*pow(y, 3) + 12.*pow(y, 2)*log(y) + 8.*y - 1)/pow(y - 1., 5);
  else if (fabs(x-y)<loop_wg) 
      return (-3.*pow(y, 2) + (pow(y, 2) + 4.*y + 1)*log(y) + 3.)/pow(y - 1., 5);
    
  return -x*(x*(2*x + 1.) - y*(x + 2.))*log(x)/(pow(x - 1., 4)*pow(x - y, 2)) + 
         pow(y, 2)*log(y)/(pow(x - y, 2)*pow(y - 1., 3)) + (x*(x + 5.) + pow(y, 2)*(5.*x + 1.) 
         - 3.*y*pow(x + 1., 2))/(pow(x - 1., 2)*(x - y)*(2.*x - 2.)*pow(y - 1., 2));
}

__device__ double tep_K(double x, double y)
{
  if (fabs(y-1) < loop_wg && fabs(x-1) < loop_wg) 
      return -1./12.;
  else if (fabs(y-1)<loop_wg) 
      return -0.5*x*(-5.*pow(x, 2) + 2.*x*(x + 2.)*log(x) + 4.*x + 1.)/pow(x - 1., 4);
  else if (fabs(x-1)<loop_wg) 
      return (1./6.)*(-2.*pow(y, 3) + 6.*pow(y, 2)*log(y) - 3.*pow(y, 2) + 6.*y - 1.)/pow(y - 1., 4);
  else if (fabs(x-y)<loop_wg) 
      return -0.5*y*(pow(y, 2) + 4.*y - (4.*y + 2)*log(y) - 5.)/pow(y - 1., 4);
      
  return -pow(x, 2)*(x*(x + 1.) - 2.*y)*log(x)/(pow(x - 1., 3)*pow(x - y, 2)) 
         + x*pow(y, 2)*log(y)/(pow(x - y, 2)*pow(y - 1., 2)) 
         + x*(-2.*x + y*(x + 1.))/(pow(x - 1., 2)*(x - y)*(y - 1.));
}

__device__ double tep_gw_1(double x)
{

  if(fabs(x-1.)<loop_wg) return -1./60.;
  return -x*(4*x + 3)*log(x)/pow(-x + 1, 5) + (1./12.)*(pow(x, 3) - 13*pow(x, 2) - 67*x - 5)/pow(-x + 1, 4);
}

__device__ double tep_gw_2(double x)
{

if(fabs(x-1.)<loop_wg) return 1./120.;

return 0.5*x*(23*x + 12)*log(x)/pow(-x + 1, 6) 
  + (1./24.)*(2*pow(x,4) - 19*pow(x, 3) + 113*pow(x, 2) + 309*x + 15)/pow(-x + 1, 5);
}


__device__ double tep_gw_3(double x)
{

  if(fabs(x-1.)<loop_wg) return 0.;
  return -3.0/2.0*x*(5*x + 2)*log(x)/pow(-x + 1, 7) + (1./40.)*(pow(x, 5) -9*pow(x, 4) 
                                                                  + 41*pow(x, 3) - 159*pow(x, 2) - 284*x - 10)/pow(-x + 1, 6);
}


/*--------------------------------------- EDM functions --------------------------------------------*/

__device__ double fhwc(double x, double y)
{
  if(fabs(x-1.)<loop_w && fabs(y-1.)<loop_w) return +13./45.;
  double g_0;
  double g_1;
  if (fabs(x-1) < loop_wg){
    g_0 = (-1./6. + (1 + 4*y - 5*pow(y,2) + 2*y*(2 + y)*log(y))/pow(-1 + y,4))/(1 - y);
    g_1 = -(31 - 28*y - 4*pow(y,3) + pow(y,4) + 12*(1 + 2*y)*log(y))/(6.*pow(-1 + y,5));
        }
  else if (fabs(y-1) < loop_wg) {
    g_0 = (-1./6. + (-1 - 4*x + 5*pow(x,2) - 2*x*(2 + x)*log(x))/pow(-1 + x,4))/(-1 + x);
    g_1 = (-1./6. + (-5 + 4*x + pow(x,2) - 2*(1 + 2*x)*log(x))/pow(-1 + x,4))/(-1 + x);
    return 2/3*g_0 - g_1;
  }
  else if(fabs(x-y)<loop_w){
    g_0 = 4*(3-3*pow(y,2)+(1+4*y+pow(y,2))*log(y))/pow((-1+y),5);
    g_1 = (-2*(-1+y)*(1+y*(10+y)) + 12*y*(1+y)*log(y))/(y*pow(-1+y, 5));
    return 2/3*g_0 - g_1;
  }
  else{
    double f0_x = -(1+4*x-5*pow(x,2)+2*x*(2+x)*log(x))/pow((1-x),4);
    double f0_y = -(1+4*y-5*pow(y,2)+2*y*(2+y)*log(y))/pow((1-y),4);
    g_0 = (f0_x - f0_y)/(x-y);
    double f1_x = - (5-4*x-pow(x,2) + 2*(1+2*x)*log(x))/pow((1-x),4);
    double f1_y = - (5-4*y-pow(y,2) + 2*(1+2*y)*log(y))/pow((1-y),4);
    g_1 = (f1_x - f1_y)/(x-y);
  }
  return 2/3*g_0 - g_1;
}

__device__ double fhwc_2(double x, double y)
{
  if(fabs(x-1.)<loop_w && fabs(y-1.)<loop_w) return +2./15.;
  double g_0;
  if (fabs(x-1) < loop_wg){
    g_0 = (-1./6. + (1 + 4*y - 5*pow(y,2) + 2*y*(2 + y)*log(y))/pow(-1 + y,4))/(1 - y);
        }
  else if (fabs(y-1) < loop_wg) {
    g_0 = (-1./6. + (-1 - 4*x + 5*pow(x,2) - 2*x*(2 + x)*log(x))/pow(-1 + x,4))/(-1 + x);
  }
  else if(fabs(x-y)<loop_w){
    g_0 = 4*(3-3*pow(y,2)+(1+4*y+pow(y,2))*log(y))/pow((-1+y),5);
  }
  else{
    double f0_x = -(1+4*x-5*pow(x,2)+2*x*(2+x)*log(x))/pow((1-x),4);
    double f0_y = -(1+4*y-5*pow(y,2)+2*y*(2+y)*log(y))/pow((1-y),4);
    g_0 = (f0_x - f0_y)/(x-y);
  }
  return g_0;
}

__device__ double fchar(double x, double y ) // (D.26)
{
  if(fabs(x-1.)<loop_w && fabs(y-1.)<loop_w) return -5./18.;
  double E_f3phi;
  double E_ffphiphi;
  if (fabs(x-1.) < loop_wg){
      E_f3phi = -(2. + 3*y - 6*pow(y,2) + pow(y,3) + 6*y*log(y))/(3.*pow(-1 + y,4));
      E_ffphiphi = (1. - 6*y + 3*pow(y,2) + 2*pow(y,3) - 6*pow(y,2)*log(y))/(3*pow(y - 1.,4));
        }
  else if (fabs(y-1.) < loop_wg) {
      E_f3phi = -(2. + 3*x - 6*pow(x,2) + pow(x,3) + 6*x*log(x))/(3.*pow(-1 + x,4));
      E_ffphiphi = (1. - 6*x + 3*pow(x,2) + 2*pow(x,3) - 6*pow(x,2)*log(x))/(3.*pow(-1 + x,4));
  }
  else if(fabs(x-y)<loop_w){
    E_f3phi = (5.-4.*x-pow(x,2)+2.*(1.+2.*x)*log(x))/pow(1.-x, 4);
    E_ffphiphi = (1.+4.*x-5.*pow(x,2)+2.*x*(2+x)*log(x))/pow(1.-x, 4);
  }
  else{
    double E_f3phi_1 = (3.-x-y-x*y)/(pow(1.-x, 2)*pow(1.-y,2));
    double E_f3phi_2 = (2.*x*log(x))/((x-y)*pow(1-x,3));
    double E_f3phi_3 = (2.*y*log(y))/((y-x)*pow(1-y,3));
    E_f3phi = E_f3phi_1 + E_f3phi_2 + E_f3phi_3;

    double E_ffphiphi_1 = (1.+x+y-3.*x*y)/(pow(1.-x, 2)*pow(1.-y,2));
    double E_ffphiphi_2 = (2.*pow(x,2)*log(x))/((x-y)*pow(1-x,3));
    double E_ffphiphi_3 = (2.*pow(y,2)*log(y))/((y-x)*pow(1-y,3));
    E_ffphiphi = E_ffphiphi_1 + E_ffphiphi_2 + E_ffphiphi_3;
  }
  return 2.*E_f3phi/3. - E_ffphiphi;

}

__device__ double fchar2(double x, double y ) // (D.26)
{
  double E_f3phi;
  if(fabs(x-1.)<loop_w && fabs(y-1.)<loop_w) return -1./6.;
  if (fabs(x-1.) < loop_wg){
      E_f3phi = -(2 + 3*y - 6*pow(y,2) + pow(y,3) + 6*y*log(y))/(3.*pow(-1 + y,4));
        }
  else if (fabs(y-1.) < loop_wg) {
      E_f3phi = -(2 + 3*x - 6*pow(x,2) + pow(x,3) + 6*x*log(x))/(3.*pow(-1 + x,4));
  }
  else if(fabs(x-y)<loop_w){
    E_f3phi = (5.-4.*x-pow(x,2)+2.*(1.+2.*x)*log(x))/pow(1.-x, 4);
  }
  else{
    double E_f3phi_1 = (3.-x-y-x*y)/(pow(1.-x, 2)*pow(1.-y,2));
    double E_f3phi_2 = (2.*x*log(x))/((x-y)*pow(1-x,3));
    double E_f3phi_3 = (2.*y*log(y))/((y-x)*pow(1-y,3));
    E_f3phi = E_f3phi_1 + E_f3phi_2 + E_f3phi_3;
  }
  return E_f3phi;
}

__device__ double fsg(double x, double y)
{
  if(fabs(x-1.)<loop_w && fabs(y-1.)<loop_w) return -2./45.;
  double f_g0;
  if (fabs(x-y)<loop_w){
    f_g0 = (1.+9.*x-9.*pow(x,2)-pow(x,3)+6.*x*(1.+x)*log(x))/(pow(1.-x,5));
  }
  else{
    f_g0 = 2.*(3.*pow(x,2) - y - 2.*x*y)*log(x)/(pow((x-y), 2) * pow(1.-x,4));
    f_g0 += 2*y*log(y)/(pow((x-y),2)*pow((1-y),3));
    f_g0 += (2.-9*y+5*pow(y,2)-pow(x,2)*(1.+y)+x*(5.-2*y+pow(y,2)))/((x-y)*pow(1-x,3)*pow(1-y,2));
    f_g0 *= x;
  }
  return -4.*f_g0/9.;
}

__device__ double fsg_2(double x, double y)
{
  if(fabs(x-1.)<loop_w && fabs(y-1.)<loop_w) return -7./60.;
  double f_g0;
  double f_g1;
  if (fabs(x-y)<loop_w){
    f_g0 = (1.+9.*x-9.*pow(x,2)-pow(x,3)+6.*x*(1.+x)*log(x))/(pow(1.-x,5));
    f_g1 = x*(6-6*pow(x,2)+2*(1+4*x+pow(x,2)))*log(x)/(pow(1-x,5));
  }
  else{
    f_g0 = (2.*(3.*pow(x,2) - y - 2.*x*y)*log(x))/(pow((x-y), 2) * pow(1.-x,4));
    f_g0 += 2*y*log(y)/(pow((x-y),2)*pow((1-y),3));
    f_g0 += (2.-9*y+5*pow(y,2)-pow(x,2)*(1.+y)+x*(5.-2*y+pow(y,2)))/((x-y)*pow(1-x,3)*pow(1-y,2));
    f_g0 *= x;

    f_g1 = (2*x*(x+2*pow(x,2)-2*y-x*y)*log(x))/(pow(x-y,2)*pow(1-x,4));
    f_g1 += 2.*pow(y,2)*log(y)/(pow(x-y,2)*pow(1.-y, 3));
    f_g1 += ((-3+y)*y-pow(x,2)*(-1+3*y)-x*(-5.+6*y-5*pow(y,2)))/((x-y)*pow(1-x,3)*pow(1-y,2));
    f_g1 *= x;

  }

  return -1./6.*f_g0 + 3./2.*f_g1;
}

__device__ double fsg_d(double x, double y ) // (A.29)
{
  if(fabs(x-1.)<loop_w && fabs(y-1.)<loop_w) return 4./135;
  double f0_3;
  if (fabs(x-1.) < loop_wg){
    f0_3 =  -(y*(-37. + 8*y + 36*pow(y,2) - 8*pow(y,3) + pow(y,4) - 12*(1 + 4*y)*log(y)))/(6.*pow(-1 + y,6));
  }
  else if (fabs(y-1.) < loop_wg) {
    f0_3 =  -(x*(-37 + 8*x + 36*pow(x,2) - 8*pow(x,3) + pow(x,4) - 12*(1 + 4*x)*log(x)))/(6.*pow(-1 + x,6));
  }

  else if(fabs(x-y)<loop_w){
    f0_3 = (-1.+12*x+36*x*x-44*pow(x,3)-3*pow(x,4)+12*pow(x,2)*(3+2*x)*log(x))/(3*pow(1.-x,6));
  }
  else{
    double f0_3_1 = -2*(x+y+2*x*y-4*pow(x,2))*log(x)/(pow(x-y,3)*pow(1.-x,4));
    double f0_3_2 = 2*(x+y+2*x*y-4*pow(y,2))*log(y)/(pow(x-y,3)*pow(1.-y,4));
    double f0_3_3 = (4-6*x+13*pow(x,2)-5*pow(x,3)-6*y-14*x*y+3*x*x*y-pow(x,3)*y+13*pow(y,2)+3*x*pow(y,2)+2*x*x*y*y-5*pow(y,3)-x*pow(y,3))/(pow(x-y,2)*pow(1.-x,3)*pow(1.-y,3));
    f0_3 = x*y*(f0_3_1+f0_3_2+f0_3_3);
    return -4.0*f0_3/9.;
  }
  return -4.0*f0_3/9.;
}

__device__ double fsg_d2(double x, double y)  // (A.29) second term 
{

  if(fabs(x-1.)<loop_w && fabs(y-1)<loop_w) return 11./180.;
  double f0_3,f1_3;
  if (fabs(x-1.) < loop_wg){
            f0_3 =  -(y*(-37. + 8*y + 36*pow(y,2) - 8*pow(y,3) + pow(y,4) - 12*(1 + 4*y)*log(y)))/(6.*pow(-1 + y,6));
            f1_3 = (y*(3 + 44*y - 36*pow(y,2) - 12*pow(y,3) + pow(y,4) + 12*y*(2 + 3*y)*log(y)))/(6.*pow(-1 + y,6));
        }
  else if (fabs(y-1.) < loop_wg) {
    f0_3 =  -(x*(-37 + 8*x + 36*pow(x,2) - 8*pow(x,3) + pow(x,4) - 12*(1 + 4*x)*log(x)))/(6.*pow(-1 + x,6));
    f1_3 = (x*(3 + 44*x - 36*pow(x,2) - 12*pow(x,3) + pow(x,4) + 12*x*(2 + 3*x)*log(x)))/(6.*pow(-1 + x,6));
  }
  else if(fabs(x-y)<loop_w){
    f0_3 = (-1.+12.*x+36.*x*x-44.*pow(x,3)-3.*pow(x,4)+12.*pow(x,2)*(3.+2.*x)*log(x))/(3*pow(1.-x,6));
    f1_3 = 2*x*(1. + 18.*x - 9*pow(x,2) - 10*pow(x,3) + 3*x*log(x)*(3 + 6*x + pow(x,2)))/(3*pow(1.-x,6));
  }
  else{
    double f0_3_1 = -2*(x+y+2*x*y-4*pow(x,2))*log(x)/(pow(x-y,3)*pow(1.-x,4));
    double f0_3_2 = 2*(x+y+2*x*y-4*pow(y,2))*log(y)/(pow(x-y,3)*pow(1.-y,4));
    double f0_3_3 = (4-6*x+13*pow(x,2)-5*pow(x,3)-6*y-14*x*y+3*x*x*y-pow(x,3)*y+13*pow(y,2)+3*x*pow(y,2)+2*x*x*y*y-5*pow(y,3)-x*pow(y,3))/(pow(x-y,2)*pow(1.-x,3)*pow(1.-y,3));
    f0_3 = x*y*(f0_3_1+f0_3_2+f0_3_3);
    double f1_3_1 = 2*x*log(x)*(3*x*x-2*y-x*y)/(pow(x-y,3)*pow(1.- x,4));
    double f1_3_2 = -2*y*(3*y*y-2*x-x*y)*log(y)/(pow(x-y,3)*pow(1.-y,4));
    double f1_3_3 = (2*x+5*pow(x,2)-pow(x,3)+2*y-22*x*y+7*pow(x,2)*y-5*pow(x,3)*y+5*pow(y,2)+7*x*pow(y,2)+6*pow(x,2)*pow(y,2)-pow(y,3)-5*x*pow(y,3))/(pow(x-y,2)*pow(1.-x,3)*pow(1.-y,3));
    f1_3 = x*y*(f1_3_1+f1_3_2+f1_3_3);
  }
  return -f0_3/6. + 3.*f1_3/2.;
}




__device__ double fsg_u(double x) // (A.30)
{
  if(fabs(x-1.)<loop_w) return -8./135;
  double f0_3_num = -3 - 44*x + 36*pow(x,2) + 12*pow(x,3) - pow(x,4) - 12*x*(3*x + 2)*log(x);
  double f0_3_den = 3*pow(1.-x,6);
  double f0_3 = f0_3_num/f0_3_den;
  return 8.0*f0_3/9;
}

__device__ double fsg_u2(double x) // (A.30) second term
{
  if(fabs(x-1.)<loop_w) return 11./180;
  double f0_3_num = -3 - 44*x + 36*pow(x,2) + 12*pow(x,3) - pow(x,4) - 12*x*(3*x + 2)*log(x);
  double f0_3_den = 3*pow(1.-x,6);
  double f0_3 = f0_3_num/f0_3_den;
  double f1_3_num = 2*(-10 -9*x + 18*pow(x,2) + pow(x,3) -3*(1 + 6*x + 3*pow(x,2))*log(x));
  double f1_3_den = 3*pow(1.-x,6);
  double f1_3 = f1_3_num/f1_3_den;
  return -f0_3/6 + 3.*f1_3/2;
}


__device__ double fH(double x) // (A.31)
{
  if(fabs(x-1.)<loop_w) return -7./9;
  double f0_0 = (1. - pow(x,2) + 2*x*log(x))/(pow(1-x,3));
  double f1_0 = (3. - 4*x + pow(x,2) + 2*log(x))/(pow(1-x,3));
  return -f0_0 + 2*f1_0/3.0;
}

__device__ double fH2(double x) // (A.31) second term
{
  if(fabs(x-1.)<loop_w) return -2./3;
  double f1_0 = (3. - 4*x + pow(x,2) + 2*log(x))/(pow(1-x,3));
  return f1_0;
}



__device__ double fsH(double x) // (A.32)
{
  if(fabs(x-1.)<loop_w) return -5./18;
  double f0_1 = (-1.0 -4.0*x + 5.0*pow(x,2) - 2.0*x*(x + 2.0)*log(x))/(pow(1.0 - x,4));
  double f1_1 = (- 5.0 + 4.0*x + x*x + 2*.0*log(x))/(pow(1.0 -x,4));
  return 2.0*f0_1/3 - f1_1;
}

__device__ double fsH2(double x) // (A.32) second term
{
  if(fabs(x-1.)<loop_w) return -1./6;
  double f0_1 = (-1.0 -4.0*x + 5.0*pow(x,2) - 2.0*x*(x + 2.0)*log(x))/(pow(1.0 - x,4));
  return f0_1;
  
}

__device__ double gsH(double x) // (A.33)
{
  if(fabs(x-1.)<loop_w) return -2./15;
  double f0_2 = (1. + 9.*x - 9.*pow(x,2) - pow(x,3) + 6.*x*(x + 1.)*log(x))/(pow(1. - x,5));
  double f1_2 = 2*(3. - 3.*pow(x,2) + (1. + 4.*x + pow(x,2))*log(x))/(pow(1. - x,5));
  return 2.*f0_2 - f1_2;
}

__device__ double gsH2(double x) // (A.33) second term
{
  if(fabs(x-1.)<loop_w) return 1./10;
  double f0_2 = (1. + 9.*x - 9.*pow(x,2) - pow(x,3) + 6.*x*(x + 1.)*log(x))/(pow(1. - x,5));
  return f0_2;
}


//   P-ll loop functions with U-D splitting, Teppei, Isabel:
__device__ double LoopF(double x,double y)
{
  if (fabs(y-1)<loop_wg && fabs(x-1)<loop_wg) return 0.5;
  
  else if (fabs(y-1)<loop_wg) return (x*log(x) - x + 1)/(pow(x, 2) - 2*x + 1);
  else if (fabs(x-1)<loop_wg) return (y*log(y) - y + 1)/(pow(y, 2) - 2*y + 1);
  else if (fabs(x-y)<loop_wg) return (x - log(x) - 1)/(pow(x, 2) - 2*x + 1);
  
  else return x*log(x)/((x - 1)*(x - y)) + y*log(y)/((-x + y)*(y - 1));
}
__device__ double LoopG(double x,double y)
{
  if (fabs(x-1) < loop_wg && fabs(y-1)< loop_wg) return -1./6;

  else if (fabs(x-1) < loop_wg) return -(pow(y, 2) - 2*y*log(y) - 1)/(2*pow(y, 3) - 6*pow(y, 2) + 6*y - 2);
  else if (fabs(y-1) < loop_wg) return -(pow(x, 2) - 2*x*log(x) - 1)/(2*pow(x, 3) - 6*pow(x, 2) + 6*x - 2);
  else if (fabs(x-y) < loop_wg) return -1.0*(1.0*x*log(x) - 2.0*x + 1.0*log(x) + 2.0)/
                                  (1.0*pow(x, 3) - 3.0*pow(x, 2) + 3.0*x - 1.0);
  else return x*log(x)/(pow(x - 1, 2)*(x - y)) + y*log(y)/((-x + y)*pow(y - 1, 2)) + 1.0/((x - 1)*(y - 1));
}
__device__ double LoopH(double x,double y)
{

  if (fabs(x-1) < loop_wg && fabs(y-1)< loop_wg) return 1./12;
  else if (fabs(x-1) < loop_wg) return (pow(y, 2) - 4*y*log(y) + 4*y - 2*log(y) - 5)/
                                 (2*pow(y, 4) - 8*pow(y, 3) + 12*pow(y, 2) - 8*y + 2);
  else if (fabs(y-1) < loop_wg) return (pow(x, 3) - 6*pow(x, 2) + 6*x*log(x) + 3*x + 2)/
                                 (6*pow(x, 4) - 24*pow(x, 3) + 36*pow(x, 2) - 24*x + 6);
  else if (fabs(x-y) < loop_wg) return (2*pow(x, 2)*log(x) - 5*pow(x, 2) + 4*x*log(x) + 4*x + 1)/
                               (2*pow(x, 5) - 8*pow(x, 4) + 12*pow(x, 3) - 8*pow(x, 2) + 2*x);
  else return x*log(x)/(pow(x - 1, 2)*pow(x - y, 2)) + (x*y + x - 2*pow(y, 2))*log(y)/
  (pow(x - y, 2)*pow(y - 1, 3)) - (2*x - y - 1)/((x - 1)*(x - y)*pow(y - 1, 2));
}

/// gluino loop functions
__device__ double g1_2(double x)
{
if(fabs(x-1.)<loop_wg) return 1./360.;  
return (1./18.)*x*(49*x + 26)*log(x)/pow(-x + 1, 6) + (1./216.)*(4*pow(x, 4) - 39*pow(x, 3) + 
237*pow(x, 2) + 665*x + 33)/pow(-x + 1, 5);
}

__device__ double g1_3(double x)
{
if(fabs(x-1.)<loop_wg) return 1./3780.;
return -1./18.*x*(32*x + 13)*log(x)/pow(-x + 1, 7) + (1./1080.)*(6*pow(x, 5) - 55*pow(x, 4) + 255*pow(x, 3) 
- 1005*pow(x, 2) - 1835*x - 66)/pow(-x + 1, 6);
}

__device__ double g4_2(double x)
{
if(fabs(x-1.)<loop_wg) return -1./6.;
return (10./3.)*x*(5*x + 1)*log(x)/pow(-x + 1, 6) + (1./18.)*(7*pow(x, 4) - 48*pow(x, 3) + 192*pow(x, 2) + 
212*x - 3)/pow(-x + 1, 5);
}

__device__ double g4_3(double x)
{
if(fabs(x-1.)<loop_wg) return 37./630.;
return -1./3.*x*(31*x + 5)*log(x)/pow(-x + 1, 7) + (1./180.)*(21*pow(x, 5) - 152*pow(x, 4) + 528*pow(x, 3) -
 1452*pow(x, 2) - 1117*x + 12)/pow(-x + 1, 6);
}

__device__ double g5_2(double x)
{
if(fabs(x-1.)<loop_wg) return 1./90.;
return (2./9.)*x*(19*x + 11)*log(x)/pow(-x + 1, 6) + (1./54.)*(pow(x, 4) - 12*pow(x, 3) + 84*pow(x, 2) +
 272*x + 15)/pow(-x + 1, 5);
}

__device__ double g5_3(double x)
{
if(fabs(x-1.)<loop_wg) return -1./378.;
return -1./9.*x*(25*x + 11)*log(x)/pow(-x + 1, 7) + (1./540.)*(3*pow(x,5) - 32*pow(x, 4) + 168*pow(x, 3) - 
732*pow(x, 2) - 1507*x - 60)/pow(-x + 1, 6);
}

__device__ double g7_1(double x)
{
if(fabs(x-1.)<loop_wg) return -1./27.;
return -4./9.*x*(x + 2)*log(x)/pow(-x + 1, 4) + (1./9.)*(-10*x - 2)/pow(-x + 1, 3);
}

__device__ double g7_2(double x)
{
if(fabs(x-1.)<loop_wg) return -1./45.;
return -4./3.*x*(x + 1)*log(x)/pow(-x + 1, 5) + (1./9.)*(-2*pow(x, 2) - 20*x - 2)/pow(-x + 1, 4);
}


__device__ double g8_1(double x)
{
if(fabs(x-1.)<loop_wg) return -5./36.;
return (1./3.)*(x + 11)/pow(-x + 1, 3) + (1./6.)*(-pow(x, 2) + 16*x + 9)*log(x)/pow(-x + 1, 4);
}


__device__ double g8_2(double x)
{
if(fabs(x-1.)<loop_wg) return -7./120.;
return (1./12.)*(-pow(x, 2) + 44*x + 53)/pow(-x + 1, 4) + (1./2.)*(2*pow(x, 2) + 11*x + 3)*log(x)/pow(-x + 1, 5);
}

/*-----Teppei Loop functions for epsilon_k---------------------------*/
__device__ double tep_f3(double x,double y)
{

  if((fabs(x-1.)<loop_wg) && (fabs(y-1.)>loop_wg)) return (6*pow(y, 3)*log(y) - 17*pow(y, 3) 
                                                           + 18*pow(y, 2)*log(y) + 9*pow(y, 2) + 9*y - 1)
                                                     /(6*pow(y, 5) - 30*pow(y, 4) + 60*pow(y, 3) - 60*pow(y, 2) + 30*y - 6);

  if((fabs(y-1.)<loop_wg) && (fabs(x-1.)>loop_wg)) return (6*pow(x, 3)*log(x) - 17*pow(x, 3) 
                                                           + 18*pow(x, 2)*log(x) + 9*pow(x, 2) +9*x - 1)/
                                                     (6*pow(x, 5) - 30*pow(x, 4) + 60*pow(x, 3) - 60*pow(x, 2) + 30*x - 6);


  if((fabs(x-y)<loop_wg) && (fabs(x-1.)>loop_wg)) return (pow(x, 3) - 9*pow(x, 2) + 18*x*log(x) - 9*x + 6*log(x) + 17)
                                                    /(6*pow(x, 5) - 30*pow(x, 4) + 60*pow(x, 3) - 60*pow(x, 2) + 30*x - 6);


  if((fabs(x-y)<loop_wg) && (fabs(x-1)<loop_wg)) return 1./20.;
  
  
  return -pow(x, 2)*(x*(x + y + 1) - 3*y)*log(x)/
    (pow(-x + 1, 3)*pow(x - y, 3)) - pow(y, 2)*(-3*x + y*(x + y + 1))*log(y)/(pow(-x + y, 3)*pow(-y + 1, 3)) 
    - (2*pow(x, 2)*pow(y, 2) - 2*pow(x, 2)*y + 2*pow(x,2) - 2*x*pow(y, 2) - 2*x*y +
       2*pow(y, 2))/(pow(-x + 1, 2)*pow(x - y, 2)*pow(-y + 1, 2));
}


__device__ double tep_g4_1(double x,double y)
{
  if((fabs(x-1.)<loop_wg) &&(fabs(y-1.)>loop_wg)) return (14*pow(y, 4)*log(y) - 39*pow(y, 4) + 38*pow(y, 3)*log(y) 
+ 27*pow(y, 3) - 4*pow(y, 2)*log(y)+ 15*pow(y, 2) - 3*y)/(6*pow(y, 5) - 30*pow(y, 4) + 60*pow(y, 3) - 60*pow(y, 2) + 30*y - 6);

  if((fabs(y-1.)<loop_wg) &&(fabs(x-1.)>loop_wg)) 
    return  (14*pow(x, 4)*log(x) - 39*pow(x, 4) + 38*pow(x, 3)*log(x) + 27*pow(x, 3) -
             4*pow(x, 2)*log(x) + 15*pow(x, 2) - 3*x)/(6*pow(x, 5) - 30*pow(x, 4) + 60*pow(x, 3) - 60*pow(x, 2) + 30*x - 6);

  if((fabs(x-y)<loop_wg) &&(fabs(x-1.)>loop_wg)) return  (7*pow(x, 5) - 61*pow(x, 4) + 114*pow(x, 3)*log(x) - 45*pow(x, 3) 
+ 30*pow(x, 2)*log(x) + 101*pow(x, 2) - 2*x)/(18*pow(x, 5) - 90*pow(x, 4) + 180*pow(x, 3) - 180*pow(x, 2) + 90*x - 18);

  if((fabs(x-y)<loop_wg) && (fabs(x-1)<loop_wg)) return 23./180.;


return -1./3.*pow(x, 2)*y*(pow(x, 2)*(7*x + 5) + y*(x*(7*x - 21) + 2))*log(x)/(pow(-x + 1, 3)*pow(x - y, 3)) -
 1./3.*x*pow(y, 2)*(x*(y*(7*y - 21) + 2) + pow(y, 2)*(7*y + 5))*log(y)/(pow(-x + y, 3)
*pow(-y + 1, 3)) + (1./3.)*x*y*(-14*pow(x, 2)*pow(y, 2) + 15*pow(x, 2)*y - 13*pow(x, 2) + 15*x*pow(y, 2) + 8*x*y + x
 - 13*pow(y, 2) + y)/(pow(-x + 1, 2)*pow(x - y, 2)*pow(-y + 1, 2));
}


__device__ double tep_g5_1(double x,double y)
{

  if((fabs(x-1.)<loop_wg) &&(fabs(y-1.)>loop_wg)) return (2*pow(y, 4)*log(y) - 9*pow(y, 4) + 26*pow(y, 3)*log(y)
 - 27*pow(y, 3) + 20*pow(y, 2)*log(y) + 33*pow(y, 2) + 3*y)/(18*pow(y, 5) - 90*pow(y, 4) + 180*pow(y, 3) - 180*pow(y, 2)
 + 90*y - 18);


  if((fabs(y-1.)<loop_wg) &&(fabs(x-1.)>loop_wg)) return (2*pow(x, 4)*log(x) - 9*pow(x, 4) + 26*pow(x, 3)*log(x) 
- 27*pow(x, 3) + 20*pow(x, 2)*log(x) + 33*pow(x, 2) + 3*x)/(18*pow(x, 5) - 90*pow(x, 4) + 180*pow(x, 3) - 180*pow(x, 2)
 + 90*x - 18);


  if((fabs(x-y)<loop_wg) &&(fabs(x-1.)>loop_wg)) return (pow(x, 5) - 19*pow(x, 4) + 78*pow(x, 3)*log(x) - 99*pow(x, 3)
 + 66*pow(x, 2)*log(x) + 107*pow(x, 2) + 10*x)/(54*pow(x, 5) - 270*pow(x, 4) + 540*pow(x, 3) - 540*pow(x, 2) + 270*x - 54);

  if((fabs(x-y)<loop_wg) && (fabs(x-1)<loop_wg)) return -7./540.;

return -1./9.*pow(x, 2)*y*(pow(x, 2)*(x + 11) + y*(x - 5)*(x + 2))*log(x)/(pow(-x + 1, 3)*pow(x - y, 3)) -
 1./9.*x*pow(y, 2)*(x*(y - 5)*(y + 2) + pow(y, 2)*(y + 11))*log(y)/(pow(-x + y, 3)*pow(-y + 1, 3)) 
- 1./9.*x*y*(2*pow(x, 2)*pow(y, 2) + 3*pow(x, 2)*y + 7*pow(x, 2) + 3*x*pow(y, 2) - 32*x*y + 5*x + 7*pow(y, 2)
 + 5*y)/(pow(-x + 1, 2)*pow(x - y, 2)*pow(-y + 1, 2));
}



__device__ double tep_ggw_1(double x,double y)
{

  if((fabs(x-1.)<loop_wg) &&(fabs(y-1.)>loop_wg)) return -(-2*pow(y, 29./2.) + 32*pow(y, 27./2.) - 246*pow(y, 25./2.) 
+ 24*pow(y, 23./2.)*log(y) + 1160*pow(y, 23./2.) - 240*pow(y, 21./2.)*log(y) - 3674*pow(y, 21./2.) +
 1080*pow(y, 19./2.)*log(y) + 8184*pow(y, 19./2.) - 2880*pow(y, 17./2.)*log(y) - 13134*pow(y, 17./2.)
 + 5040*pow(y, 15./2.)*log(y) + 15312*pow(y, 15./2.) - 6048*pow(y, 13./2.)*log(y) - 12870*pow(y, 13./2.)
 + 5040*pow(y, 11./2.)*log(y) + 7568*pow(y, 11./2.) - 2880*pow(y, 9./2.)*log(y) - 2882*pow(y, 9./2.) +
 1080*pow(y, 7./2.)*log(y) + 552*pow(y, 7./2.) - 240*pow(y, 5./2.)*log(y) + 34*pow(y, 5./2.) + 
24*pow(y, 3./2.)*log(y) - 40*pow(y, 3./2.) + 6*sqrt(y) + pow(y, 14) - 18*pow(y, 13) + 12*pow(y, 12)*log(y) + 
125*pow(y, 12) - 120*pow(y, 11)*log(y) - 472*pow(y, 11) + 540*pow(y, 10)*log(y) + 1089*pow(y, 10) - 1440*pow(y, 9)*log(y) - 
1562*pow(y, 9) + 2520*pow(y, 8)*log(y) + 1221*pow(y, 8) - 3024*pow(y, 7)*log(y) + 2520*pow(y, 6)*log(y) - 1221*pow(y,6) - 
1440*pow(y, 5)*log(y) + 1562*pow(y, 5) + 540*pow(y, 4)*log(y) - 1089*pow(y, 4) - 120*pow(y, 3)*log(y) + 472*pow(y, 3) + 
12*pow(y, 2)*log(y) - 125*pow(y, 2) + 18*y - 1)/(24*pow(y, 15) - 360*pow(y, 14) + 2520*pow(y, 13) - 10920*pow(y, 12) +
 32760*pow(y, 11) - 72072*pow(y, 10) + 120120*pow(y, 9) - 154440*pow(y, 8) + 154440*pow(y, 7) - 120120*pow(y, 6) + 
72072*pow(y, 5) - 32760*pow(y, 4) + 10920*pow(y,3) - 2520*pow(y, 2) + 360*y - 24);



  if((fabs(y-1.)<loop_wg) &&(fabs(x-1.)>loop_wg)) return -(-2*pow(x, 29./2.) + 32*pow(x, 27./2.) - 246*pow(x, 25./2.) 
+ 24*pow(x, 23./2.)*log(x) + 1160*pow(x, 23./2.) - 240*pow(x, 21./2.)*log(x) - 3674*pow(x, 21./2.) +
 1080*pow(x, 19./2.)*log(x) + 8184*pow(x, 19./2.) - 2880*pow(x, 17./2.)*log(x) - 13134*pow(x, 17./2.) + 
5040*pow(x, 15./2.)*log(x) + 15312*pow(x, 15./2.) - 6048*pow(x, 13./2.)*log(x) - 12870*pow(x, 13./2.) +
 5040*pow(x, 11./2.)*log(x) + 7568*pow(x, 11./2.) - 2880*pow(x, 9./2.)*log(x) - 2882*pow(x, 9./2.) +
 1080*pow(x, 7./2.)*log(x) + 552*pow(x, 7./2.) - 240*pow(x, 5./2.)*log(x) + 34*pow(x, 5./2.) +
 24*pow(x, 3./2.)*log(x) - 40*pow(x, 3./2.) + 6*sqrt(x) + pow(x, 14) - 18*pow(x, 13) + 12*pow(x, 12)*log(x) + 
125*pow(x, 12) - 120*pow(x, 11)*log(x) - 472*pow(x, 11) + 540*pow(x, 10)*log(x) + 1089*pow(x, 10) - 1440*pow(x, 9)*log(x) - 
1562*pow(x, 9) + 2520*pow(x, 8)*log(x) + 1221*pow(x, 8) - 3024*pow(x, 7)*log(x) + 2520*pow(x, 6)*log(x) - 1221*pow(x, 6) -
 1440*pow(x, 5)*log(x) + 1562*pow(x, 5) + 540*pow(x, 4)*log(x) - 1089*pow(x, 4) - 120*pow(x, 3)*log(x) + 472*pow(x, 3) + 
12*pow(x, 2)*log(x) - 125*pow(x, 2) + 18*x - 1)/(24*pow(x, 15) - 360*pow(x, 14) + 2520*pow(x, 13) - 10920*pow(x, 12) + 
32760*pow(x, 11) - 72072*pow(x, 10) + 120120*pow(x, 9) - 154440*pow(x, 8) + 154440*pow(x, 7) - 120120*pow(x, 6) + 
72072*pow(x, 5) - 32760*pow(x, 4) + 10920*pow(x,3) - 2520*pow(x, 2) + 360*x - 24);



  if((fabs(x-y)<loop_wg) &&(fabs(x-1.)>loop_wg)) return (pow(x, 3)*sqrt(pow(x, 2)) - pow(x, 3) - 9*pow(x, 2)*sqrt(pow(x, 2)) +
 6*pow(x, 2)*log(x) - 9*pow(x, 2) + 18*x*sqrt(pow(x, 2))*log(x) - 9*x*sqrt(pow(x, 2)) + 6*x*log(x) + 9*x +
 6*sqrt(pow(x, 2))*log(x) + 17*sqrt(pow(x, 2)) + 1)/(6*pow(x, 5) - 30*pow(x, 4) + 60*pow(x, 3) - 60*pow(x, 2) + 30*x - 6);


  if((fabs(x-y)<loop_wg) && (fabs(x-1)<loop_wg)) return 1./30.;


return -pow(x, 2)*log(x)/(pow(-x + 1, 4)*(2*x - 2*y)) - pow(y, 2)*log(y)/((-2*x + 2*y)*pow(-y + 1, 4)) -
 sqrt(x*y)*(x*log(x)/(pow(-x + 1, 4)*(x - y)) + y*log(y)/((-x + y)*pow(-y + 1, 4)) + (1./6.)*(-pow(x, 2)*pow(y, 2) +
 2*pow(x, 2) + 5*x*y*(x + y) - 10*x*y - 7*x + 2*pow(y, 2) - 7*y + 11)/(pow(-x + 1, 3)*pow(-y + 1, 3))) -
 1./12.*(2*pow(x, 2)*pow(y, 2) - pow(x, 2) + 5*x*y*(x + y) - 22*x*y + 5*x -
 pow(y, 2) + 5*y + 2)/(pow(-x + 1, 3)*pow(-y + 1, 3));

}


__device__ double tep_ggw_2(double x,double y)
{

  if((fabs(x-1.)<loop_wg) &&(fabs(y-1.)>loop_wg)) return (-6*pow(y, 41./2.) + 130*pow(y, 39./2.) - 1350*pow(y, 37./2.) +
 8970*pow(y, 35./2.) - 120*pow(y, 33./2.)*log(y) - 42720*pow(y, 33./2.) + 1800*pow(y, 31./2.)*log(y) + 
154344*pow(y, 31./2.) - 12600*pow(y, 29./2.)*log(y) - 436440*pow(y, 29./2.) + 54600*pow(y, 27./2.)*log(y) + 
983400*pow(y, 27./2.) - 163800*pow(y, 25./2.)*log(y) - 1783860*pow(y, 25./2.) + 360360*pow(y, 23./2.)*log(y) + 
2618460*pow(y, 23./2.) - 600600*pow(y, 21./2.)*log(y) - 3113396*pow(y, 21./2.) + 772200*pow(y, 19./2.)*log(y) +
 2989740*pow(y, 19./2.) - 772200*pow(y, 17./2.)*log(y) - 2301000*pow(y, 17./2.) + 600600*pow(y, 15./2.)*log(y) + 
1399560*pow(y, 15./2.) - 360360*pow(y, 13./2.)*log(y) - 656760*pow(y, 13./2.) + 163800*pow(y, 11./2.)*log(y) +
 227784*pow(y, 11./2.) - 54600*pow(y, 9./2.)*log(y) - 53430*pow(y, 9./2.) + 12600*pow(y, 7./2.)*log(y) +
 6450*pow(y, 7./2.) - 1800*pow(y, 5./2.)*log(y) + 330*pow(y, 5./2.) + 120*pow(y, 3./2.)*log(y) - 
230*pow(y, 3./2.) + 24*sqrt(y) + 2*pow(y, 20) - 45*pow(y, 19) + 495*pow(y, 18) - 60*pow(y, 17)*log(y) - 3405*pow(y, 17) +
 900*pow(y, 16)*log(y) + 16125*pow(y, 16) - 6300*pow(y, 15)*log(y) - 55428*pow(y, 15) + 27300*pow(y, 14)*log(y) + 
142860*pow(y, 14) - 81900*pow(y, 13)*log(y) - 281460*pow(y, 13) + 180180*pow(y, 12)*log(y) + 427440*pow(y, 12) -
 300300*pow(y, 11)*log(y) - 498550*pow(y, 11) + 386100*pow(y, 10)*log(y) + 436722*pow(y, 10) - 386100*pow(y, 9)*log(y) - 
268710*pow(y, 9) + 300300*pow(y, 8)*log(y) + 89310*pow(y, 8) - 180180*pow(y, 7)*log(y) + 20460*pow(y, 7) +
 81900*pow(y, 6)*log(y) - 48900*pow(y, 6) - 27300*pow(y, 5)*log(y) + 34332*pow(y, 5) + 6300*pow(y, 4)*log(y) - 
14730*pow(y, 4) - 900*pow(y, 3)*log(y) + 4155*pow(y, 3) + 60*pow(y, 2)*log(y) -745*pow(y, 2) + 75*y - 3)/(60*pow(y, 21) - 
1260*pow(y, 20) + 12600*pow(y, 19) - 79800*pow(y, 18) + 359100*pow(y, 17) - 1220940*pow(y, 16) + 3255840*pow(y, 15) -
 6976800*pow(y, 14) + 12209400*pow(y, 13) -17635800*pow(y, 12) + 21162960*pow(y, 11) - 21162960*pow(y, 10) +
 17635800*pow(y, 9) - 12209400*pow(y, 8) + 6976800*pow(y, 7) - 3255840*pow(y, 6) + 1220940*pow(y, 5) - 359100*pow(y, 4) +
 79800*pow(y, 3) -12600*pow(y, 2) + 1260*y - 60);




  if((fabs(y-1.)<loop_wg) &&(fabs(x-1.)>loop_wg)) return (-6*pow(x, 41./2.) + 130*pow(x, 39./2.) - 1350*pow(x, 37./2.) + 
8970*pow(x, 35./2.) - 120*pow(x, 33./2.)*log(x) - 42720*pow(x, 33./2.) + 1800*pow(x, 31./2.)*log(x) + 
154344*pow(x, 31./2.) - 12600*pow(x, 29./2.)*log(x) - 436440*pow(x, 29./2.) + 54600*pow(x, 27./2.)*log(x) + 
983400*pow(x, 27./2.) - 163800*pow(x, 25./2.)*log(x) - 1783860*pow(x, 25./2.) + 360360*pow(x, 23./2.)*log(x) +
 2618460*pow(x, 23./2.) - 600600*pow(x, 21./2.)*log(x) - 3113396*pow(x, 21./2.) + 772200*pow(x, 19./2.)*log(x) +
 2989740*pow(x, 19./2.) - 772200*pow(x, 17./2.)*log(x) - 2301000*pow(x, 17./2.) + 600600*pow(x, 15./2.)*log(x) +
 1399560*pow(x, 15./2.) - 360360*pow(x, 13./2.)*log(x) - 656760*pow(x, 13./2.) + 163800*pow(x, 11./2.)*log(x) + 
227784*pow(x, 11./2.) - 54600*pow(x, 9./2.)*log(x) - 53430*pow(x, 9./2.) + 12600*pow(x, 7./2.)*log(x) +
 6450*pow(x, 7./2.) - 1800*pow(x, 5./2.)*log(x) + 330*pow(x, 5./2.) + 120*pow(x, 3./2.)*log(x) - 
230*pow(x, 3./2.) + 24*sqrt(x) + 2*pow(x, 20) - 45*pow(x, 19) + 495*pow(x, 18) - 60*pow(x, 17)*log(x) - 3405*pow(x, 17) +
 900*pow(x, 16)*log(x) + 16125*pow(x, 16) - 6300*pow(x, 15)*log(x) - 55428*pow(x, 15) + 27300*pow(x, 14)*log(x) +
 142860*pow(x, 14) - 81900*pow(x, 13)*log(x) - 281460*pow(x, 13) + 180180*pow(x, 12)*log(x) + 427440*pow(x, 12) -
 300300*pow(x, 11)*log(x) - 498550*pow(x, 11) + 386100*pow(x, 10)*log(x) + 436722*pow(x, 10) - 386100*pow(x, 9)*log(x) - 
268710*pow(x, 9) + 300300*pow(x, 8)*log(x) + 89310*pow(x, 8) - 180180*pow(x, 7)*log(x) + 20460*pow(x, 7) +
 81900*pow(x, 6)*log(x) - 48900*pow(x, 6) - 27300*pow(x, 5)*log(x) + 34332*pow(x, 5) + 6300*pow(x, 4)*log(x) - 
14730*pow(x, 4) - 900*pow(x, 3)*log(x) + 4155*pow(x, 3) + 60*pow(x, 2)*log(x) -745*pow(x, 2) + 75*x - 3)/(60*pow(x, 21) - 
1260*pow(x, 20) + 12600*pow(x, 19) - 79800*pow(x, 18) + 359100*pow(x, 17) - 1220940*pow(x, 16) + 3255840*pow(x, 15) -
 6976800*pow(x, 14) + 12209400*pow(x, 13) -17635800*pow(x, 12) + 21162960*pow(x, 11) - 21162960*pow(x, 10) +
 17635800*pow(x, 9) - 12209400*pow(x, 8) + 6976800*pow(x, 7) - 3255840*pow(x, 6) + 1220940*pow(x, 5) - 359100*pow(x, 4) +
 79800*pow(x, 3) -12600*pow(x, 2) + 1260*x - 60);




  if((fabs(x-y)<loop_wg) &&(fabs(x-1.)>loop_wg)) 
    return  -(2*pow(x, 4)*sqrt(pow(x, 2)) - pow(x, 4) - 16*pow(x, 3)*sqrt(pow(x, 2)) +
              12*pow(x, 3) + 72*pow(x, 2)*sqrt(pow(x, 2)) - 36*pow(x, 2)*log(x) + 36*pow(x, 2) 
              - 96*x*sqrt(pow(x, 2))*log(x) + 16*x*sqrt(pow(x, 2)) - 24*x*log(x) - 44*x - 24*sqrt(pow(x, 2))*log(x) 
              - 74*sqrt(pow(x, 2)) - 3)/(12*pow(x, 6) - 72*pow(x, 5) + 180*pow(x, 4) - 240*pow(x, 3) + 180*pow(x, 2) - 72*x + 12);


  if((fabs(x-y)<loop_wg) && (fabs(x-1)<loop_wg)) return -1./20.;


return pow(x, 2)*log(x)/(pow(-x + 1, 5)*(x - y)) + pow(y, 2)*log(y)/((-x + y)*pow(-y + 1, 5)) + 
sqrt(x*y)*(2*x*log(x)/(pow(-x + 1, 5)*(x - y)) + 2*y*log(y)/((-x + y)*pow(-y + 1, 5)) + (1./6.)*(-pow(x,3)*pow(y, 3) - 
3*pow(x, 3) + 5*pow(x, 2)*pow(y, 2)*(x + y) - 27*pow(x, 2)*pow(y, 2) + 13*pow(x, 2) + 45*x*y*(x + y) - 13*x*y*(pow(x, 2) +
 pow(y, 2))- 45*x*y - 23*x - 3*pow(y, 3) + 13*pow(y, 2) - 23*y + 25)/(pow(-x + 1, 4)*pow(-y + 1, 4))) + 
(1./12.)*(pow(x, 3)*pow(y, 3) + pow(x, 3) - 7*pow(x, 2)*pow(y, 2)*(x + y) + 15*pow(x, 2)*pow(y, 2) - 5*pow(x, 2) + 
33*x*y*(x + y) - 7*x*y*(pow(x, 2) + pow(y, 2)) - 75*x*y + 13*x + pow(y, 3) -
 5*pow(y, 2) + 13*y + 3)/(pow(-x + 1, 4)*pow(-y + 1, 4));
}



__device__ double tep_ggw_3(double x,double y)
{

  if((fabs(x-1.)<loop_wg) &&(fabs(y-1.)>loop_wg)) return -(-4*pow(y, 55./2.) + 114*pow(y, 53./2.) - 1570*pow(y, 51./2.) +
 13920*pow(y, 49./2.) - 89340*pow(y, 47./2.) + 120*pow(y, 45./2.)*log(y) + 442400*pow(y, 45./2.) -
 2520*pow(y, 43./2.)*log(y) - 1758240*pow(y, 43./2.) + 25200*pow(y, 41./2.)*log(y) + 5755860*pow(y, 41./2.) -
 159600*pow(y, 39./2.)*log(y) - 15794680*pow(y, 39./2.) + 718200*pow(y, 37./2.)*log(y) + 36761010*pow(y, 37./2.)
 - 2441880*pow(y, 35./2.)*log(y) - 73127010*pow(y, 35./2.) + 6511680*pow(y, 33./2.)*log(y) +
 124917020*pow(y, 33./2.) - 13953600*pow(y, 31./2.)*log(y) - 183683640*pow(y, 31./2.) +
 24418800*pow(y, 29./2.)*log(y) + 232637520*pow(y, 29./2.) - 35271600*pow(y, 27./2.)*log(y) -
 253516240*pow(y, 27./2.) + 42325920*pow(y, 25./2.)*log(y) + 237094920*pow(y, 25./2.) -
 42325920*pow(y, 23./2.)*log(y) - 189478260*pow(y, 23./2.) + 35271600*pow(y, 21./2.)*log(y) +
 128574710*pow(y, 21./2.) - 24418800*pow(y, 19./2.)*log(y) - 73415430*pow(y, 19./2.) + 
13953600*pow(y, 17./2.)*log(y) + 34825560*pow(y, 17./2.) - 6511680*pow(y, 15./2.)*log(y) - 
13472140*pow(y, 15./2.) + 2441880*pow(y, 13./2.)*log(y) + 4131600*pow(y, 13./2.) - 
718200*pow(y, 11./2.)*log(y) - 957840*pow(y, 11./2.) + 159600*pow(y, 9./2.)*log(y) + 152420*pow(y, 9./2.) - 
25200*pow(y, 7./2.)*log(y) - 12240*pow(y,7./2.) + 2520*pow(y, 5./2.)*log(y) - 666*pow(y, 5./2.) -
 120*pow(y, 3./2.)*log(y) + 266*pow(y, 3./2.) - 20*sqrt(y) + pow(y, 27) - 29*pow(y, 26) + 408*pow(y, 25) -
 3720*pow(y, 24) + 60*pow(y, 23)*log(y) + 24640*pow(y, 23) - 1260*pow(y, 22)*log(y) - 125640*pow(y, 22) + 
12600*pow(y, 21)*log(y) + 509850*pow(y, 21) - 79800*pow(y, 20)*log(y) - 1681130*pow(y, 20) + 359100*pow(y, 19)*log(y) +
 4566705*pow(y, 19) - 1220940*pow(y, 18)*log(y) - 10317285*pow(y, 18) + 3255840*pow(y, 17)*log(y) + 19510150*pow(y, 17) -
 6976800*pow(y, 16)*log(y) - 30998310*pow(y, 16) + 12209400*pow(y, 15)*log(y) + 41434440*pow(y, 15) - 17635800*pow(y, 14)*log(y) - 
46524920*pow(y, 14) + 21162960*pow(y, 13)*log(y) + 43663140*pow(y, 13) - 21162960*pow(y, 12)*log(y) - 33895620*pow(y, 12) +
 17635800*pow(y, 11)*log(y) + 21338995*pow(y, 11) - 12209400*pow(y, 10)*log(y) - 10461495*pow(y, 10) + 6976800*pow(y, 9)*log(y) +
 3598980*pow(y, 9) - 3255840*pow(y, 8)*log(y) - 519860*pow(y, 8) + 1220940*pow(y, 7)*log(y) - 302280*pow(y,7) - 
359100*pow(y, 6)*log(y) + 274560*pow(y, 6) + 79800*pow(y, 5)*log(y) - 120350*pow(y, 5) - 12600*pow(y, 4)*log(y) + 34830*pow(y, 4)
 + 1260*pow(y, 3)*log(y) - 6885*pow(y, 3) - 60*pow(y, 2)*log(y) + 889*pow(y, 2) - 66*y + 2)/(120*pow(y, 28) - 3360*pow(y, 27) + 
45360*pow(y, 26) - 393120*pow(y, 25) + 2457000*pow(y, 24) - 11793600*pow(y, 23) + 45208800*pow(y, 22) - 142084800*pow(y, 21) +
 372972600*pow(y, 20) - 828828000*pow(y, 19) + 1574773200*pow(y, 18) - 2576901600*pow(y, 17) + 3650610600*pow(y, 16) -
 4493059200*pow(y, 15) + 4813992000*pow(y, 14) - 4493059200*pow(y, 13) + 3650610600*pow(y, 12) - 2576901600*pow(y, 11) +
 1574773200*pow(y, 10) - 828828000*pow(y, 9) + 372972600*pow(y, 8) - 142084800*pow(y, 7) + 45208800*pow(y, 6) - 
11793600*pow(y, 5) + 2457000*pow(y, 4) - 393120*pow(y, 3) + 45360*pow(y, 2) -3360*y + 120);



  if((fabs(y-1.)<loop_wg) &&(fabs(x-1.)>loop_wg)) return -(-4*pow(x, 55./2.) + 114*pow(x, 53./2.) - 
1570*pow(x, 51./2.) + 13920*pow(x, 49./2.) - 89340*pow(x, 47./2.) + 120*pow(x, 45./2.)*log(x) +
 442400*pow(x, 45./2.) - 2520*pow(x, 43./2.)*log(x) - 1758240*pow(x, 43./2.) + 25200*pow(x, 41./2.)*log(x) +
 5755860*pow(x, 41./2.) - 159600*pow(x, 39./2.)*log(x) - 15794680*pow(x, 39./2.) +
 718200*pow(x, 37./2.)*log(x) + 36761010*pow(x, 37./2.) - 2441880*pow(x, 35./2.)*log(x) -
 73127010*pow(x, 35./2.) + 6511680*pow(x, 33./2.)*log(x) + 124917020*pow(x, 33./2.) - 
13953600*pow(x, 31./2.)*log(x) - 183683640*pow(x, 31./2.) + 24418800*pow(x, 29./2.)*log(x) +
 232637520*pow(x, 29./2.) - 35271600*pow(x, 27./2.)*log(x) - 253516240*pow(x, 27./2.) +
 42325920*pow(x, 25./2.)*log(x) + 237094920*pow(x, 25./2.) - 42325920*pow(x, 23./2.)*log(x) -
 189478260*pow(x, 23./2.) + 35271600*pow(x, 21./2.)*log(x) + 128574710*pow(x, 21./2.) -
 24418800*pow(x, 19./2.)*log(x) - 73415430*pow(x, 19./2.) + 13953600*pow(x, 17./2.)*log(x) +
 34825560*pow(x, 17./2.) - 6511680*pow(x, 15./2.)*log(x) - 13472140*pow(x, 15./2.) + 
2441880*pow(x, 13./2.)*log(x) + 4131600*pow(x, 13./2.) - 718200*pow(x, 11./2.)*log(x) -
 957840*pow(x, 11./2.) + 159600*pow(x, 9./2.)*log(x) + 152420*pow(x, 9./2.) - 25200*pow(x, 7./2.)*log(x) -
 12240*pow(x,7./2.) + 2520*pow(x, 5./2.)*log(x) - 666*pow(x, 5./2.) - 120*pow(x, 3./2.)*log(x) +
 266*pow(x, 3./2.) - 20*sqrt(x) + pow(x, 27) - 29*pow(x, 26) + 408*pow(x, 25) - 3720*pow(x, 24) + 60*pow(x, 23)*log(x) +
 24640*pow(x, 23) - 1260*pow(x, 22)*log(x) - 125640*pow(x, 22) + 12600*pow(x, 21)*log(x) + 509850*pow(x, 21) -
 79800*pow(x, 20)*log(x) - 1681130*pow(x, 20) + 359100*pow(x, 19)*log(x) + 4566705*pow(x, 19) - 1220940*pow(x, 18)*log(x) -
 10317285*pow(x, 18) + 3255840*pow(x, 17)*log(x) + 19510150*pow(x, 17) - 6976800*pow(x, 16)*log(x) - 30998310*pow(x, 16) +
 12209400*pow(x, 15)*log(x) + 41434440*pow(x, 15) - 17635800*pow(x, 14)*log(x) - 46524920*pow(x, 14) +
 21162960*pow(x, 13)*log(x) + 43663140*pow(x, 13) - 21162960*pow(x, 12)*log(x) - 33895620*pow(x, 12) + 
17635800*pow(x, 11)*log(x) + 21338995*pow(x, 11) - 12209400*pow(x, 10)*log(x) - 10461495*pow(x, 10) + 
6976800*pow(x, 9)*log(x) + 3598980*pow(x, 9) - 3255840*pow(x, 8)*log(x) - 519860*pow(x, 8) + 1220940*pow(x, 7)*log(x) -
 302280*pow(x,7) - 359100*pow(x, 6)*log(x) + 274560*pow(x, 6) + 79800*pow(x, 5)*log(x) - 120350*pow(x, 5) -
 12600*pow(x, 4)*log(x) + 34830*pow(x, 4) + 1260*pow(x, 3)*log(x) - 6885*pow(x, 3) - 60*pow(x, 2)*log(x) + 889*pow(x, 2) - 
66*x + 2)/(120*pow(x, 28) - 3360*pow(x, 27) + 45360*pow(x, 26) - 393120*pow(x, 25) + 2457000*pow(x, 24) - 11793600*pow(x, 23) +
 45208800*pow(x, 22) - 142084800*pow(x, 21) + 372972600*pow(x, 20) - 828828000*pow(x, 19) + 1574773200*pow(x, 18) -
 2576901600*pow(x, 17) + 3650610600*pow(x, 16) - 4493059200*pow(x, 15) + 4813992000*pow(x, 14) - 4493059200*pow(x, 13) +
 3650610600*pow(x, 12) - 2576901600*pow(x, 11) + 1574773200*pow(x, 10) - 828828000*pow(x, 9) + 372972600*pow(x, 8) -
 142084800*pow(x, 7) + 45208800*pow(x, 6) - 11793600*pow(x, 5) + 2457000*pow(x, 4) - 393120*pow(x, 3) + 45360*pow(x, 2)
 -3360*x + 120);

if((fabs(x-y)<loop_wg) &&(fabs(x-1.)>loop_wg)) return (3*pow(x, 5)*sqrt(pow(x, 2)) - pow(x, 5) - 25*pow(x, 4)*sqrt(pow(x, 2)) +
 10*pow(x, 4) + 100*pow(x, 3)*sqrt(pow(x, 2)) - 60*pow(x, 3) - 300*pow(x, 2)*sqrt(pow(x, 2)) + 120*pow(x, 2)*log(x) -
 80*pow(x, 2) + 300*x*sqrt(pow(x, 2))*log(x) + 25*x*sqrt(pow(x, 2)) + 60*x*log(x) + 125*x + 60*sqrt(pow(x, 2))*log(x) +
 197*sqrt(pow(x, 2)) + 6)/(60*pow(x, 7) - 420*pow(x, 6) + 1260*pow(x, 5) - 2100*pow(x, 4) +2100*pow(x, 3) - 1260*pow(x, 2) +
 420*x - 60);

if((fabs(x-y)<loop_wg) && (fabs(x-1)<loop_wg)) return 2./105.;


return -pow(x, 2)*log(x)/(pow(-x + 1, 6)*(2*x - 2*y)) - pow(y, 2)*log(y)/((-2*x + 2*y)*pow(-y + 1, 6)) -
 sqrt(x*y)*(x*log(x)/(pow(-x + 1, 6)*(x - y)) + y*log(y)/((-x + y)*pow(-y + 1, 6)) + (1./60.)*(-3*pow(x, 4)*pow(y, 4) +
 12*pow(x, 4) + 17*pow(x, 3)*pow(y, 3)*(x + y) - 98*pow(x, 3)*pow(y, 3) - 63*pow(x, 3) + 262*pow(x, 2)*pow(y, 2)*(x + y) -
 43*pow(x, 2)*pow(y, 2)*(pow(x, 2) + pow(y, 2)) - 618*pow(x, 2)*pow(y, 2) + 137*pow(x, 2) + 622*x*y*(x + y) -
 358*x*y*(pow(x, 2) + pow(y, 2)) + 77*x*y*(pow(x, 3) + pow(y, 3)) - 418*x*y - 163*x + 12*pow(y, 4) - 63*pow(y, 3) + 
137*pow(y, 2) - 163*y + 137)/(pow(-x +1, 5)*pow(-y + 1, 5))) - 1./120.*(2*pow(x, 4)*pow(y, 4) - 3*pow(x, 4) -
 13*pow(x, 3)*pow(y, 3)*(x + y) + 92*pow(x, 3)*pow(y, 3) + 17*pow(x, 3) - 188*pow(x, 2)*pow(y, 2)*(x + y) + 
47*pow(x, 2)*pow(y,2)*(pow(x, 2) + pow(y, 2)) + 192*pow(x, 2)*pow(y, 2) - 43*pow(x, 2) + 352*x*y*(x + y) - 
148*x*y*(pow(x, 2) + pow(y, 2)) + 27*x*y*(pow(x, 3) + pow(y, 3)) - 548*x*y + 77*x - 3*pow(y, 4) + 17*pow(y, 3) -
 43*pow(y, 2) + 77*y + 12)/(pow(-x + 1, 5)*pow(-y + 1, 5));
}
// more loops for bsg
//Loop functions h7 h8 f7_2 f8_2 f7_1 f8_1

__device__ double h7(double x)
{
  if(fabs(x-1.)<loop_wg) return -7.0/36.0;
  

return (1.0/12.0)*(-5*pow(x, 2) + 3*x)/pow(-x + 1, 2) - 1.0/6.0*(3*pow(x, 2)- 2*x)*log(x)/
  pow(-x + 1, 3);
}

__device__ double h8(double x)
{

 if(fabs(x-1.)<loop_wg) return -1.0/6.0;

return (1.0/2.0)*x*log(x)/pow(-x + 1, 3) + (1.0/4.0)*(-pow(x, 2) + 3*x)/pow(-x + 1, 2);
}

__device__ double f7_2(double x)
{

  if(fabs(x-1.)<loop_wg) return 5.0/144.0;

return (1.0/24.0)*(7*x - 13)/pow(-x + 1, 3) - 1.0/12.0*(-2*pow(x, 2) + 2*x +3)*log(x)/
  pow(-x + 1, 4);
}

__device__ double f8_2(double x)
{
  if(fabs(x-1.)<loop_wg) return 1.0/48.0;

return (1.0/4.0)*x*(x + 2)*log(x)/pow(-x + 1, 4) + (1.0/8.0)*(5*x + 1)/pow(-x + 1, 3);
}


__device__ double f7_1(double x,double y)
{
  if((fabs(x-1.)<loop_wg) && (fabs(y-1.))>loop_wg )
    return -(25*pow(y, 8) - 200*pow(y, 7) + 700*pow(y, 6) - 1400*pow(y, 5) 
             - 576*pow(y, 4)*pow(log(y), 2) + 2016*pow(y, 4)*log(y) 
             - 14*pow(y, 4) + 1152*pow(y, 3)*pow(log(y), 2) - 7776*pow(y, 3)*log(y) 
             + 8680*pow(y, 3) + 1152*pow(y, 2)*pow(log(y), 2) + 6480*pow(y, 2)*log(y) 
             - 20252*pow(y, 2) - 1728*y*pow(log(y), 2) + 4896*y*log(y) 
             + 18520*y - 1296*pow(log(y), 2) - 5616*log(y) - 6059)/
      (10368*pow(y, 9)- 93312*pow(y, 8) + 373248*pow(y, 7) 
       - 870912*pow(y, 6) + 1306368*pow(y, 5) - 1306368*pow(y, 4) 
       + 870912*pow(y, 3) - 373248*pow(y, 2) + 93312*y - 10368);

  if((fabs(y-1.)<loop_wg) and(fabs(x-1.))>loop_wg) 
    return -(25*pow(x, 8) - 200*pow(x, 7) + 700*pow(x, 6) - 1400*pow(x, 5) -
             576*pow(x, 4)*pow(log(x), 2) + 2016*pow(x, 4)*log(x) - 14*pow(x, 4) 
             + 1152*pow(x, 3)*pow(log(x), 2) - 7776*pow(x, 3)*log(x) + 8680*pow(x, 3) 
             + 1152*pow(x, 2)*pow(log(x), 2) + 6480*pow(x, 2)*log(x) - 20252*pow(x, 2) 
             - 1728*x*pow(log(x), 2) + 4896*x*log(x) + 18520*x - 1296*pow(log(x), 2) 
             - 5616*log(x) - 6059)/
      (10368*pow(x, 9) -93312*pow(x, 8) + 373248*pow(x, 7) - 870912*pow(x, 6) 
       + 1306368*pow(x, 5) - 1306368*pow(x, 4) + 870912*pow(x, 3) - 373248*pow(x, 2) 
       + 93312*x - 10368);

  if((fabs(x-y)<loop_wg) &&(fabs(x-1.)>loop_wg)) 
    return -(16*pow(x, 5)*pow(log(x), 2) - 64*pow(x, 5)*log(x) + 63*pow(x, 5) 
             - 24*pow(x, 4)*pow(log(x), 2) + 238*pow(x, 4)*log(x) 
             - 369*pow(x, 4) - 72*pow(x, 3)*pow(log(x), 2) - 108*pow(x, 3)*log(x) 
             + 762*pow(x, 3) + 68*pow(x, 2)*pow(log(x), 2) 
             - 368*pow(x, 2)*log(x) - 630*pow(x, 2) + 84*x*pow(log(x), 2) + 284*x*log(x) + 
             135*x + 18*log(x) + 39)/
      (72*pow(x, 10) - 648*pow(x, 9) + 2592*pow(x, 8)- 6048*pow(x, 7) + 9072*pow(x, 6) 
       - 9072*pow(x, 5) + 6048*pow(x, 4) - 2592*pow(x, 3) + 648*pow(x, 2) - 72*x);


 if((fabs(x-y)<loop_wg) && (fabs(x-1)<loop_wg)) return - 13.0/2592.0;

return (2*pow((1.0/24.0)*(7*x - 13)/pow(-x + 1, 3) 
              - 1.0/12.0*(-2*pow(x, 2)+ 2*x + 3)*log(x)/pow(-x + 1, 4), 2) 
        - 2*pow((1.0/24.0)*(7*y - 13)/pow(-y + 1,3) 
                - 1.0/12.0*(-2*pow(y, 2) + 2*y + 3)*log(y)/pow(-y + 1, 4), 2))/(x - y);
}


__device__ double f8_1(double x,double y)
{
  if((fabs(x-1.)<loop_wg) &&(fabs(y-1.)>loop_wg)) 
    return -(pow(y, 8) - 8*pow(y, 7) + 28*pow(y, 6) - 56*pow(y, 5) 
             - 144*pow(y, 4)*pow(log(y), 2) + 720*pow(y, 4)*log(y) - 830*pow(y, 4) 
             - 576*pow(y, 3)*pow(log(y),2) + 864*pow(y, 3)*log(y) + 1384*pow(y, 3) 
             - 576*pow(y, 2)*pow(log(y), 2)- 1296*pow(y, 2)*log(y) - 188*pow(y, 2) 
             - 288*y*log(y) - 296*y - 35)/(1152*pow(y, 9)
                                           - 10368*pow(y, 8)+ 41472*pow(y,7) 
                                           - 96768*pow(y, 6) + 145152*pow(y, 5) 
                                           - 145152*pow(y, 4) + 96768*pow(y, 3) 
                                           - 41472*pow(y, 2) + 10368*y - 1152);


  if((fabs(y-1.)<loop_wg) &&(fabs(x-1.)>loop_wg)) 
    return -(pow(x, 8) - 8*pow(x, 7) + 28*pow(x, 6) - 56*pow(x, 5) 
             - 144*pow(x, 4)*pow(log(x), 2) + 720*pow(x, 4)*log(x) - 830*pow(x, 4) 
             - 576*pow(x, 3)*pow(log(x),2) + 864*pow(x, 3)*log(x) + 1384*pow(x, 3) 
             - 576*pow(x, 2)*pow(log(x), 2) - 1296*pow(x, 2)*log(x) - 188*pow(x, 2) 
             - 288*x*log(x) - 296*x - 35)/(1152*pow(x, 9)- 10368*pow(x, 8) 
                                           + 41472*pow(x, 7) - 96768*pow(x, 6) 
                                           + 145152*pow(x, 5) - 145152*pow(x, 4) 
                                           + 96768*pow(x, 3) 
                                           - 41472*pow(x, 2) + 10368*x - 1152);


  if((fabs(x-y)<loop_wg) && (fabs(x-1.)>loop_wg))
    return -(2*pow(x, 4)*pow(log(x), 2) - 11*pow(x, 4)*log(x) + 15*pow(x, 4) 
             + 12*pow(x, 3)*pow(log(x), 2) - 28*pow(x, 3)*log(x) 
             - 12*pow(x, 3) + 18*pow(x, 2)*pow(log(x), 2) + 18*pow(x, 2)*log(x) 
             - 18*pow(x, 2) + 4*x*pow(log(x), 2) + 20*x*log(x) 
             + 12*x + log(x) + 3)/(4*pow(x, 9) - 36*pow(x, 8) + 144*pow(x, 7) 
                                   - 336*pow(x,6) + 504*pow(x, 5) 
                                   - 504*pow(x, 4) + 336*pow(x, 3) - 144*pow(x, 2) + 36*x - 4);


  if((fabs(x-y)<loop_wg) && (fabs(x-1)<loop_wg)) return - 1.0/720.0;

  return (2*pow((1.0/4.0)*x*(x + 2)*log(x)/pow(-x + 1, 4) + (1.0/8.0)*(5*x + 1)/
                pow(-x + 1, 3), 2)- 2*pow((1.0/4.0)*y*(y + 2)*log(y)/pow(-y + 1, 4) 
                                          + (1.0/8.0)*(5*y + 1)/pow(-y + 1, 3), 2))/(x - y);
}



////////////////////////////////////////////////////////////////////////////////
//                  Integral Ingredients for C9, C10                         ///
//             P_ijk -> from 2F1 analytical expansion                        ///
//              Author: Ramón Ángel Ruiz Fernández                           ///
//                    Email: rruizfer@cern.ch                                ///
//                    Credit: Amine Boussejra                                ///
//              TODO: Improve z=-1 region for 2F1 -> Checked                 ///
//              TODO: Put printf in 1.0e9 regions to help debugging            ///
////////////////////////////////////////////////////////////////////////////////




__device__ double Beta_E(double i, double j){
  // Beta Euler Function
  return tgammaf(i)*tgammaf(j)/tgammaf(i+j);
}


__device__ double factorial (const int n){
  // Factorial algorithm not based on recursivity
   if (n <= 0) { return 1.; }
   double x = 1;
   int b = 0;
   do {
      b++;
      x *= b;
   } while(b!=n);

   return x;
}


__device__ double Pochhammer(double x, int n){
    double res = 1;
    // x = (double) x;
    if (n == 0) return 1;
    else{
        for (int i = 0; i < n; i++)   res *= (x+i) ;
        if (res == 0) printf("Warning in %s, result is null\n",__func__);
        return res;
    }
}


// Replaced by Digamma function wo recurssion (better for CUDA)-> digamma_2
//     // Test implementation : from https://www2.mpia-hd.mpg.de/~mathar/progs/digamma.c and the boost stdlib (https://www.boost.org/doc/libs/1_74_0/libs/math/doc/html/math_toolkit/sf_gamma/digamma.html)
//     // Tested, seems to work correctly. However, Psi(0) = -inf.
//     if(x == 0.) {
//       return 1.0e9;
//     }
//     if (x < 0) return digamma(1.-x) - M_PI/tan(M_PI*x); // Reflection formula
//     else if (x < 1) return digamma(1. + x) - 1./x ;
//     else if (x == 1.) return -M_GAMMAl;
//     else if (x == 2.) return 1. - M_GAMMAl;
//     else if (x == 3.) return 1.5 - M_GAMMAl;
//     else if (x > 3.) return 0.5*(digamma(x/2.) + digamma((x+1.)/2.)) +  M_LN2l ;
//     else{
//       static double Kncoe[] = { .30459198558715155634315638246624251L,
//                 .72037977439182833573548891941219706L, -.12454959243861367729528855995001087L,
//                 .27769457331927827002810119567456810e-1L, -.67762371439822456447373550186163070e-2L,
//                 .17238755142247705209823876688592170e-2L, -.44817699064252933515310345718960928e-3L,
//                 .11793660000155572716272710617753373e-3L, -.31253894280980134452125172274246963e-4L,
//                 .83173997012173283398932708991137488e-5L, -.22191427643780045431149221890172210e-5L,
//                 .59302266729329346291029599913617915e-6L, -.15863051191470655433559920279603632e-6L,
//                 .42459203983193603241777510648681429e-7L, -.11369129616951114238848106591780146e-7L,
//                 .304502217295931698401459168423403510e-8L, -.81568455080753152802915013641723686e-9L,
//                 .21852324749975455125936715817306383e-9L, -.58546491441689515680751900276454407e-10L,
//                 .15686348450871204869813586459513648e-10L, -.42029496273143231373796179302482033e-11L,
//                 .11261435719264907097227520956710754e-11L, -.30174353636860279765375177200637590e-12L,
//                 .80850955256389526647406571868193768e-13L, -.21663779809421233144009565199997351e-13L,
//                 .58047634271339391495076374966835526e-14L, -.15553767189204733561108869588173845e-14L,
//                 .41676108598040807753707828039353330e-15L, -.11167065064221317094734023242188463e-15L } ;
//
//             register double Tn_1 = 1.0L ;/* T_{n-1}(x),     started at n=1 */
//             register double Tn = x-2.0L ;/* T_{n}(x) , s   tarted at n=1 */
//             register double resul = Kncoe[0] + Kncoe[1]*Tn ;
//             x -= 2.0L ;
//             for(int n = 2 ; n < sizeof(Kncoe)/sizeof(double) ;n++){
//                 const double Tn1 = 2.0L * x * Tn - Tn_1 ;//Chebyshev recursion, Eq. 22.7.4 Abramowitz-Stegun
//                 resul += Kncoe[n]*Tn1 ;
//                 Tn_1 = Tn ;
//                 Tn = Tn1 ;                                  
//             }
//             return resul ;
//     }
// }

__device__ double polyeval(const double x, const double coef[], const int n){
  double ans;
  int i;
  const double *p;
  p = coef;
  ans = *p++;
  i = n;
  do 
  ans = ans* x + *p++;
  while (--i);
  return (ans);
}

__device__ double digamma_imp_1_2(double x){
  /*
  Rational approximation on [1,2] taken from Boost
  Form used: digamma(x) = (x-root)*(Y+R(x-1))
  */
  double r, g;

  const float Y = 0.99558162689208984f;

  const double root1 = 1569415565.0 / 1073741824.0;
  const double root2 = (381566830.0 / 1073741824.0) / 1073741824.0;
  const double root3 = 0.9016312093258695918615325266959189453125e-19;

  const double P[] = {-0.0020713321167745952, -0.045251321448739056,
               -0.28919126444774784,   -0.65031853770896507,
               -0.32555031186804491,   0.25479851061131551};
  const double Q[] = {-0.55789841321675513e-6,
               0.0021284987017821144,
               0.054151797245674225,
               0.43593529692665969,
               1.4606242909763515,
               2.0767117023730469,
               1.0};
  g = x - root1;
  g -= root2;
  g -= root3;
  r = polyeval(x - 1.0, P, 5) / polyeval(x - 1.0, Q, 6);

  return g * Y + g * r;
}

__device__ double psi_asy(double x){
  double y, z;
  const double A[] = {8.33333333333333333333E-2, -2.10927960927960927961E-2,
             7.57575757575757575758E-3, -4.16666666666666666667E-3,
             3.96825396825396825397E-3, -8.33333333333333333333E-3,
             8.33333333333333333333E-2};

  if (x < 1.0e17) {
    z = 1.0 / (x * x);
    y = z * polyeval(z, A, 6);
  } else {
    y = 0.0;
  }

  return log(x) - (0.5 / x) - y;
}

__device__ double digamma(double x){
  double y = 0.0;
  double q, r;
  int i, n;
// Checking poles
  if (isnan(x)) {
    return 1.0e9;
  } else if (x == 0) {
    return 1.0e9;
  } else if (x < 0.0) {

    // argument reduction before evaluating tan(pi * x) 
    // q int part of x and r fractional part
    r = modf(x, &q);
    if (r == 0.0) {
      return 1.0e9;
    }
    y = -M_PI / tan(M_PI * r);
    x = 1.0 - x;
  }

  //Positive integers up to 10
  if ((x <= 10.0) && (x == floor(x))) {
      n = (int)x;
      for (i = 1; i < n; i++) {
        y += 1.0 / i;
      }
      y -= M_EULER;
      return y;
    }

  // use the recurrence relation to move x into [1, 2] 
  if (x < 1.0) {
    y -= 1.0 / x;
    x += 1.0;
  } else if (x < 10.0) {
    while (x > 2.0) {
      x -= 1.0;
      y += 1.0 / x;
    }
  }
  if ((1.0 <= x) && (x <= 2.0)) {
    y += digamma_imp_1_2(x);
    return y;
  }

  //x is large, use the asymptotic series
  y += psi_asy(x);
  return y;
}


  //TODO: Old version of Hypergeometric 2F1 -> correct when you are not close to
  //-1 () or +1 (not physical for us)
  /*
  __device__ double hypgeo2F1_Bateman_2_10_eq19(int a, int b, int  c, double z){
      // Credits Amine Boussejra
      // Analytic continuation taken from Bateman. Conditions :
      // abs(arg(-z)) < pi , abs(b-a) = m, c = a +m + l, l positive integer.
      // if z is close to -1 (left or right), the result cannot be completely trusted. 
      // Precision is achieved to 10-3.

      int m;
      int flag = 0 ;
      if((b-a)>0) m = b - a ;
      else { m = a - b ; int foo = a; a = b; b = foo;}    // Now b-a =m>0
      int l = c - a - m;
          double res=0;
          // 3 sums :
          double prefactor1, prefactor2, prefactor3 ;
          prefactor1 = pow(-1.,m+l)*pow(-z,-a-m);
          prefactor2 = pow(-z,-a-m)/factorial(l+m-1);
          prefactor3 = pow(-z,-a);
          double sum1 = 0, sum2 = 0, sum3 = 0;
          double sum1_nm1 =0, sum1_nm2 = 0, sum1_nm3 = 0; // only 2 is ok ?
          for (int n = l; n< n+1; n++){
            sum1_nm3 = sum1_nm2;
            sum1_nm2 = sum1_nm1;
            sum1_nm1 = sum1;
            sum1 += Pochhammer(a,n+m)*factorial(n-l)/(factorial(n+m)*factorial(n)) * pow(z,-n);
            if ((n>3) && (isnan(sum1))) {
                printf("1.0e9 occured : n = %d \t sum1_nm1 = %.6e\n", n, sum1_nm1);
                flag = 1 ;
                break;
            }
            else if( (n>3) && ( fabs((sum1 - sum1_nm1)/sum1)<EPS_PREC && 
                  fabs((sum1_nm1 - sum1_nm2)/sum1_nm1)<EPS_PREC && 
                  fabs((sum1_nm2 - sum1_nm3)/sum1_nm2)<EPS_PREC) )break;

            else if(( n > 100 ) && ( fabs((sum1 - sum1_nm1)/sum1)<EPS_PREC 
                  && fabs((sum1_nm1 - sum1_nm2)/sum1_nm1)<1e-4) )break;
          
            else if(( n > 300) && ( fabs((sum1 - sum1_nm1)/sum1)<1e-4) )break;
            else if(n>700 && fabs((sum1 - sum1_nm1)/sum1)>1e-3){ flag = 1 ; break; } //Limit for convergence
        }
        if(l!=0){
            for (int n = 0; n < l; n++)
                sum2+= Pochhammer(a,n+m)*Pochhammer(1-m-l,n+m)*pow(z,-n)/(factorial(n)*factorial(n+m)) 
                  * (log(-z)+digamma(1+m+n)+digamma(1+n)-digamma(a+m+n)-digamma(l-n));
        }
        if(m!=0){
            for (int n = 0; n < m; n++)
                sum3+= factorial(m-n-1)*Pochhammer(a,n)*pow(z,-n)/(factorial(m+l-n-1)*factorial(n));
        }
        res = prefactor1*sum1 + prefactor2*sum2 + prefactor3*sum3 ;
        if(flag) {
            return 1.0e9;
        }
        return (tgammaf(a+m+l)/tgammaf(a+m)) * res; 
  }
  */

  /*
  __device__ double hypgeo2F1_radius1(double a, double b, double c, double z){
    // Based on Series expansion
    // Warning: For |z| close to 1 (x->0) cannot be completely trusted
    // This is not a problem for this project
    double term = a * b * z / c;
    double result = 1.0 + term;
    int n = 1;

    do {
        a++, b++, c++, n++;
        term *= a*b*z /c /n;
        result += term;
    } while(abs(term)>EPS_PREC);

    return result;
  }
  */

  //TODO: Old 2f1 deprecated
  /*
  __device__ double hypgeo2F1(double a, double b, double c, double z){
      // Author Ramon Ruiz
      // a, b, c need to be integers and > 0
      // z>1. cannot happen, not coded
      // |z| < 1 coded using series expansion
      // z < -1 Coded using Bateman expresion -> Credit Amine Boussejra

      if ( (z>1.) || ( (a<0) || (b<0) || (c<0) ) ) return 1.0e9;
      else if( (z > -1.)&&(z<1.))  return hypgeo2F1_radius1(a,b,c,z);
      else if ( z <= -1 ){
          // Main problem of the integral z=-1 -> Mchar = \sqrt(2) M_sq
        int a1 = (int) a;
        int b1 = (int) b;
        int c1 = (int) c;
          return hypgeo2F1_Bateman_2_10_eq19(a1,b1,c1,z); 
      }
      else {
        return 1.0e9;
          }
      }
  */



  __device__ double hys2f1(double a, double b, double c, double x, double *loss) {
    double f, g, h, k, m, s, u, umax;
    // int i, ib, intflag = 0;
    int i, ib;

    if (fabs(b) > fabs(a)) {
      /* Ensure that |a| > |b| ... */
      f = b;
      b = a;
      a = f;
    }

    ib = round(b);

    if (fabs(b - ib) < EPS && ib <= 0 && fabs(b) < fabs(a)) {
      f = b;
      b = a;
      a = f;
    }

    //not needed
    // if ((fabs(a) > fabs(c) + 1 || intflag) && fabs(c - a) > 2 && fabs(a) > 2) {
    //   /* |a| >> |c| implies that large cancellation error is to be expected.
    //    *
    //    * We try to reduce it with the recurrence relations
    //    */
    //   return hyp2f1ra(a, b, c, x, loss);
    // }

    i = 0;
    umax = 0.0;
    f = a;
    g = b;
    h = c;
    s = 1.0;
    u = 1.0;
    k = 0.0;
    do {
      if (fabs(h) < EPS) {
        *loss = 1.0;
        return MAXNUM;
      }
      m = k + 1.0;
      u = u * ((f + k) * (g + k) * x / ((h + k) * m));
      s += u;
      k = fabs(u); /* remember largest term summed */
      if (k > umax)
      umax = k;
    k = m;
    if (++i > 10000) { /* should never happen */
      *loss = 1.0;
      return (s);
    }
  } while (s == 0 || fabs(u / s) > MACHEP);

  /* return estimated relative error */
  *loss = (MACHEP * umax) / fabs(s) + (MACHEP * i);

  return s;
}

__device__ double hyt2f1(double a, double b, double c, double x, double *loss) {
  double p, q, r, s, t, y, d, err, err1;
  double ax, id, d1, d2, e, y1;
  int i, aid;
  int ia, ib, neg_int_a = 0, neg_int_b = 0;

  ia = round(a);
  ib = round(b);

  if (a <= 0 && fabs(a - ia) < EPS) { /* a is a negative integer */
  neg_int_a = 1;
  }

  if (b <= 0 && fabs(b - ib) < EPS) { /* b is a negative integer */
  neg_int_b = 1;
  }

  err = 0.0;
  s = 1.0 - x;
  if (x < -0.5 && !(neg_int_a || neg_int_b)) {
  if (b > a) {
    y = pow(s, -a) * hys2f1(a, c - b, c, -x / s, &err);
  } else {
    y = pow(s, -b) * hys2f1(c - a, b, c, -x / s, &err);
  }
  // goto done;
  *loss = err;
  return (y);
  }

  d = c - a - b;
  id = round(d); /* nearest integer to d */

  if (x > 0.9 && !(neg_int_a || neg_int_b)) {
  if (fabs(d - id) > EPS) {
    y = hys2f1(a, b, c, x, &err);
    if (err < ETHRESH) {
      *loss = err;
      return (y);
    }
    q = hys2f1(a, b, 1.0 - d, s, &err);
    q *= tgammaf(d) / (tgammaf(c - a) * tgammaf(c - b));
    r = pow(s, d) * hys2f1(c - a, c - b, d + 1.0, s, &err1);
    r *= tgammaf(-d) / (tgammaf(a) * tgammaf(b));
    y = q + r;

    q = fabs(q); /* estimate cancellation error */
    r = fabs(r);
    if (q > r)
      r = q;
    err += err1 + (MACHEP * r) / y;

    y *= tgammaf(c);
    *loss = err;
    return (y);

  } else {
    if (id >= 0.0) {
      e = d;
      d1 = d;
      d2 = 0.0;
      aid = id;
    } else {
      e = -d;
      d1 = 0.0;
      d2 = d;
      aid = -id;
    }

    ax = log(s);

    y = digamma(1.0) + digamma(1.0 + e) - digamma(a + d1) - digamma(b + d1) -
        ax;
    y /= tgammaf(e + 1.0);

    p = (a + d1) * (b + d1) * s / tgammaf(e + 2.0); 
    t = 1.0;
    do {
      r = digamma(1.0 + t) + digamma(1.0 + t + e) - digamma(a + t + d1) -
          digamma(b + t + d1) - ax;
      q = p * r;
      y += q;
      p *= s * (a + t + d1) / (t + 1.0);
      p *= (b + t + d1) / (t + 1.0 + e);
      t += 1.0;
      if (t > 10000) { /* should never happen */
        *loss = 1.0;
        return 1.0e9;
      }
    } while (y == 0 || fabs(q / y) > EPS);

    if (id == 0.0) {
      y *= tgammaf(c) / (tgammaf(a) * tgammaf(b));
      *loss = err;
      return (y);
    }

    y1 = 1.0;

    if (aid == 1)
    {
    p = tgammaf(c);
    y1 *= tgammaf(e) * p / (tgammaf(a + d1) * tgammaf(b + d1));

    y *= p / (tgammaf(a + d2) * tgammaf(b + d2));
    if ((aid & 1) != 0)
      y = -y;

    q = pow(s, id); /* s to the id power */
    if (id > 0.0)
      y *= q;
    else
      y1 *= q;

    y += y1;
      *loss = err;
      return (y);

    }

    t = 0.0;
    p = 1.0;
    for (i = 1; i < aid; i++) {
      r = 1.0 - e + t;
      p *= s * (a + t + d2) * (b + t + d2) / r;
      t += 1.0;
      p /= t;
      y1 += p;
    }
  }
  }

  y = hys2f1(a, b, c, x, &err);
  *loss = err;
  return (y);
}

__device__ double hyp2f1(double a, double b, double c, double x){
  // double d, d1, d2, e;
  double d;
  double p, q, r, s, y, ax;
  double ia, ib, ic, id, err;
  double t1;
  // int i, aid;
  int neg_int_a = 0, neg_int_b = 0;
  int neg_int_ca_or_cb = 0;

  err = 0.0;
  ax = fabs(x);
  s = 1.0 - x;
  ia = round(a); /* nearest integer to a */
  ib = round(b);

  if (x == 0.0) {
    return 1.0;
  }

  d = c - a - b;
  id = round(d);
  if ((a == 0 || b == 0) && c != 0) {
    return 1.0;
  }

  if (a <= 0 && fabs(a - ia) < EPS) { /* a is a negative integer */
    neg_int_a = 1;
  }

  if (b <= 0 && fabs(b - ib) < EPS) { /* b is a negative integer */
    neg_int_b = 1;
  }

  if (d <= -1 && !(fabs(d - id) > EPS && s < 0) && !(neg_int_a || neg_int_b)) {
    return pow(s, d) * hyt2f1(c - a, c - b, c, x, &err);
  }
  if (d <= 0 && x == 1 && !(neg_int_a || neg_int_b))
    return MAXNUM;

  if (ax < 1.0 || x == -1.0) {
    /* 2F1(a,b;b;x) = (1-x)**(-a) */
    if (fabs(b - c) < EPS) { /* b = c */
      y = pow(s, -a);       /* s to the -a power */
      return y;
    }
    if (fabs(a - c) < EPS) { /* a = c */
      y = pow(s, -b);       /* s to the -b power */
      return y;
    }
  }

  if (c <= 0.0) {
    ic = round(c);            /* nearest integer to c */
    if (fabs(c - ic) < EPS) { /* c is a negative integer */
      /* check if termination before explosion */
      if (neg_int_a && (ia > ic))
        return hyt2f1(a, b, c, x, &err);
      if (neg_int_b && (ib > ic))
        return hyt2f1(a, b, c, x, &err);
      return MAXNUM;
    }
  }

  if (neg_int_a || neg_int_b) /* function is a polynomial */
    return hyt2f1(a, b, c, x, &err);

  t1 = fabs(b - a);
  if (x < -2.0 && fabs(t1 - round(t1)) > EPS) {
    p = hyt2f1(a, 1 - c + a, 1 - b + a, 1.0 / x, &err);
    q = hyt2f1(b, 1 - c + b, 1 - a + b, 1.0 / x, &err);
    p *= pow(-x, -a);
    q *= pow(-x, -b);
    t1 = tgammaf(c);
    s = t1 * tgammaf(b - a) / (tgammaf(b) * tgammaf(c - a));
    y = t1 * tgammaf(a - b) / (tgammaf(a) * tgammaf(c - b));
    return s * p + y * q;
  } else if (x < -1.0) {
    if (fabs(a) < fabs(b)) {
      return pow(s, -a) * hyt2f1(a, c - b, c, x / (x - 1), &err);
    } else {
      return pow(s, -b) * hyt2f1(b, c - a, c, x / (x - 1), &err);
    }
  }

  if (ax > 1.0) /* series diverges  */
    return MAXNUM;

  p = c - a;
  ia = round(p);                           /* nearest integer to c-a */
  if ((ia <= 0.0) && (fabs(p - ia) < EPS)) /* negative int c - a */
    neg_int_ca_or_cb = 1;

  r = c - b;
  ib = round(r);                           /* nearest integer to c-b */
  if ((ib <= 0.0) && (fabs(r - ib) < EPS)) /* negative int c - b */
    neg_int_ca_or_cb = 1;

  id = round(d); /* nearest integer to d */
  q = fabs(d - id);

  if (fabs(ax - 1.0) < EPS) { /* |x| == 1.0 */
    if (x > 0.0) {
      if (neg_int_ca_or_cb) {
        if (d >= 0.0)
          return pow(s, d) * hys2f1(c - a, c - b, c, x, &err);
        else
          return MAXNUM;
      }
      if (d <= 0.0)
        return MAXNUM;
      y = tgammaf(c) * tgammaf(d) / (tgammaf(p) * tgammaf(r));
      return y;
    }
    if (d <= -1.0)
      return MAXNUM;
  }
  if (d < 0.0) {
    /* Try the power series first */
    y = hyt2f1(a, b, c, x, &err);
    // if (err < ETHRESH)
      return y;
    /* Apply the recurrence if power series fails */
    // err = 0.0;
    // aid = 2 - id;
    // e = c + aid;
    // d2 = hyp2f1(a, b, e, x);
    // d1 = hyp2f1(a, b, e + 1.0, x);
    // q = a + b + 1.0;
    // for (i = 0; i < aid; i++) {
      // r = e - 1.0;
      // y = (e * (r - (2.0 * e - q) * x) * d2 + (e - a) * (e - b) * x * d1) /
          // (e * r * s);
      // e = r;
      // d1 = d2;
      // d2 = y;
    // }
    // return y;
  }

  if (neg_int_ca_or_cb) {
    /* negative integer c-a or c-b */
    return pow(s, d) * hys2f1(c - a, c - b, c, x, &err);
  }
  return hyt2f1(a, b, c, x, &err);
}







__device__ double P_loop(int i, int j, int k, double a, double b){
    // Author Ramon Ruiz
    // Credits Amine Boussejra
    // Computes the P_ijk(a,b) loop integrals using hypgeo2F1
    // Based on 2201.04659 Appendix B
    // Warning: Known issuer -> z = -1 region (2F1)
    double result;
    if ( ( (i<0) || (j<0) || (k<0) ) || ( (a<0) || (b<0) ) ) return 0.;

    if ( k!=1 )
    {
          if ( (i==0) && (j==3) && (k==2) && (fabs(a-b)>loop_w) ){
            result =  0.;
          }
          else if(fabs(a-b)>loop_w ){ 
          result = (1./((1-k)*(a-b))) * Beta_E(i,j+1)*( hyp2f1(k-1,i,i+j+1,1-a) - hyp2f1(k-1,i,i+j+1,1-b) );}
          else if (fabs(a-b)<=loop_w) result = Beta_E(i+1,j+1)*hyp2f1(k,i+1,i+j+2,1-a);
          else{
              result= 0.0;
          }
    }
    else //K==1 easy
    {
          if ( (i==1) && (j==1) && (fabs(a-b)>loop_w) )
          {//P_111(a,b)
            if (fabs(a-1.)<=loop_w)
            {//P_111(1,b)
              result = (-0.25 + 1.*b - 0.75*pow(b,2) + 0.5*pow(b,2)*log(b))/pow(-1. + b,3);
            }
            else if (fabs(b-1.)<= loop_w)
            {//P_111(a,1)
              result = (-0.25 + 1.*a - 0.75*pow(a,2) + 0.5*pow(a,2)*log(a))/pow(-1. + a,3);
            }
            else
            { //P_111(a,b)
            result= (1.0/2.0)*(pow(a - 1, 2)*(a - b)*pow(b - 1, 2)
              + pow(a - 1, 2)*pow(b - 1, 2)*(log(a) - log(b))*(a*b - a - b + 1) 
              + pow(a - 1, 2)*(2*b - 1)*(-log(b/fabs(b - 1)) + log(1.0/fabs(b - 1)))*(a*b - a - b + 1) + 
              (2*a - 1)*pow(b - 1, 2)*(log(a/fabs(a - 1)) - 
                log(1.0/fabs(a - 1)))*(a*b - a - b + 1))/(pow(a - 1, 2)*(a - b)*pow(b - 1, 2)*(a*b - a - b + 1));
            }
          }
          else if( (i==1) && (j==1) && (fabs(a-b)<=loop_w) )
          {//P_111(a,a)
            if (fabs(a-1.)<=loop_w){//P_111(1,1)
              result = 1./6.;
              }
            else{ 
              result= (1.0/2.0)*(pow(a, 2) - 2*a*log(a) - 1)/(pow(a, 3) - 3.*pow(a, 2) + 3.*a - 1);
                }
            }

          else if ( (i==0) && (j==2 ) && (fabs(a-b)<=loop_w)) 
          {//P_021(a,a)
            if(fabs(a-1.)>loop_w){
              result= (1.0/2.0)*(2.*pow(a, 2)*log(a) - 3.*pow(a, 2) + 4.*a - 1)/(pow(a, 3) - 3.*pow(a, 2) + 3.*a - 1);
              }
            else{ //P_021(1,1)
              result = 0.;
               }
          }
          else if ( (i==0) && (j==2) && (fabs(a-b)>loop_w) )
          {
              result = 0;
          }
          else{
            result= 0.;
          }
    }
  return result;
}

__device__ double f_loop(double a, double b, double c){
  double res;
  if(a<EPS_PREC|b<EPS_PREC|c<EPS_PREC) {return 1.0e9; }

  //TODO: Seems to work -> Further test
  if(a<EPS_PREC||fabs(a-1)<EPS_PREC||fabs(a-c)<EPS_PREC||fabs(a-c+1)<EPS_PREC){
    return f_loop(a+2.*EPS_PREC, b,c);
  }
  if(b<EPS_PREC||fabs(b-1)<EPS_PREC||fabs(b-c)<EPS_PREC||fabs(b-c+1)<EPS_PREC){
    return f_loop(a,b+2.*EPS_PREC,c);
  } 
  if ( fabs(c-1)<EPS_PREC||fabs(c-a)<EPS_PREC||fabs(c-b)<EPS_PREC||fabs(c-b-1)<EPS_PREC||fabs(c-a-1)<EPS_PREC ){
    return f_loop(a,b,c+2.*EPS_PREC);
    }
  if (fabs(a-b)<EPS_PREC){
    res =  0.25*(a*pow(a - 1, 3)*pow(a - c, 2)*pow(c - 1, 2)*(-a + 2.*c - 1)*(a*c - a - c + 1) 
                + pow(c, 2)*pow(a - 1, 3)*log(a/c)*(a*c - a - c + 1)*(-pow(a, 3)*c + pow(a, 3) 
                + pow(a, 2)*pow(c, 2) + pow(a, 2)*c - 2.*pow(a, 2) - 2.*a*pow(c, 2) + a*c + a 
                + pow(c, 2) - c) + pow(a - 1, 3)*pow(a - c, 2)*pow(c - 1, 2)*(pow(a, 3)*c - pow(a, 3) 
                - pow(a, 2)*pow(c, 2) - pow(a, 2)*c + 2.*pow(a, 2) + 2.*a*pow(c, 2) - a*c - a - pow(c, 2) + c) 
                + pow(a - c, 2)*log(a)*(-2.*a*c + a + 1)*(a*c - a - c + 1)*(-pow(a, 3)*c + pow(a, 3) 
                + pow(a, 2)*pow(c, 2) + pow(a, 2)*c - 2.*pow(a, 2) - 2.*a*pow(c, 2) + a*c + a 
                + pow(c, 2) - c))/(pow(a - 1, 3)*pow(a - c, 2)*pow(c - 1, 2)*(a*c - a - c + 1)*(-pow(a, 3)*c 
                + pow(a, 3) + pow(a, 2)*pow(c, 2) + pow(a, 2)*c - 2.*pow(a, 2) - 2.*a*pow(c, 2) 
                + a*c + a + pow(c, 2) - c));
  }
  else {
    res = (1.0/4.0)*(pow(a - 1, 2)*(a - c)*(-b*(b - 1)*(b - c)*(c - 1) 
        +  pow(c, 2)*pow(b - 1, 2)*(log(c/b)) - (b - c)*log(b)*(-2.*b*c + b + c)) 
        + pow(b - 1, 2)*(b - c)*(a*(a - 1)*(a - c)*(c - 1) + pow(c, 2)*pow(a - 1, 2)*(log(a/c)) 
        + (a - c)*(log(a))*(-2*a*c + a + c)))/(pow(a - 1, 2)*pow(a - b, 2)*(a - c)*pow(b - 1, 2)*(b - c)*pow(c - 1, 2)) ;
  }

  return res;
}



//From here based on 1205.1500 duLR_23
//Implementation Ramon Ruiz

//Preliminaries

__device__ double c_term0(double m1, double m2, double m3){
    return (m1 * log(m1/pow(MU_0,2.))) / ((m1 - m2)*(m1 - m3));
}

__device__ double c_term2(double m1, double m2, double m3){ 
    return pow(m1,2.) * log(m1/pow(MU_0,2.)) / ((m1 - m2)*(m1 - m3)) ;   
}

__device__ double d_term2(double m1, double m2, double m3, double m4){
    if( (fabs(m1 - m2) > EPS_PREC) && (fabs(m1 - m3) > EPS_PREC) && (fabs(m1-m4 ) > EPS_PREC)){
        double res= pow(m1,2.) * log(m1/pow(MU_0,2.)) / ((m1 - m2)*(m1 - m3)*(m1 - m4));
        return res;
        }
    
    else{
      return 1.0e9;
    }
}


__device__ double c_0(double m1, double m2, double m3){
    if (fabs(m2-m3)>EPS_PREC && fabs(m1-m3)>EPS_PREC){
        double res = -1.*( c_term0(m1, m2, m3) + c_term0(m2, m1, m3) + c_term0(m3, m2, m1));
        return res;
    }
    else if ( fabs(m1-m2) < EPS_PREC && fabs(m1-m3)>EPS_PREC ){
        return -1.*(m2 - m3*log(m2) + m3*log(m3) - m3)/(pow(m2, 2) - 2*m2*m3 + pow(m3, 2));
    }
    else if ( fabs(m2 - m3) < EPS_PREC && fabs(m1 - m3)>EPS_PREC)  {
        return -1.* (m1*log(m1) - m1*log(m2) - m1 + m2 )/(pow(m1, 2) - 2*m1*m3 + pow(m3, 2));
    }  
    else if(fabs(m1-m2)<EPS_PREC && fabs(m2-m3)<EPS_PREC) return -1./(2.*m1);
    else{
      return 1.0e9;
    }
}

__device__ double c_2( double m1, double m2, double m3 ){
    if( fabs(m2 - m3) > EPS_PREC && fabs(m1-m3)>EPS_PREC){
       double res = (3./8.) - (1./4) *( c_term2(m1, m2, m3) + c_term2(m2, m1, m3) + c_term2(m3, m2, m1) ); 
       return res;
    }
    else if(fabs(m1)<EPS_PREC||fabs(m2)<EPS_PREC||fabs(m3)<EPS_PREC){
        return 1.0e9;
    }
    else if (fabs(m2 - m3)< EPS_PREC && fabs(m1-m3)>EPS_PREC){
        double res = c_2(m1, m2+1.2*EPS_PREC,m3);
        return res; 
    }
    else if(fabs(m1-m3)<EPS_PREC && fabs(m2-m3)>EPS_PREC){
        double res = c_2(m1+1.2*EPS_PREC, m2, m3);
        return res;
    }
    else if(fabs(m1-m2)<EPS_PREC && fabs(m2-m3)>EPS_PREC){
        double res = c_2(m1+1.2*EPS_PREC, m2, m3);
        return res ;
    }
    else if (fabs(m1-m2)<EPS_PREC && fabs(m1-m3)<EPS_PREC){
        double res = 1.0*(0.375*m1 - 0.25)/m1;
        return res;
    }
    else{
      return 1.0e9;
    }
}

__device__ double f_7(double x){
    if( fabs(x-1.) > EPS_PREC ){
        return (1.0/6.0)*(43*pow(x, 2) - 101*x + 52)/pow(1 - x, 3) + (2*pow(x, 3) - 9*x + 6)*log(x)/pow(1 - x, 4); 
    }
    else if (fabs(x-1)<=EPS_PREC){
        return -7./4. ;
    }
    else{
      return 1.0e9;
    }
}

__device__ double d_2(double m1, double m2, double m3, double m4){
    
    // All equal
    if (fabs(m1 - m2 - m3 - m4)<EPS_PREC) return -1./4.* 1/(3*m3);

    // Three out of four equal
    else if (fabs(m1-m2-m3)<EPS_PREC) return d_2(1.2*EPS_PREC+m1, -1.2*EPS_PREC+m2,m3, m4); // 1,2,3 
    else if (fabs(m1-m2-m4)<EPS_PREC) return d_2(1.2*EPS_PREC+m1,m2,m3,-1.2*EPS_PREC+m4); // 1,2,4
    else if (fabs(m1-m3-m4)<EPS_PREC) return d_2(1.2*EPS_PREC+m1,m2, -1.2*EPS_PREC+m3,m4); // 1,3,4
    else if (fabs(m2-m3-m4)<EPS_PREC) return d_2(m1, m2,1.2*EPS_PREC+m3, -1.2*EPS_PREC+m4); // 2,3,4

    // Two
    else if ( fabs(m1 - m2)<EPS_PREC && fabs(m1-m3)>EPS_PREC && fabs(m1-m4)>EPS_PREC ) // 1,2
        return (-1./4.)*(m2*(2.*log(m2/pow(MU_0,2)) + 1)/((m2-m3)*(m2-m4)) - d_term2(m2, m3, m3, m4) - d_term2(m2,m3, m4, m4) + d_term2(m3, m2,m2,m4) + d_term2(m4,m2, m3, m2) );

    else if( fabs(m1-m2)>EPS_PREC && fabs(m1-m3)<EPS_PREC && fabs(m1-m4)>EPS_PREC ) // 1,3
        return (-1./4.)*(m3*(2.*log(m3/pow(MU_0,2)) + 1)/((m3-m2)*(m3-m4)) - d_term2(m3, m2, m2, m4) - d_term2(m3, m2, m4, m4) + d_term2(m2, m3, m3, m4) + d_term2(m4, m2, m3, m3) );
    
    else if( fabs(m1-m2)>EPS_PREC && fabs(m1-m3)> EPS_PREC && fabs(m1-m4)<EPS_PREC ) // 1,4
        return (-1./4.)*(m4*(2.*log(m4/pow(MU_0,2))+1)/((m4-m2)*(m4-m3)) - d_term2(m4,m2,m2,m3) - d_term2(m4, m2,m3,m3) + d_term2(m2, m4, m3, m4) + d_term2(m3, m2, m4, m4) );

    else if( fabs(m2-m3)<EPS_PREC && fabs(m2-m4) >EPS_PREC && fabs(m2-m1)>EPS_PREC ) // 2,3
        return (-1./4.)*(m3*(2.*log(m3/pow(MU_0,2))+1)/((m3-m1)*(m3-m4)) - d_term2(m3,m1, m1, m4) - d_term2(m3,m4, m4, m1) + d_term2(m1,m3, m3, m4) + d_term2(m4,m3, m3, m1) );
    
    else if( fabs(m2-m4)<EPS_PREC && fabs(m2-m3)>EPS_PREC && fabs(m2-m1)>EPS_PREC) // 2,4
        return (-1./4.)*(m4*(2.*log(m4/pow(MU_0,2))+1)/((m4-m1)*(m4-m3)) - d_term2(m4,m1, m1, m3) - d_term2(m4,m3, m3, m1) + d_term2(m1,m4, m4, m3) + d_term2(m3,m4, m4, m1) );

    else if (fabs(m3-m4)<EPS_PREC)// 3,4 (should be the last possible pair)
        return (-1./4.)*(m4*(2.*log(m4/pow(MU_0,2))+1)/((m4-m1)*(m4-m2)) - d_term2(m4,m1, m1, m2) - d_term2(m4,m2, m2, m1) + d_term2(m1,m4, m4, m2) + d_term2(m2,m4, m4, m1) );

    else if ( fabs(m1-m2)>EPS_PREC &&  fabs(m1-m3)>EPS_PREC &&  fabs(m1-m4)>EPS_PREC && fabs(m2-m3)>EPS_PREC && fabs(m2-m4)>EPS_PREC && fabs(m3-m4)>EPS_PREC ){
        
        double res = (-1./4.)*(d_term2(m1, m2, m3, m4) + d_term2(m2, m1, m3, m4) + d_term2(m3, m2, m1, m4) + d_term2(m4, m2, m3, m1));
        return res;
    }
    else{
      return 1.0e9;
    }
}

//Main Funcitons for HW loops
//Author Ramón Ángel 
__device__ void F_Zp( 
                        pycuda::complex<double> U[2][2] , 
                        pycuda::complex<double> V[2][2],
                        double x[], 
                        double x_tr,
                        pycuda::complex<double> result)
{
  pycuda::complex<double> x_av = (1./6.)*(5+x_tr);
  pycuda::complex<double> sum; 
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
          sum += V[j][0]*pycuda::conj(V[i][1])*pycuda::conj(U[j][0])*U[i][0]*sqrt(x[i]*x[j])
                 *((c_0(x_tr, x[i], x[j]) - c_0(1., x[i], x[j]))/(x_tr - 1.))
                 -(2.*V[i][0]*pycuda::conj(V[j][0])* ( (c_2(x_tr, x[i], x[j])
                 -c_2(1., x[i], x[j]))/(x_tr - 1.)) ) +( 2.*Krodelta(i,j)
                *((c_2(x[j], 1., x_tr) - c_2(x[j], 1., 1.))/(x_tr - 1.) ));
        }

    }
    result = x_av * sum ;
}

__device__ void F_gammap(
                        pycuda::complex<double> U[2][2] ,
                        pycuda::complex<double> V[2][2],
                        double x[],
                        double x_tr,
                        pycuda::complex<double> result){
   
    pycuda::complex<double> x_av = (1./6.)*(5. + x_tr);
    pycuda::complex<double> sum;

    for (int i = 0; i < 2; i++) {
        sum += V[i][0]*pycuda::conj(V[i][1])*x_tr*( (x[i]/x_tr)*f_7(x[i]/x_tr)
                - x[i]*f_7(x[i]) )/( -x[i]+x[i]/x_tr ) ;
    }
    result = (1./9.)*x_av*sum;
}

__device__ void F_box(
                        pycuda::complex<double> U[2][2],
                        pycuda::complex<double> V[2][2],
                        double x[],
                        double x_tr,
                        double x_snu,
                        pycuda::complex<double> result){

    pycuda::complex<double> x_av = (1./6.)*(5.+x_tr);
    pycuda::complex<double> sum;
   
    // if( fabs(x_tr-1)<EPS_PREC) return F_box(param, x, x_tr-1.5*EPS_PREC,x_snu);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
           
            sum += V[i][0] * pycuda::conj(V[i][1]) * pow(pycuda::abs(V[j][0]),2.) *
              ( (d_2(x[i], x[j], x_tr, x_snu) - d_2(x[i], x[j], 1., x_snu))/(x_tr - 1.) );
        }
    }
    result = 4.*x_av*sum;
}


__device__ double f_1(double x){
    if( fabs(x) < EPS_PREC){
        return -7./6. ;
    }
    else if (fabs(x-1) < EPS_PREC){
        return -5./12. ;
    }
    else{
        return (-7. + 5.*x + 8.*pow(x,2.))/(6.*pow(1.-x,3.)) -
                  log(x)*(2.*x-3.*pow(x,2.))/(pow(1.-x,4.));
    }
}




