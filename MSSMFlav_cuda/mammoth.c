// Diego Martinez Santos, Miriam Lucio, Isabel Suarez, Ramón Ángel Ruiz Fernández
// Based mostly on Teppei Kitahara's notes, plus 0909.1333
//WR: Warning Ramon
/*-------------------------------------------------------------------*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pycuda-complex.hpp>
#include "loops.c"
#include "masses.c"

#define g3 1.22//param->g3_Q
#define GF 1.1663787e-05 // GeV^(-2)
#define mW 80.379 //Updated RR (TeVatron not included)
#define MZ2 8315. //Updated RR
#define mt 162.6 //updated RR
#define m_b 4.18 //updated RR
#define m_s .0934 //updated RR
#define m_u .00216 // updated RR
#define m_c 1.27 // updated RR
#define m_d .00467  //updated RR
#define f_Bs .2303 //updated RR
// #define f_Bd .278 //updated RR
#define f_Bd .2303/1.207 //updated fBs*fBdfBs
#define fBsfBd 1.207
#define f_K .1557 //updated RR
#define f_pi .1307
#define f_D .205


#define SW2 0.2334  // 2201.04659.pdf
// #define inv_alpha_MZ 128.9  

#define m_Bs 5.36692 //updated RR
#define m_Bd 5.2797 //updated RR
#define m_Bu 5.27934 //updated RR
#define m_D 1.86483
#define m_K 0.49761
//MKp? is not mK+ WR
#define m_Kp 0.4667
#define m_mu .1057 //updated RR
#define m_pi .1396 //updated RR

#define tau_Bd 2279635258358.6626 //update 
#define tau_Bs 2279635258358.6626 //update
#define tau_KL 7.775075987841946e+16
#define tau_KS 136018237082066.88


#define Y0_xt 0.962965584213785//From Isabel's Inami-Lim
#define YNL 0.000324310972212118 //From Isabel's Inami-Lim

// #define B_1s 0.952// https://arxiv.org/pdf/1602.03560.pdf Table XV
// #define B_2s 0.806
// #define B_3s 1.10
// #define B_4s 1.022
// #define B_5s 0.943

// #define B_1d 0.913
// #define B_2d .761
// #define B_3d 1.07
// #define B_4d 1.040
// #define B_5d 0.964

//RR: Updated s,d with 1909.11087  
#define B_1s 0.849// https://arxiv.org/pdf/1602.03560.pdf Table XV
#define B_2s 0.835
#define B_3s 0.854
#define B_4s 1.031
#define B_5s 0.959

#define B_1d 0.835
#define B_2d .791
#define B_3d 0.775
#define B_4d 1.063
#define B_5d 0.994

#define B_1k 0.525
#define B_2k 0.488
#define B_3k 0.743
#define B_4k 0.920
#define B_5k 0.707

#define B_1c 0.757
#define B_2c 0.653
#define B_3c 0.968
#define B_4c 0.915
#define B_5c 0.974


#define S0_xt 2.313
#define eta_B_hat 0.8393
// #define deltagamma12 0.087

//Not updated, check if there are updates (not needed for B meson paper)
#define wplus 4.53e-02
#define Omega_eff 0.148
#define ReA0 3.3201e-07
#define kappa_e 0.94
#define epsk_SM 2.12e-03
#define epek_SM 1.06e-04


///CKM elements RR 2023
#define Vud (0.974492833033007+0.*I)
#define Vus (0.224388097873252+0.*I)
#define Vub (0.00130632864370354 -0.00346314966500170*I)
#define Vcd (-0.224246232180289 -0.000148765496575128*I)
#define Vcs (0.973633022165068 +0.*I)
#define Vcb (0.0418610006030414 +0.*I)
#define Vtd (0.00812133824605011 -0.00337156553665105*I)
#define Vts (-0.0410866735816255 -0.000777776663096328*I)
#define Vtb (0.999116588155949+0.*I)
//Updated RR 2023 

//C7 shit
#define sgCA -1.

//m_b
#define C7_SM -0.292 //(https://indico.cern.ch/event/116810/contributions/73866/attachments/55142/79318/slides.pdf)
//2GeV

//From 1012.3167
//From 1606.00916
#define C7_2_SM -0.3189
#define C8_2_SM -0.1780
#define C7p_2_SM 0.
#define C8p_2_SM 0.


// #define CONSTR_C7 -0.02
// #define sCONSTR_C7 0.02

//Updated RR 2023: 2304.07330
#define CONSTR_C7 0.
#define CONSTR_sC7 0.02

// #define ALgg (0.539256 - 3.95632*I )*1e-11
// #define ASgg (-2.64983 + 1.14208*I )*1e-11

//Constraints

//Updated RR 2023
#define CONSTR_RDMs 0.996 //EXP/SM
#define CONSTR_sRDMs 0.044 //EXP/SM

//Updated RR 2023
#define CONSTR_RDMd 0.996/1.067  //EXP/SM 
#define CONSTR_sRDMd sqrt( pow(0.044/1.066,2) + pow(0.996*0.086,2)/pow(1.066,4))

//Updated RR 2023
#define CONSTR_Rds 0.937 //DMd/DMs EXP/SM
#define CONSTR_sRds .076

#define CONSTR_RDMK 1.
#define CONSTR_sRDMK 1.

#define CONSTR_DMD 3.3e-15
#define CONSTR_sDMD 3.3e-15

//RR Update 2023
#define CONSTR_Asl -0.13 // this is DM*asl/DG - (DM*asl/DG)_SM, updated
#define CONSTR_sAsl 0.60 // updated

//Updated RR 2023 (penguins negligible), preliminary HFLAV from -2Beta_s
#define CONSTR_dphis -0.0011
#define CONSTR_sdphis 0.0161

//From Asl -> arctan(Asl * dms /dgs) -> only dms is recalculated 
#define CONSTR_phis12  0.00404
#define CONSTR_sphis12 0.00070 


//Updated RR 2023 (penguins negligible), preliminary HFLAV from -2 Beta 
#define CONSTR_dphid  -0.0013
#define CONSTR_sdphid  0.041

//Paper B sector
// #define CONSTR_RepsK 1.05
// #define CONSTR_sRepsK .1

// #define CONSTR_epek 16.6e-04
// #define CONSTR_sepek_EXP 2.3e-04
// #define CONSTR_sepek_TH 5.07e-04

//Updated RR 2023
#define CONSTR_Rsmm .975
#define CONSTR_sRsmm .099

//Updated RR 2023
#define CONSTR_Rdmm 3.9
#define CONSTR_sRdmm 4.4

//No kaons only B sector
// #define CONSTR_KLmm_EXP 6.84e-9
// #define CONSTR_sKLmm_EXP .11e-9
// #define CONSTR_sKLmm_TH_plus .80e-09
// #define CONSTR_sKLmm_TH_minus 1.4e-09
// #define CONSTR_KMUNU 1.0004
// #define CONSTR_sKMUNU 0.0095
//Updated RR 2023
#define CONSTR_RTAUNU 1.26
#define CONSTR_sRTAUNU 0.3


//RR 2017 -> Update needed TODO
// #define CONSTR_RE_C9 -0.24
// #define sCONSTR_RE_C9 0.49
// #define CONSTR_IM_C9 -2.19
// #define sCONSTR_IM_C9 0.73

// #define CONSTR_RE_C10 0.66
// #define sCONSTR_RE_C10 0.60
// #define CONSTR_IM_C10 -0.97
// #define sCONSTR_IM_C10 0.51

//RR Update 2023
#define CONSTR_mH 125.25 // 2023 PDG
#define CONSTR_smH 2. //Dominated teo unc \approx 2 GeV 

//WR: Not found anyhting new
#define CONSTR_DACP 0.05 
#define CONSTR_sDACP 0.04

#define CONSTR_dn 0.0e-26  //e cm arXiv: 2001.11966 
#define CONSTR_sdn 1.1e-26

#define CONSTR_dHg 2.20e-30 // e cm arXiv: 1601.04339
#define CONSTR_sdHg 3.12e-30


//1702.02234 Re(C9), Im(C9), Re(C10), Im(C10)
// __device__ double C9_C10_nominal[4] = {-0.24, -2.19, 0.66, -0.97};
// Covariance inverse 1702.02234 private calculation RR
// __device__ double C9_C10_covinv[4][4] = {
//                                           {11.8676669, 2.81787741, -4.82009537, 6.978532},
//                                           {2.5890383, 5.74521396, 2.2278013, 5.61995217},
//                                           {-5.16145435, 2.05380467, 6.98814266, -0.67910774},
//                                           {6.8141275, 5.71547974, -0.41160529, 11.14888938}
//                                        };
// CRAP terms
/*__device__ double eps_stuff[10] = {  2.34296169e-01,  -1.30166756e-01,  -3.04491955e-01,
                                     2.31983358e-01,  -1.27310920e-01,  -2.15515400e-01,
                                     3.17510510e+01,   1.05993710e+02,  -5.83730567e-02,
                                     -2.14221970e-01};
*/
__device__ double Qepsp[10] = {0.345112, 0.132542, 0.0340124,-0.178558, 0.152483, 0.288073, 2.65313, 17.3046, 0.526475, 0.281154};
    
__device__ double U01fit[10][10] = {{1.3809792354675492,-0.6585927482687062,0.,0.,0.,0.,0.,0.,0.,0.},
                                    {-0.657931250288899,1.3829382911049737,0.,0.,0.,0.,0.,0.,0.,0.},
                                    {-0.026574589014316377,0.04460468307432865,1.3466986567767019,
                                     -0.5060575886608385,0.10836384648098668, 0.3508575125083425,0.018543769902154065,
                                     0.057947261329368026,-0.05576992372332773,0.056483473016109946},
                                    {0.03218621663756335,-0.0894963678165702,-0.7162658523641305,1.0763545959282423,
                                     -0.11403765079074088, -0.5953891192263856, -0.020807911521830827,-0.10027899436013457,
                                     0.10288419189226461,-0.08630312090486437},
                                    {0.006732161952786792,0.009856861079149834,0.06040207681352058,0.034208842934741246,
                                     0.8730074659001046, 0.39155889884432327,-0.00919203579802793,0.006705805426145523,
                                     -0.0023157060502377123, -0.0026851869926986316},
                                    {0.04266085146568667,-0.14453330511516918,-0.12203660635190902,-0.5807513319725348,
                                     1.1955563236389366, 3.8441537379362805,-0.044163822089577975,-0.23605795517237577,
                                     0.16762861796843237,-0.15561949041472778},
                                    {-0.006909679110575131,-0.0008125672627141157,-0.005330699828726684,0.005611086593181599,
                                     -0.015471264772306311, -0.012601808277168683,0.8833651638894245,0.334904431661722,
                                     -0.019803016522456408,-0.006946461007793779},
                                    {-0.006195896705029066,0.0003856000902087146,-0.005099115648659901,0.009086400892747586,
                                     -0.021306241232826817, -0.05363097089726912,1.376988336074456,5.06978619552465,
                                     -0.021906563535464815,-0.0043203160658777644},
                                    {-0.008813985612648064,-0.0008268277940356593,0.00584572719642164,0.00020400041219667715,
                                     -0.007654550411511217, -0.005258639841212538,-0.026586638946338063,-0.015175830449604966,
                                     1.3477677315998196,-0.663440808289155},
                                    {0.0025686387499803333,-0.00026953648639904123,-0.004176060856127733,0.011128859411698833,
                                     0.002071274901735221, -0.0008223719955175611,0.008501969688157014,0.004498939301120175,
                                     -0.6452648362052416,1.3771262132913318}};

__device__ double U02fit[10][10] = {{0.04902331457642449,-0.06577880637613859,0,0,0,0,0,0,0,0},
                                    {-0.06571207562366349,0.04921499221546199,0,0,0,0,0,0,0,0},
                                    {-0.0028580868439494014,0.004911420035351476,0.04272616402848554,-0.04447611173132893,
                                     0.017431516870294636,0.06239200292642038,0.0031060166653866816,0.011446451359137931,
                                     -0.005919426521676229,0.0077858492949352655},
                                    {0.00446681688364278,-0.0064090653516778,-0.0580149607373086,0.025370742834642364,
                                     -0.027180387733536416,-0.09403627654685497,-0.004846096801401712,-0.016807380907160267,
                                     0.008209122366246501,-0.010203833245134061},
                                    {0.00005067044056746364,-0.0007733430218079221,0.00015261611218813287,-0.005655198510884633,
                                     0.004400436696351807,0.0447896955551871,-0.0005976876592999581,-0.000872503933322475,
                                     0.0013323435007476202,-0.0013281679902572342},
                                    {0.007678819019776685,-0.019043972129646685,-0.0015770933538727253,-0.08912543193106118,
                                     0.15328368762792716,0.3765813427151738,-0.01304252541696912,-0.058734226418171664,
                                     0.024641218314207246,-0.033314274617451664},
                                    {-0.0009706511291316293,7.903303186649738e-6,-0.0014207694553599172,0.0009646900226810194,
                                     -0.0033719235620867814,-0.0038384140340767497,-0.0008275652639209308,0.04034256327618928,
                                     -0.003464475294357028,-0.0008592251103264992},
                                    {-0.0020337678420198116,0.000038029133269440444,-0.0026903864012142913,0.0027686264558118897,
                                     -0.00805481468604563,-0.01680155688674205,0.20187955639524865,0.6221150567297565,
                                     -0.008118590360519904,-0.0015801648605779659},
                                    {-0.0015126491108777285,-0.00005677627177786517,0.0004380515195125719,-0.0006318404874162453,
                                     -0.002302756706742392,-0.0015129178817664608,-0.005735887011592235,-0.003895956691425457,
                                     0.042081544548199006,-0.0662632526503892},
                                    {0.0008234204399839714,-8.251741480608195e-7,-0.0007222002652853608,0.0019619876513739093,
                                     0.0009301875913188679,-0.00025824316268070773,0.0028266017842895584,0.001194432494017896,
                                     -0.06154225782206215,0.048397869570983514}};


// RGE things
// Alphas in MZ scale
__device__ double alpha_in[3] = { 0.016887 ,0.03391670679768305, 0.1179};

// bi for SM (arXiv:0001257 example)
__device__ double rgSMb0[3] = {0.0,-22.0/3,-11};
__device__ double rgSMb1[3] = {4./3,4./3,4./3};
__device__ double rgSMb2[3] = {1./10,1.6,0};

// bi MSSM 
__device__ double rgSYb0[3] = {0.,-6,-9};
__device__ double rgSYb1[3] = {2,2,2}; 
__device__ double rgSYb2[3] = {3./10,1./2,0};

// Only One loop WR
__device__ void  rge_alpha_SM(double alpha_out[3], double mu, int Nfam, int Nhiggs)
{
  int i;
  double b;
  double sc =log(mu/sqrt(MZ2))/(2*M_PI);
  
  for (i=0;i <3; i++) 
  {
    b = rgSMb0[i] + rgSMb1[i]*Nfam + rgSMb2[i]*Nhiggs;
    alpha_out[i] = alpha_in[i]/(1 - b*alpha_in[i]*sc);
  }
  return;
  
}
// Same for SUSY
__device__ void  rge_alpha_SUSY(double alpha_out[3], double mu0, double mu, int Nfam, int Nhiggs)
{
  int i;
  double b;
  double sc =log(mu/mu0)/(2*M_PI);
  
  for (i=0;i <3; i++) 
  {
    b = rgSYb0[i] + rgSYb1[i]*Nfam + rgSYb2[i]*Nhiggs;
    alpha_out[i] = alpha_in[i]/(1 - b*alpha_in[i]*sc);
  }
  return;
}

//Based on arXiv: 1905.08257
__device__ void edm_RGE(double *d_in, double *d_out,
                        double alpha_m2, double alpha_mb, double alpha_mt, double alpha_MS,
                        bool isDown 
                        )
{
  double _[2][2];
  double out[2][2];
  double Qq = 2./3.;
  if (isDown) Qq = -1./3.;
  // printf("Qq = %.3f\n", Qq);

  double etas[3] = {alpha_mb/alpha_m2, alpha_mt/alpha_mb, alpha_MS/alpha_mt};
  double fs[3] = {4, 5, 6};
  double betas[3] = {(33 - 2*fs[0])/3, (33 - 2*fs[1])/3, (33 - 2*fs[2])/3};
  double U0[2][2] = { {pow(etas[0], 4./(3*betas[0])), 8*Qq * pow(etas[0], 2./(3.*betas[0])) * (pow(etas[0], 2./(3.*betas[0])) - 1.)},
                      {0., pow(etas[0], 2./(3*betas[0]))}  };
  double U1[2][2] = {{pow(etas[1], 4./(3*betas[1])), 8*Qq * pow(etas[1], 2./(3.*betas[1])) * (pow(etas[1], 2./(3.*betas[1])) - 1.)},
         {0., pow(etas[1], 2./(3*betas[1]))}
        };
  double U2[2][2] = {{pow(etas[2], 4./(3*betas[2])), 8*Qq * pow(etas[2], 2./(3.*betas[2])) * (pow(etas[2], 2./(3.*betas[2])) - 1.)},
         {0., pow(etas[2], 2./(3*betas[2]))}
        };

  mult2x2(_, U0, U1);
  mult2x2(out, _, U2);

  d_out[0] = out[0][0]*d_in[0] + out[0][1]*d_in[1];
  d_out[1] = out[1][0]*d_in[0] + out[1][1]*d_in[1];
}

//Based on arXiv: 980239
__device__ void c7c8_RGE(pycuda::complex<double> *_in, pycuda::complex<double> *_out,
                        double alpha_m2, double alpha_mb, double alpha_mt, double alpha_MS)
                        //we will need it at 2GeV for ACP and at mb for
                        //comparing with global fits
{
  double _[2][2];
  double out[2][2];
  double etas[3] = {alpha_mb/alpha_m2, alpha_mt/alpha_mb, alpha_MS/alpha_mt};
  double fs[3] = {4, 5, 6};
  double betas[3] = {(33 - 2*fs[0])/3, (33 - 2*fs[1])/3, (33 - 2*fs[2])/3};
  double U0[2][2] = { {pow(etas[0], 16./(3*betas[0])), 8./3. * (pow(etas[0], 14./(3.*betas[0]))  - pow(etas[0], 16./(3.*betas[0])) )},
                      {0., pow(etas[0], 14./(3*betas[0]))}  };
  double U1[2][2] = { {pow(etas[1], 16./(3*betas[1])), 8./3. * (pow(etas[1], 14./(3.*betas[1]))  - pow(etas[1], 16./(3.*betas[1])) )},
                      {0., pow(etas[1], 14./(3*betas[1]))}  };
  double U2[2][2] = { {pow(etas[2], 16./(3*betas[2])), 8./3. * (pow(etas[2], 14./(3.*betas[2]))  - pow(etas[2], 16./(3.*betas[2])) )},
                      {0., pow(etas[2], 14./(3*betas[2]))}  };

  mult2x2(_, U0, U1);
  mult2x2(out, _, U2);
  _out[0] = out[0][0]*_in[0] + out[0][1]*_in[1];
  _out[1] = out[1][0]*_in[0] + out[1][1]*_in[1];
  
}






//Based on arXiv: 9707225
__device__ void mixing_RGE(pycuda::complex<double> *C_in, pycuda::complex<double> *C_out,
                           double alpha_m3, double alpha_mb, double alpha_mt, double alpha_MS, bool isB)
{
 
  double X23_1[2][2];
  double X45_1[2][2];
  double R23[2][2];
  double R45[2][2];
  double dummy[2][2];
  double eta_1_K = pow(alpha_mt/alpha_mb, 0.260869565217391)*pow(alpha_MS/alpha_mt,0.285714285714286);
  if (!isB) eta_1_K = eta_1_K*pow(alpha_mb/alpha_m3,0.24);
  double eta23_K[2][2] = { { pow(eta_1_K, 1.0/6.0*(1 - sqrt(241.))), 0.0 } , 
                           { 0.0, pow(eta_1_K, 1.0/6.0*(1 + sqrt(241.)))} };
  
  double eta45_K[2][2] = {{pow(eta_1_K, -4), 0.0} , 
                          {0.0, pow(eta_1_K, 1.0/2.0) }};
  
  
  double X23[2][2] = {{1.0/2.0*( -15 - sqrt(241.)), 1.0/2.0*( -15 + sqrt(241.))},
                      {1.0, 1.0}};
  
  double X45[2][2] = {{ 1, -1 },
                      { 0, 3 }};
  
  inv2x2(X23_1,X23);
  inv2x2(X45_1, X45);
  mult2x2(dummy,eta23_K, X23_1);
  mult2x2(R23, X23, dummy);
  mult2x2(dummy,eta45_K, X45_1);
  mult2x2(R45,X45,dummy);
  C_out[0] = eta_1_K*C_in[0];
  
  C_out[1] = R23[0][0]*C_in[1] + R23[0][1]*C_in[2];
  C_out[2] = R23[1][0]*C_in[1] + R23[1][1]*C_in[2];
  
  C_out[3] = R45[0][0]*C_in[3] + R45[0][1]*C_in[4];
  C_out[4] = R45[1][0]*C_in[3] + R45[1][1]*C_in[4];
 
  return;
}



__device__ double Bmm_SM(int k)
{
  double tau, f, MP;
  pycuda::complex<double> Vtterm, Vcterm;
  double mW2 = mW*mW;
  double g2_2 = GF*mW2*8./sqrt(2.);
  pycuda::complex<double> I = pycuda::complex<double>(0,1);

  switch(k)
  {
  case 1:
    tau = tau_Bs;
    f = f_Bs;
    MP = m_Bs;
    Vtterm = Vtb*pycuda::conj(Vts);
    Vcterm = Vcb*pycuda::conj(Vcs);
    break;
    
  case 2:
    tau = tau_Bd;
    f = f_Bd;
    MP = m_Bd;
    Vtterm = Vtb*pycuda::conj(Vtd);
    Vcterm = Vcb*pycuda::conj(Vcd);
    break;
  }
  double CA = pycuda::abs(2*m_mu/MP * g2_2*4*GF*(Vtterm*Y0_xt+ Vcterm*YNL)/(16*M_PI*M_PI*sqrt(2.)));
  double MP2 = MP*MP;
  double stuff = 1 - 4*m_mu*m_mu/MP2;
  //caca;
  
  return tau*f*f*MP2*MP/(32*M_PI)*sqrt(stuff)*(CA*CA);
  
}


__device__ double epsprime(double Mg, double Mdl2, double Mdr2, double Mur2, double mu, double tb, double tbcorr, 
                           double BG,double alpha_mb, double alpha_mt, double alpha_s,
                           pycuda::complex<double> d12LL, pycuda::complex<double> d12RR)
{//a la teppei'17
  int i,j;
  
  double Mg2 = Mg*Mg;
  double alpha_s2 = alpha_s*alpha_s;
  
  double mk2 = m_K * m_K;
  double mpi2 = m_pi*m_pi;
  double preglu = alpha_s2/(4.*sqrt(2.)*Mg2);
  // gluino box
  pycuda::complex<double> pre = preglu*d12LL;
  double xl2 = Mdl2/Mg2;
  double xr2 = Mdr2/Mg2;
  //double xql2 = Mql2/Mg2;
  double xqr2 = Mur2/Mg2;
  
  double f = tep_f(xl2,xqr2) ;
  double g = tep_g (xl2,xqr2);

  pycuda::complex<double> C1 = pre * ( f -5.*g)/18.;
  pycuda::complex<double> C2 = pre * ( 7*f + g)/6.;
  f = tep_f(xl2,xl2) ;
  g = tep_g(xl2,xl2) ;
  pycuda::complex<double> C3 = pre * ( -5./9*f + 1./36*g);
  pycuda::complex<double> C4 = pre * ( 1./3*f + 7./12*g);
  
  pre = preglu*d12RR;

  f = tep_f(xr2,xl2) ;
  g = tep_g(xr2/Mg2,xl2) ;
  
  pycuda::complex<double> C1p = pre * ( f -5.*g)/18.;
  pycuda::complex<double> C2p = pre * ( 7*f + g)/6.;
  
  f = tep_f(xr2,xqr2);
  g = tep_g(xr2,xqr2/Mg2) ;
  
  pycuda::complex<double> C3p = pre * ( -5./9*f + 1./36*g);
  pycuda::complex<double> C4p = pre * ( 1./3*f + 7./12*g);
  pre = alpha_s*M_PI/3.;
  pycuda::complex<double> Cg = pre*mu*m_s*tb/(tbcorr*Mg2*Mg2*Mg)* 
    (Mdl2*d12LL*(tep_I(xl2,xr2) + 9*tep_J(xl2,xr2))-Mdr2*d12RR*(tep_I(xr2,xl2) + 9*tep_J(xr2,xl2)));
  Cg += pre*m_s*(-d12LL/Mdl2*(M3(1./xl2)+9*M4(1./xl2)) + d12RR/Mdr2*(M3(1./xr2)+9*M4(1./xr2)));
  double thing = 0.;
  double C1u = pycuda::imag(C1);
  double C2u = pycuda::imag(C2);
  double C3u = pycuda::imag(C3);
  double C4u = pycuda::imag(C4);
  
  double C1pu = pycuda::imag(C1p);
  double C2pu = pycuda::imag(C2p);
  double C3pu = pycuda::imag(C3p);
  double C4pu = pycuda::imag(C4p);
  
  xqr2 = xr2;
  pre = preglu*d12LL;

  f = tep_f(xl2,xqr2) ;
  g = tep_g (xl2,xqr2);

  C1 = pre * ( f -5.*g)/18.;
  C2 = pre * ( 7*f + g)/6.;
  f = tep_f(xl2,xl2) ;
  g = tep_g(xl2,xl2) ;
  C3 = pre * ( -5./9*f + 1./36*g);
  C4 = pre * ( 1./3*f + 7./12*g);
  
  pre = preglu*d12RR;

  f = tep_f(xr2,xl2) ;
  g = tep_g(xr2/Mg2,xl2) ;
  
  C1p = pre * ( f -5.*g)/18.;
  C2p = pre * ( 7*f + g)/6.;
  
  f = tep_f(xr2,xqr2);
  g = tep_g(xr2,xqr2/Mg2) ;
  
  C3p = pre * ( -5./9*f + 1./36*g);
  C4p = pre * ( 1./3*f + 7./12*g);

  double C1d = pycuda::imag(C1);
  double C2d = pycuda::imag(C2);
  double C3d = pycuda::imag(C3);
  double C4d = pycuda::imag(C4);
  
  double C1pd = pycuda::imag(C1p);
  double C2pd = pycuda::imag(C2p);
  double C3pd = pycuda::imag(C3p);
  double C4pd = pycuda::imag(C4p);
  //RGE Uhat
  double shit = log(sqrt(Mg*sqrt(Mdl2) )/1000.);
  double eps_stuff[10] = {0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
  
  
  for (i=0;i<10;i++) // columna
  { 
    for (j=0;j<10; j++) //fila
    {
      eps_stuff[i] += Qepsp[j]* (U01fit[j][i] + U02fit[j][i]*shit);
      
    }
  }
  

  thing += eps_stuff[2] * (C3u + 2*C3d - C3pu - 2*C3pd)/3.;
  thing += eps_stuff[3] * (C4u + 2*C4d - C4pu - 2*C4pd)/3.;
  thing += eps_stuff[4] * (C1u + 2*C1d - C1pu - 2*C1pd)/3.;
  thing += eps_stuff[5] * (C2u + 2*C2d - C2pu - 2*C2pd)/3.;
  thing += 2.*eps_stuff[6] * (C1u - C1d - C1pu  + C1pd)/3.;
  thing += 2.*eps_stuff[7] * (C2u - C2d - C2pu  + C2pd)/3.;
  thing += 2.*eps_stuff[8] * (C3u - C3d - C3pu  + C3pd)/3.;
  thing += 2.*eps_stuff[9] * (C4u - C4d - C4pu  + C4pd)/3.;

  double alpha_m1p3[3];
  rge_alpha_SM(alpha_m1p3, 1.3, 2, 1);

  double eta_s = pow(alpha_mt/alpha_mb, 2./23)*pow(alpha_mb/alpha_m1p3[2],2./25)*pow(alpha_s/alpha_mt,2./21);
  
  double chromo = (1-Omega_eff)*11.*sqrt(3.)*mpi2*mk2*eta_s*BG*pycuda::imag(Cg)/(64*M_PI*M_PI*f_pi*(m_s+m_d));
  /*
  printf("C1u: %e C2u: %e C3u: %e C4u:%e \n", C1u, C2u, C3u, C4u);
  printf("C1pu: %e C2pu: %e C3pu: %e C4pu:%e \n", C1pu, C2pu, C3pu, C4pu);
  printf("C1d: %e C2d: %e C3d: %e C4d:%e \n", C1d, C2d, C3d, C4d);
  printf("C1pd: %e C2pd: %e C3pd: %e C4pd:%e \n", C1pd, C2pd, C3pd, C4pd);
  */

  //printf("%e %e %e %e \n", wplus, ReA0, thing, chromo);
  
  return epek_SM*epsk_SM + wplus*(thing + chromo )/ReA0;
  //return epek_SM*epsk_SM + wplus*(chromo )/ReA0; Only chromo contribution

}




/*--------------------------------------------------------------------*/

__device__ double Pll_master(double tau, double f, double MP, double ml, double absA, double absB)
{
  double MP2 = MP*MP;
  double stuff = 1 - 4*ml*ml/MP2;
  return tau*f*f*MP2*MP/(32*M_PI)*sqrt(stuff)*(absB*absB*stuff +absA*absA);
  
}

__device__ pycuda::complex<double> BSM_M12(pycuda::complex<double> VCKM, int k,
                                           pycuda::complex<double> ddLL,pycuda::complex<double> ddRR, 
                                           pycuda::complex<double> duLL,
                                           double pCha_1, pycuda::complex<double> pChap_3, 
                                           double preCg_1, double preCgp_1, double preCg_4, double preCg_5, 
                                           double prea1, double prea2, double prea3, double prea4,
                                           pycuda::complex<double> C1_0,pycuda::complex<double> C1p_0, 
                                           pycuda::complex<double> C4_0, pycuda::complex<double> C5_0 ,
                                           double alpha_l, double alpha_b, double alpha_t, double alpha_MS)
{
  double fP, mP, m_d1, m_d2, B_1, B_3, B_4, B_5;
  
  pycuda::complex<double> VCKM2 = VCKM*VCKM;
  
  switch(k) {
  case 1: 
    fP = f_Bs;
    mP = m_Bs;
    m_d1 = m_b;
    m_d2 = m_s;
    B_1 = B_1s;
    B_3 = B_3s;
    B_4 = B_4s;
    B_5 = B_5s;
    break;

  case 2: 
    fP = f_Bd;
    mP = m_Bd;
    m_d1 = m_b;
    m_d2 = m_s;
    B_1 = B_1d;
    B_3 = B_3d;
    B_4 = B_4d;
    B_5 = B_5d;
    break;
  case 3:    
    fP = f_K;
    mP = m_K;
    m_d1 = m_s;
    m_d2 = m_d;
    B_1 = B_1k;
    B_3 = B_3k;
    B_4 = B_4k;
    B_5 = B_5k;
    break;
    
  case 4:
    fP = f_D;
    mP = m_D;
    m_d1 = m_u;
    m_d2 = m_c;
    B_1 = B_1c;
    B_3 = B_3c;
    B_4 = B_4c;
    B_5 = B_5c;
    
    break;
    
  }
  double fP2 = fP*fP;
  double mP2 = mP*mP;
  double quarks = m_d1+m_d2;
  double mq2 = quarks*quarks;
  double crap1 =fP2*mP2*mP2/(12*mq2); 

  //RR clarification: Factor 4 from 1-\gamma5 (old notation) vs PL and PR notation 
  double Q1=2*B_1*fP2*mP2/3; 
  //double Q2=-5*B_2*crap1;
  
  // double Q3 = B_3*crap1;
  // double Q4 = 6*B_4*crap1;
  // double Q5 = 2*B_5*crap1;
  //RR exact parametrization (from 1909.11087)
  double Q3 = B_3*fP2*mP2*mP2/(12*mq2);
  double Q4 = B_4*fP2*mP2*(mP2/(2*mq2) + 1./12.);
  double Q5 = B_5*fP2*mP2*(mP2/(6*mq2) + 1./4.);
  
  
  pycuda::complex<double> Cha_1 =  pCha_1*VCKM2;
	
  // pycuda::complex<double> Chap_3 = pChap_3*VCKM2*mq2; 
  pycuda::complex<double> Chap_3 = pChap_3*VCKM2*m_b*m_b;  //RR: Based on paper
  
  pycuda::complex<double> Cg_1 = preCg_1 * ddLL*ddLL;
  pycuda::complex<double> Cg_4 = preCg_4 * ddLL*ddRR;
  pycuda::complex<double> Cg_5 = preCg_5 * ddLL*ddRR;
  
  pycuda::complex<double> Cgp_1 = preCgp_1 * ddRR*ddRR;


  // MLs modified this (according to Teppeis notes) // RR checked
  pycuda::complex<double> a1 = prea1*ddLL*ddRR;
  pycuda::complex<double> a2 = prea2*ddRR;
  pycuda::complex<double> a3 = prea3*VCKM;
  pycuda::complex<double> a4 = prea4*duLL;

  // ML modified this (according to Teppeis notes)/ RR checked
  pycuda::complex<double> C_in[5] = {Cha_1 + Cg_1 + C1_0 + Cgp_1 + C1p_0, 0., Chap_3,a1+ a2*(a3+a4) + Cg_4 + C4_0, Cg_5 + C5_0};
  pycuda::complex<double> C_out[5] = {0.,0.,0.,0.,0.};

  mixing_RGE(C_in, C_out, alpha_l, alpha_b, alpha_t, alpha_MS, k!=3);
  
  return (Q1*C_out[0] + Q3*C_out[2]  + Q4*C_out[3] + Q5*C_out[4])/(2*mP) ;
}

__global__ void MIA_observables(double *pll_obs, double *m12_obs,
                                double *edm_obs,  pycuda::complex<double>  *wc_obs,
                                double *mass_obs, double *chi2, 
                                double *chi2_RDMs, double *chi2_RDMd, double *chi2_Rds, 
                                // double *chi2_RDMK, double *chi2_DMD, 
                                double *chi2_dphis, double *chi2_dphid, double *chi2_Asl,
                                double *chi2_C7,double *chi2_DACP, 
                                double *chi2_RTAUNU,  double *chi2_Rsmm, double *chi2_Rdmm,
                                double *chi2_dn, double *chi2_dHg, 
                                double *cost,
                                double *param,
                                pycuda::complex<double> *offdiag_deltas, 
                                double sign_Agg, int Nevt)
{
  pycuda::complex<double> I = pycuda::complex<double>(0,1);
  // Event 
  int row = threadIdx.x + blockDim.x * blockIdx.x; //ntuple entry
  if (row >= Nevt) { return;}
  // Observables
  // Old
  // int k0 = row*10;// M12 obs: 3 Bs, 2 Bd, 3 K0, 2 D0
  // int i0 = row*8;// Pll, Plnu obs 
  // int e0 = row*15;// EDM obs 12 (previous + dn + dp + dHg)
  
  int k0 = row*5;// M12 obs: 3 Bs, 2 Bd 
  int i0 = row*3;// Bsmm, Bdmm, Btaunu
  int e0 = row*7;// d_edm, u_edm, d_cedm, u_cedm, dn, dp, dHg
  int wc0 = row*7;// C7 (mb), C7p (mb), C8 (mb), C8p (mb) , \Delta ACP (2GeV), \Delta C9, \Delta C10
  int m0 = row*9; //mh, 2charg, 4N, M1, M2

  // inputs
  int j0 = row*17;// reals
  int d0 = row*14; // deltas : 3 x 4 ( [12,13,23] x [dLL, dRR, uLL, uRR] ) = 12 + 2  (LR_33 D + U)


  chi2[row] = 0.;
  chi2_RDMs[row] = 0.;
  chi2_RDMd[row] = 0.;
  chi2_Rds[row] = 0.;
  // chi2_RDMK[row] = 0.;
  // chi2_DMD[row] = 0.;
  chi2_dphis[row] = 0.;
  chi2_dphid[row] = 0.;
  chi2_Asl[row] = 0.;
  // chi2_RepsK[row] = 0.;
  // chi2_epek[row] = 0.;
  chi2_Rsmm[row] = 0.;
  chi2_Rdmm[row] = 0.;
  // chi2_KLmm[row] = 0.;
  chi2_RTAUNU[row] = 0.;
  // chi2_KMUNU[row] = 0.;
  chi2_C7[row] = 0.;
  chi2_DACP[row] = 0.;
  chi2_dHg[row] = 0.;
  chi2_dn[row] = 0.;

  // maxdelta[row] = 0;
  // C5g[row] = 0;
  // C4g[row] = 0;
  // C4H[row] = 0;
  // C4[row] = 0;
  // phC5g[row] = 0;
  // phC4g[row] = 0;
  // phC4H[row] = 0;
  // phC4[row] = 0;
  // ReC4[row] = 0;
  // ImC4[row] = 0;
// -- Shit that we are calculating here but that ideally we should have done it only once
  
  double pi2 = M_PI*M_PI;
  double sq2 = sqrt(2.);
  double mt2 = mt*mt;
  double mt4 = mt2*mt2;
  double mW2 = mW*mW;
  double mW4 = mW2*mW2;
  // double g3_2 = g3*g3;

  double g2_2 = GF*mW2*8./sq2;
   
  //alpha_s2 = 0.00603358;
  //alpha2_2 = 0.00101072;
  //alpha_s = sqrt(alpha_s2);
  //alpha_2 = sqrt(alpha2_2);
  // end global shit (for now)

  // Read input parameters
  // /*
  double mtilde = param[j0];  // mQ su(2) protected
  double tanb   = param[j0 + 1];
  double Mg     = param[j0 + 2];// this is actually M3
  double MA     = param[j0 + 3];

  // SM couplings in 3, mb, mt, M_susy = sqrt(msq*m3) -> WR all calculation must be done at this scale and then apply RGE
  double alpha_m3[3];
  double alpha_mb[3];  
  double alpha_mt[3]; 
  double alpha_MS[3];
  double alpha_m2[3];


  rge_alpha_SM(alpha_m2, 2., 2, 1);
  rge_alpha_SM(alpha_m3, 3.0, 2, 1);
  rge_alpha_SM(alpha_mb, m_b, 2, 1);
  // rge_alpha_SM(alpha_mt, mt, 2, 1);
  // rge_alpha_SM(alpha_MS, sqrt(mtilde*Mg), 2, 1); 
  // RR clarification : RGE_SUSY for scale much bigger than mususy

  // RR change 3 active families higher top scale
  rge_alpha_SM(alpha_mt, mt, 3, 1);
  rge_alpha_SM(alpha_MS, sqrt(mtilde*Mg), 3, 1); 



  
  // Scenario A, B, D, E
  double mu = param[j0 + 4]*sign_Agg;
  //To check that you do correct sign ! :)
//   if (row%100000==0){
//     printf("mu = %.4f\n", mu);
// }

  double M_1 = alpha_MS[0]/alpha_MS[2]*Mg*param[j0 + 5] ; //GUT arguments, the parameters should be 2, WR
  double M_2 = alpha_MS[1]/alpha_MS[2]*Mg*param[j0 + 6] ;
  
  //Scenario C
  if((param[j0+5]==5000.0)&&(param[j0+6]==3000.0)){
    M_1 = param[j0 + 5] ; 
    M_2 = param[j0 + 6] ;

  }
  
  //Trilinear Couplings
  // double Au = param[j0 + 7];
  // double Ac = param[j0 + 8];
  double At_pre = param[j0 + 9];  //WR to be honest i do not understand that
  //Split masses
  double xdLR = param[j0 + 10];
  double xuRL = param[j0 + 11];

  //This identifies Scenario D
  if (xuRL==10.){
    xuRL = 1e8/(mtilde*mtilde);
    param[j0+11] = xuRL; //To save it correctly in the tuple
  }

  double xtRdL = param[j0 + 12]; //Shoold be 1
  double xtLdL = param[j0 + 13]; //Should be 1
  //Other parameters
  double rh = param[j0+14]; 
  // double BG = param[j0+15]; 

  double msnu = param[j0+16];
  
  // First generations
  // double muR2 = mtilde*mtilde;
  // double mdL2 = muR2;
  // double mdR2 = mdL2/xdLR;

  //Protected by SU(2)
  double mdL2 = mtilde*mtilde;
  double muL2 = mtilde*mtilde;
  //Right singlets
  double mdR2 = mdL2/xdLR;
  double muR2 = muL2*xuRL;

  //Third generation 
  double mtR2 = mdL2*xtRdL*xuRL; //kept as muR2
  double mtL2 = mdL2*xtLdL; //kept as mtilde2

  double At = At_pre;
  // double At_fitted, par0, par1, par2;

  ///// ML's task: reorganize this /////
  //WR: This shit was used for higgs mass
  //WR: I think it does not make any sense
  // WR: For the moment just the At of the scan_ranges

  // if (At_pre==0.){
  //   At = At_pre;
  // }

  // if(At_pre > 0){
  //   par0 = 9.03303e-04;
  //   par1 = -2.62675e-04;
  //   par2 = 2.18413e+00; 
  //   //TODO: include this errors somehow in the estimate (Q)
  //   // FCN=1.39124 FROM HESSE     STATUS=OK             16 CALLS          50 TOTAL
  //   //                   EDM=7.59969e-22    STRATEGY= 1      ERROR MATRIX ACCURATE
  //   //EXT PARAMETER                                   STEP         FIRST  
  //   //NO.   NAME      VALUE            ERROR          SIZE      DERIVATIVE
  //   // 1  p0           9.03303e-04   2.43897e-04   5.98736e-08  -2.78142e-07
  //   // 2  p1          -2.62675e-04   3.74336e-05   3.72430e-09  -5.51489e-06
  //   // 3  p2           2.18413e+00   2.49368e-01   2.92167e-05  -7.02993e-10
   
  //   //if At_fitted<0, then set At=0, else At = At_fitted
  //   // WR : This is not valid when mstop != mtilde2
  //   At_fitted = mdL2*(par0/tanb + par1) + sqrt(mdL2)*(par0/tanb + par2);// ML changed this to accommodate for fitting function in terms of mstop, tanb  M_stop (GeV) = m_Q = m_u. ; // param[j0 + 9];
  //   if(At_fitted < 0){ 
  //     At = 0;
	// }
  //   else{
  //     At = At_fitted;
  // }		
	// }
  
  // if(At_pre < 0){
  //   par0 = -7.72767e-04;
  //   par1 = 2.96675e-04;
  //   par2 = -2.34158e+00;
  //   //TODO: include this errors somehow in the estimate (Q)
  //   // FCN=4.94189 FROM HESSE     STATUS=OK             16 CALLS          52 TOTAL
  //   //                 EDM=7.15798e-20    STRATEGY= 1      ERROR MATRIX ACCURATE 
  //   //   EXT PARAMETER                                   STEP         FIRST   
  //   //   NO.   NAME      VALUE            ERROR          SIZE      DERIVATIVE 
  //   //   1  p0          -7.72767e-04   9.52012e-05   2.21987e-08  -9.00234e-07
  //   //   2  p1           2.96675e-04   2.70472e-05   1.48198e-09  -7.49148e-06
  //   //   3  p2          -2.34158e+00   2.06104e-01   1.34719e-05   1.15375e-09
  //   //if At_fitted>0, then set At=0, else At = At_fitted
  //   At_fitted = mdL2*(par0/tanb + par1) + sqrt(mdL2)*(par0/tanb + par2);// ML changed this to accommodate for fitting function in terms of mstop, tanb  M_stop (GeV) = m_Q = m_u. ; // param[j0 + 9];
  //   if(At_fitted > 0){ 
  //     At = 0;
	// }
  //   else{
  //     At = At_fitted;
	// }
  // } 

  
  // LL MI's
  pycuda::complex<double> ddLL_12 = offdiag_deltas[d0];
  pycuda::complex<double> ddLL_13 = offdiag_deltas[d0 + 1];
  pycuda::complex<double> ddLL_23 = offdiag_deltas[d0 + 2];


  // RR MI's
  pycuda::complex<double> ddRR_12 = offdiag_deltas[d0 + 6];
  pycuda::complex<double> ddRR_13 = offdiag_deltas[d0 + 7];
  pycuda::complex<double> ddRR_23 = offdiag_deltas[d0 + 8];

  //Artifact to save the tuple correctly with ddRRs
  // offdiag_deltas[d0 + 6] = ddRR_12;
  // offdiag_deltas[d0 + 7] = ddRR_13;
  // offdiag_deltas[d0 + 8] = ddRR_23;
  
  pycuda::complex<double> duRR_12 = offdiag_deltas[d0 + 9];
  pycuda::complex<double> duRR_13 = offdiag_deltas[d0 + 10];
  pycuda::complex<double> duRR_23 = offdiag_deltas[d0 + 11];
  //hacks
  //ddRR_12 = ddLL_12;
  
  //conjugates
  pycuda::complex<double> ddLL_21 = pycuda::conj(ddLL_12);
  pycuda::complex<double> ddLL_31 = pycuda::conj(ddLL_13);
  pycuda::complex<double> ddLL_32 = pycuda::conj(ddLL_23);

 
  pycuda::complex<double> ddRR_21 = pycuda::conj(ddRR_12);
  pycuda::complex<double> ddRR_31 = pycuda::conj(ddRR_13);
  pycuda::complex<double> ddRR_32 = pycuda::conj(ddRR_23);

  pycuda::complex<double> duRR_21 = pycuda::conj(duRR_12);
  pycuda::complex<double> duRR_31 = pycuda::conj(duRR_13);
  pycuda::complex<double> duRR_32 = pycuda::conj(duRR_23);

  double ddLL_11 = 1.0;
  double ddLL_22 = 1.0;
  //WR: TODO: CHECK 
  double ddLL_33 = 1.0;
  //WR: TODO:CHECK
  // double ddRR_33 = 1.0;
  
  // up-type LL calculated by SU(2) symmetry
  offdiag_deltas[d0 + 3] = (Vcb*ddLL_13 + Vcd*ddLL_11 + Vcs*ddLL_12)*pycuda::conj(Vud) 
    + (Vcb*ddLL_23 + Vcd*ddLL_21 + Vcs*ddLL_22)*pycuda::conj(Vus) 
    + (Vcb*ddLL_33 + Vcd*ddLL_31 + Vcs*ddLL_32)*pycuda::conj(Vub);
  
  offdiag_deltas[d0 + 4] = (Vtb*ddLL_13 + Vtd*ddLL_11 + Vts*ddLL_12)*pycuda::conj(Vud) 
    + (Vtb*ddLL_23 + Vtd*ddLL_21 + Vts*ddLL_22)*pycuda::conj(Vus) 
    + (Vtb*ddLL_33 + Vtd*ddLL_31 + Vts*ddLL_32)*pycuda::conj(Vub);// 
  
  offdiag_deltas[d0 + 5] = (Vtb*ddLL_13 + Vtd*ddLL_11 + Vts*ddLL_12)*pycuda::conj(Vcd) 
    + (Vtb*ddLL_23 + Vtd*ddLL_21 + Vts*ddLL_22)*pycuda::conj(Vcs) 
    + (Vtb*ddLL_33 + Vtd*ddLL_31 + Vts*ddLL_32)*pycuda::conj(Vcb); //

  pycuda::complex<double> duLL_12 = offdiag_deltas[d0 + 3];
  pycuda::complex<double> duLL_13 = offdiag_deltas[d0 + 4];
  pycuda::complex<double> duLL_23 = offdiag_deltas[d0 + 5];

  pycuda::complex<double> duLL_21 = pycuda::conj(duLL_12);
  pycuda::complex<double> duLL_31 = pycuda::conj(duLL_13);
  pycuda::complex<double> duLL_32 = pycuda::conj(duLL_23);

  // LR MI's
  // pycuda::complex<double> ddLR_21 = 0.;
  // pycuda::complex<double> ddLR_23 = 0.;
  // pycuda::complex<double> ddLR_32 = 0.;
  // pycuda::complex<double> ddLR_33 = 0.;
  // pycuda::complex<double> ddRL_23 = 0.;
  // pycuda::complex<double> ddRL_32 = 0.;
  // pycuda::complex<double> ddRL_33 = 0.;
  // pycuda::complex<double> duLR_23 = 0.;
  // pycuda::complex<double> duLR_33 = 0.;

  
  // pycuda::complex<double> ddLR_21 = offdiag_deltas[d0 + 12]; //Only kaons
  // pycuda::complex<double> ddLR_23 = offdiag_deltas[d0 + 13]; //C9 + C10 (gluino)
  // pycuda::complex<double> ddLR_33 = offdiag_deltas[d0 + 14]; //C9, C10 (2MI not relevant) (related w/ Ab)

  // pycuda::complex<double> ddRL_23 = offdiag_deltas[d0 + 15]; //C10, C7 (conj)
  // pycuda::complex<double> ddRL_33 = offdiag_deltas[d0 + 16]; //C10p 

  //TODO Check if it is conjugate
  // pycuda::complex<double> ddRL_32 = pycuda::conj(ddLR_23); //arxiv 9511250 -> TODO
  // pycuda::complex<double> ddLR_32 = pycuda::conj(ddRL_23); //arxiv 9511250 C7',C8'

  
  double mtilde2 = mtilde*mtilde;
  double beta = atan(tanb); 
  // /* DEBUG
  // if (row%6000==0){
    // printf("ddRL_23 = %.4f\n", pycuda::real(ddRL_23));
    // printf("ddRL_33 = %.4f\n", pycuda::real(ddRL_33));
    // printf("ddLR_23 = %.4f\n", pycuda::real(ddLR_23));
    // printf("ddLR_33 = %.4f\n", pycuda::real(ddLR_33));
    // printf("duLR_23 = %.4f\n", pycuda::real(duLR_23));
    // printf("duLR_33 = %.4f\n", pycuda::real(duLR_33));

    // printf("im_ddRL_23 = %.4f\n", pycuda::imag(ddRL_23));
    // printf("im_ddRL_33 = %.4f\n", pycuda::imag(ddRL_33));
    // printf("im_ddLR_23 = %.4f\n", pycuda::imag(ddLR_23));
    // printf("im_ddLR_33 = %.4f\n", pycuda::imag(ddLR_33));
    // printf("im_duLR_23 = %.4f\n", pycuda::imag(duLR_23));
    // printf("im_duLR_33 = %.4f\n", pycuda::imag(duLR_33));
    
    //Main
    // printf("tanb=%.2f\n", tanb);
    // printf("mtilde=%.2f\n", mtilde);
    // printf("m_g=%.2f\n", Mg);
    // printf("MA=%.2f\n", MA);
    // printf("M_1=%.2f\n", M_1);
    // printf("M_2=%.2f\n", M_2);
    // trilinear
    // printf("Au=%.2f\n", Au);
    // printf("Ac=%.2f\n", Ac);
    // printf("At_pre=%.2f\n", At_pre);
    //Mass-splittings
    // printf("xdLR=%.2f\n", xdLR);
    // printf("xuRL=%.2f\n", xuRL);
    // printf("xtRdL=%.2f\n", xtRdL);
    // printf("xtLdL=%.2f\n", xtLdL);
    //Other
    // printf("rh=%.2f\n", rh);
    // printf("BG=%.2f\n", BG);
    // printf("msnu=%.2f\n", msnu);
  // }

  // */
 
  pycuda::complex<double> VtbVts = Vtb*pycuda::conj(Vts);
  pycuda::complex<double> VtbVtd = Vtb*pycuda::conj(Vtd);
  pycuda::complex<double> VtsVtd = Vts*pycuda::conj(Vtd);
  pycuda::complex<double> VtdVts = Vtd*pycuda::conj(Vts);

  pycuda::complex<double> VcbVcs = Vcb*pycuda::conj(Vcs);
  pycuda::complex<double> VcbVcd = Vcb*pycuda::conj(Vcd);
  pycuda::complex<double> VcdVcs = Vcd*pycuda::conj(Vcs);
  pycuda::complex<double> VtbVts2 = VtbVts*VtbVts;

  // /* 
  double alpha_2 = alpha_MS[1]; //SUSY scale
  double alpha_s = alpha_MS[2]; //SUSY scale

  double alpha_s2 = alpha_s*alpha_s;
  double alpha2_2 = alpha_2*alpha_2; 
  
  double mu2 = mu*mu;
  //double At2 = At*At;

  double absmu2 = mu2; //pycuda::abs(mu2); 
  double absM2_2 = M_2*M_2;  

  double Mg2 = Mg*Mg;
  double mtilde4 = mtilde2*mtilde2;
  double tanb2 = tanb*tanb;
  double tanb3 = tanb2*tanb;
  double tanb4 = tanb3*tanb;
  double MA2 = MA*MA;
  //H^{\pm}
  //WR RR correction
  // double massH2 = (MA2 + MZ2)*rh;
  double massH2 = (MA2 + mW2)*rh; //Supersymmetry beyond minimality 6.14
  // WR mdL2=mtilde2 -> xg = xgL (?) (xmu=xmuL)
	double xg = Mg2/mtilde2;
  //WR: Missleading -> should be xgdL, xgdR ...
  double xgL = Mg2/mdL2;
  double xgR = Mg2/mdR2;
  double xsnu = msnu/mtilde2;
  
	double xmu = absmu2/mtilde2;
  double xmuL = absmu2/mdL2;
  double xLmu = 1./xmuL;
  // double xmuR = absmu2/mdR2;
  double xumu = muR2/absmu2;

  double yt = mt*mt/massH2;  
  // C.12
	double eps = 2*alpha_s*mu*Mg*LoopF(xgL,1./xdLR)/(3*M_PI*mdL2);
  double eps_prime = 2*alpha_s*mu*Mg*LoopF(xgL, xuRL)/(3*M_PI*mdL2);
  // C.13
  // RR modified this according to Teppei notes
  // double epsYyt = mu*At*LoopF(xmuL,xtRdL)*yt*yt/(16*M_PI*M_PI*mdL2); //previous one
  double epsYyt = -alpha_2*mu*At*mt2*LoopF(xmuL,xuRL)/(8*M_PI*mdL2*mW2);
  
  double x_2 = absM2_2/mtilde2;
  double x_2L = absM2_2/mdL2;
  
  double tbcorr = 1.0 +eps*tanb;
  double tbcorr2 = tbcorr*tbcorr;
  double tbcorrY = tbcorr + epsYyt*tanb;
  double tbcorrY2 = tbcorrY*tbcorrY;
  double tbcorrn = 1.00-eps*tanb;
  double tbcorrn_prime = 1. - eps_prime*tanb;

  double tbcorrlep = 1.0- 3*alpha_2/(16*M_PI)*tanb;/// DMS: degenerate susy, improve;


  offdiag_deltas[d0+12] = m_b*mu*tanb/(tbcorrY*mtilde*sqrt(mdR2));
  pycuda::complex<double> ddLR_33 = offdiag_deltas[d0+12];
  offdiag_deltas[d0+13] = -mt*(At+mu/tanb)/(mtilde*sqrt(muR2));
  pycuda::complex<double> duLR_33 = offdiag_deltas[d0+13];

  // if(row%3000==0){
  //   printf("duLR_33_r = %.4f\n", pycuda::real(duLR_33));
  //   printf("At_pre = %.4f\n", At_pre);
  //   printf("At = %.4f\n", At);
  //   printf("mu = %.4f\n", mu);
  // }

  /////////////////////////////////////////////////////////////////////////////////
  ///////////////                                                        //////////
  /////////////                          M12                           //////////
  /////////////                  Reviewer: Ramon Ruiz                  //////////
  /////////////                                                        //////////
  ///////////////////////////////////////////////////////////////////////////////
  
  
  // double gluino_cte = -alpha_s2/mtilde2;

  //Gluino Contributions
  double preCg_1 = -alpha_s2*g1(xgL)/mdL2;
  double preCg_4 = -alpha_s2*tep_g4_1(xgL, xgR)/Mg2;
  double preCg_5 = -alpha_s2*tep_g5_1(xgL, xgR)/Mg2;

  double preCgp_1 =  -alpha_s2*g1(xgR)/mdR2;
  
  //Chargino Contributions
  double preCwh_1 = (-1./6.*alpha_2*alpha_s*tep_ggw_1(xgL, x_2L) -1./8*alpha2_2*tep_gw_1(x_2L) )/mdL2;
  double pCha_1 =  -alpha2_2*mt4*f1(mu2/mtR2)/(8*muR2*mW4); 
  //RR clarification :m_b in BSM_M12
  double pChap_3 = -1./8.*At*At*alpha2_2*tep_f3(mu2/muL2, mu2/muR2)*mt4*mu2*tanb2/(muL2*muL2*muR2*muR2*mW4*tbcorrY2);
 
  preCg_1 += preCwh_1; //RR -> Artifact proportional to ddLL_32 * ddLL_32

  //Added by Miriam C4, C5 gluino contributions 
  //WR: Not sure this is correct, as all C´s mixes (?), also misses double-Higgs penguin
  // pycuda::complex<double> Cg_5 = preCg_5 * ddLL_32*ddRR_32;
  // pycuda::complex<double> Cg_4 = preCg_4 * ddLL_32*ddRR_32;
  
  // pycuda::complex<double> Cg_in[5] = {0., 0., 0., Cg_4, Cg_5};
  // pycuda::complex<double> Cg_out[5] = {0.,0.,0.,0.,0.};
  
  // mixing_RGE(Cg_in, Cg_out,alpha_m3[2], alpha_mb[2], alpha_mt[2], alpha_MS[2],1!=3);
  // C5g[row] = pycuda::abs(Cg_out[4]);
  // phC5g[row] = pycuda::arg(Cg_out[4]);
  // C4g[row] = pycuda::abs(Cg_out[3]);
  // phC4g[row] = pycuda::arg(Cg_out[3]);

  // double h1g = h1(xg); 
  // double h3mu = h3(xmu);
  // double h42mu = h4(x_2,xmu);
  //WR Universal Masses -> Old To be deprecated
  // double prea1 = alpha_s*alpha_2*tanb4*absmu2/(8*M_PI*mW2*MA2*mtilde4)*tbcorr2*tbcorr2;
  // double prea2 = -alpha_s*Mg2*h1g*h1g;
  // double prea3 = alpha_2*mt2*At*Mg*h1g*h3mu/mW2;
  // double prea4 = alpha_2*M_2*Mg*h1g*h42mu; 
  
  
  //Double Higgs Penguin contribution
  double newprea1_NUM = -8*alpha_s2*alpha_2*m_b*m_b*tanb4*absmu2*Mg2;
  double newprea1_DEN = 9*M_PI*mW2*tbcorr2*tbcorrY2*MA2*mdR2*mdL2;
  double newprea1 = newprea1_NUM*LoopG(xgR,xdLR)*LoopG(xgL,1./xdLR)/newprea1_DEN;

  double newprea2_NUM = alpha2_2*alpha_s*m_b*m_b*tanb4*absmu2*Mg*LoopG(xgR,xdLR);
  double newprea2_DEN = 6*M_PI*mW2*tbcorr2*tbcorrY2*MA2*mdR2;
  double newprea2 = newprea2_NUM/newprea2_DEN;

  //RR update
  double newprea3_NUM = mt2*At*LoopF(xmuL,xuRL);
  double newprea3_DEN = mW2*mdL2; 
  double newprea3 = newprea3_NUM/newprea3_DEN;

  double newprea4 = 2*M_2*LoopG(x_2L,xmuL)/mdL2;
    
  double m_d1 = m_b;  
  double m_d2 = m_s;
  double quarks = m_d1+m_d2;
  double mq2 = quarks*quarks;

  // Higgs Penguin stuff                                                                                                        
  //RR: Modified to new C4 and C5 
  // pycuda::complex<double> a1 = newprea1*ddLL_32*ddRR_32;
  // pycuda::complex<double> a2 = newprea2*ddRR_32;
  // pycuda::complex<double> a3 = newprea3*VtbVts;
  // pycuda::complex<double> a4 = newprea4*duLL_32;

  // pycuda::complex<double> CH_in[5] = {0., 0., 0., a1+a2*(a3+a4),0.};
  // pycuda::complex<double> CH_out[5] = {0.,0.,0.,0.,0.};

  //WR: Is this valid (?), one should not do the mixing together with the rest (?)
  // mixing_RGE(CH_in, CH_out,alpha_m3[2], alpha_mb[2], alpha_mt[2], alpha_MS[2],1!=3);
  // C4H[row] = pycuda::abs(CH_out[3]);
  // phC4H[row] = pycuda::arg(CH_out[3]);

  // pycuda::complex<double> C4TOTAL_COMPLEX = Cg_out[3] + CH_out[3];

  // C4[row] = pycuda::abs(C4TOTAL_COMPLEX);
  // phC4[row] = pycuda::arg(C4TOTAL_COMPLEX);
  // ReC4[row] = pycuda::real(C4TOTAL_COMPLEX);
  // ImC4[row] = pycuda::imag(C4TOTAL_COMPLEX);


  ////////////////////////////////////////////////
  /// Higher order in MIA (for K0-K0bar mixing) //
  ////////////////////////////////////////////////

  // pycuda::complex<double> C1K = gluino_cte*(ddLL_21*ddLL_23*ddLL_31*g1_2(xg) + ddLL_23*ddLL_31*ddLL_23*ddLL_31*g1_3(xg));
  // pycuda::complex<double> C1pK = gluino_cte*(ddRR_21*ddRR_23*ddRR_31*g1_2(xg) + ddRR_23*ddRR_31*ddRR_23*ddRR_31*g1_3(xg));
  // pycuda::complex<double> C4K = gluino_cte*( 0.5*(ddLL_21*ddRR_23*ddRR_31 + ddRR_21*ddLL_23*ddLL_31 ) *g4_2(xg) +
  //                                           ddLL_23*ddLL_31*ddRR_23*ddRR_31*g4_3(xg));
  // pycuda::complex<double> C5K = gluino_cte*( 0.5*(ddLL_21*ddRR_23*ddRR_31 + ddRR_21*ddLL_23*ddLL_31 ) *g5_2(xg) +
  //                                           ddLL_23*ddLL_31*ddRR_23*ddRR_31*g5_3(xg));
  
  // 4th order MIA Higgs Penguin
  // C4K += -alpha_s2*alpha_2*m_b*m_b*tanb4*absmu2*Mg2*ddLL_23*ddLL_31*ddRR_23*ddRR_31*h2(xg);
  /// End higher order MIA
  
  ////////////////////////////////////////////////////////////
  /////////////////7       Bs - Bsbar     ////////////////////
  ////////////////////////////////////////////////////////////

  //SM
  pycuda::complex<double> M12_SM = GF*GF*VtbVts2*mW2*S0_xt*eta_B_hat*B_1s*f_Bs*f_Bs*m_Bs / (12*M_PI*M_PI); 
  //BSM 
  pycuda::complex<double> M12 = M12_SM + BSM_M12(VtbVts, 1, ddLL_32, ddRR_32,duLL_32, 
                                                //Chargino contributions
                                                 pCha_1, pChap_3,
                                                //Cg1     Cgp1       Cg4     Cg5
                                                 preCg_1, preCgp_1, preCg_4, preCg_5, 
                                                //Higgs-Penguin shit
                                                 newprea1, newprea2, newprea3, newprea4, 
                                                 0.,0.,0.,0.,
                                                 alpha_m3[2], alpha_mb[2], alpha_mt[2], alpha_MS[2]);




  m12_obs[k0] = pycuda::arg(M12/M12_SM);
  m12_obs[k0+1] = pycuda::abs(M12/M12_SM);

  //Is not really Asl but Asl*dms/dgs = -tan(phis) -> Not dependent on penguins
  m12_obs[k0+2] = tan(- m12_obs[k0] + CONSTR_phis12) - tan(CONSTR_phis12);

  

  // We will have two CONSTR_dphis in the case of having Penguins into account or not
  chi2[row] += pow( (m12_obs[k0] - CONSTR_dphis)/CONSTR_sdphis, 2); //phis
  chi2[row] += pow( (m12_obs[k0+1] - CONSTR_RDMs)/CONSTR_sRDMs, 2); //DMs
  chi2[row] += pow( (m12_obs[k0+2] - CONSTR_Asl)/CONSTR_sAsl, 2); //Asl related

  //Track of each particular chi2 contribution for debugging :)
  chi2_dphis[row] = pow( (m12_obs[k0] - CONSTR_dphis)/CONSTR_sdphis, 2);
  chi2_RDMs[row] = pow( (m12_obs[k0+1] - CONSTR_RDMs)/CONSTR_sRDMs, 2);
  chi2_Asl[row] = pow( (m12_obs[k0+2] - CONSTR_Asl)/CONSTR_sAsl, 2);


  //Miriam implementation: tbh i do not understand it  (RR)
  // dilepton asymmetry:
  // double phi_Gamma12 = 3.0944;
  //pycuda::complex<double> Gamma12 = 0.0435*(cos(-0.04)+I*sin(-0.04));
  // double DeltaGamma12 = 0.085; // ML : check this (from SuperIso)
  // double Asl_exp;
  //Asl_exp = tan(M_PI - pycuda::arg(-M12) - phi_Gamma12)*DeltaGamma12/(2*pycuda::abs(M12)/6.58e-13);
  //printf("Asl_SM %.3e \n", tan(M_PI + pycuda::arg(-M12_SM) - phi_Gamma12)*DeltaGamma12/(2*pycuda::abs(M12_SM)/6.58e-13));
  //printf("DMS_SM %.3e \n",2*pycuda::abs(M12_SM)/6.58e-13);
  //printf("phis_SM %.4e \n", (M_PI + pycuda::arg(M12_SM) - phi_Gamma12));//Gamma12));
    

  
  ////////////////////////////////////////////////////////////
  /////////////////7       Bd - Bdbar     ////////////////////
  ////////////////////////////////////////////////////////////
  M12_SM = GF*GF*VtbVtd*VtbVtd*mW*mW*S0_xt*eta_B_hat*B_1d*f_Bd*f_Bd*m_Bd / (12*M_PI*M_PI);

  //RR checked
  M12 = M12_SM + BSM_M12(VtbVtd, 2, ddLL_31, ddRR_31, duLL_31, pCha_1, pChap_3,
                         preCg_1,preCgp_1, preCg_4, preCg_5, newprea1, newprea2, newprea3, newprea4,
                         0.,0.,0.,0.,
                         alpha_m3[2], alpha_mb[2], alpha_mt[2], alpha_MS[2]);




  
  m12_obs[k0+3] = pycuda::arg(M12/M12_SM);
  m12_obs[k0+4] = pycuda::abs(M12/M12_SM);

  // Bd -Bdbar included in chi2
  chi2[row] += pow( (m12_obs[k0+3] - CONSTR_dphid)/CONSTR_sdphid, 2); //phid
  chi2_dphid[row] = pow( (m12_obs[k0+3] - CONSTR_dphid)/CONSTR_sdphid, 2);  

  // chi2[row] += pow( (m12_obs[k0+4] - CONSTR_RDMd)/CONSTR_sRDMd, 2);  //DMd
  chi2_RDMd[row] = pow( (m12_obs[k0+4] - CONSTR_RDMd)/CONSTR_sRDMd, 2);

  chi2[row] += pow((m12_obs[k0+4]/m12_obs[k0+1] - CONSTR_Rds)/CONSTR_sRds,2);  //DMd/Dms
  chi2_Rds[row] = pow((m12_obs[k0+4]/m12_obs[k0+1] - CONSTR_Rds)/CONSTR_sRds,2);
  
  
  
  // K0 -K0bar
  // M12_SM = 0.5*3.5e-15 *(1. + 2.*I*epsk_SM*sq2/kappa_e) ; // maxima gitanada
  
  // M12 = M12_SM + BSM_M12(VtsVtd, 3, ddLL_21, ddRR_21,duLL_21, pCha_1, pChap_3,
                         // preCg_1,preCgp_1, preCg_4, preCg_5, prea1, prea2, prea3, prea4,
                         // C1K, C1pK, C4K, C5K,
                         // alpha_m3[2], alpha_mb[2], alpha_mt[2], alpha_MS[2]);
  
  // double dmk = 2*fabs(pycuda::real(M12));///(2*pycuda::real(M12_SM));
  // double dmk_exp = 2*fabs(pycuda::real(M12_SM));
  
  // double epsk = kappa_e*pycuda::imag(M12)/(sq2*dmk_exp); /// Here SM is matched to EXP
  //double epsk_SM = kappa_e*abs(pycuda::imag(M12_SM))/(sq2*dmk_exp)
  // double r_dmk = dmk/dmk_exp;
 
  // m12_obs[k0+4] = epsk/epsk_SM;
  // m12_obs[k0+5] = r_dmk;
  //TODO: NOT included in chi2 for the moment
  // chi2[row] += pow((m12_obs[k0+4] - CONSTR_RepsK)/CONSTR_sRepsK,2) ;
  // chi2_RepsK[row] = pow((m12_obs[k0+4] - CONSTR_RepsK)/CONSTR_sRepsK,2) ;
  // if (epsk < 0) chi2[row] += 1.e9;
  // chi2[row] += pow((r_dmk - CONSTR_RDMK)/CONSTR_sRDMK,2) ;
  // chi2_RDMK[row] = pow((r_dmk - CONSTR_RDMK)/CONSTR_sRDMK,2) ;
  
  ////////////////////
  /// eps_K'/eps_K ///
  ////////////////////


  // double epsp = epsprime(Mg, mdL2,  mdR2, muR2, mu, tanb, tbcorr,param[j0 + 14],
                         // alpha_mb[2], alpha_mt[2], alpha_s, ddLL_12, ddRR_12);
  //exprime(-1*gluino_cte,ddLL_21, Mg*ddLR_21, xg);
  // WR 2.228e-3 valor sacado de (?)
  // m12_obs[k0+6] = epsp/2.228e-03;
  //printf ("%e \n", m12_obs[k0+6]);
  
  //TODO: Not included in chi2 for the moment 
  // chi2[row] +=(m12_obs[k0+6] - CONSTR_epek)*(m12_obs[k0+6] - CONSTR_epek) /
    // (CONSTR_sepek_TH*CONSTR_sepek_TH + CONSTR_sepek_EXP*CONSTR_sepek_EXP);
  // chi2_epek[row] = (m12_obs[k0+6] - CONSTR_epek)*(m12_obs[k0+6] - CONSTR_epek) /
    // (CONSTR_sepek_TH*CONSTR_sepek_TH + CONSTR_sepek_EXP*CONSTR_sepek_EXP);

   
  //////////////////////
  ///  D0 - D0bar   ////
  //////////////////////
  
  // M12 = BSM_M12(Vcb*Vub, 4, duLL_21, duRR_21,ddLL_21, pCha_1, pChap_3,
                // preCg_1,preCgp_1, preCg_4, preCg_5, prea1, prea2, prea3, prea4,
                // 0., 0., 0., 0.,
                // alpha_m3[2], alpha_mb[2], alpha_mt[2], alpha_MS[2]);
 
  // m12_obs[k0+7] = pycuda::real(M12);
  // m12_obs[k0+8] = pycuda::imag(M12);
  // WR abs(M12)
  // TODO: Not included in chi2 for the moment
  // chi2[row] += pow((pycuda::abs(M12)-CONSTR_DMD)/CONSTR_sDMD,2);
  // chi2_DMD[row] = pow((pycuda::abs(M12)-CONSTR_DMD)/CONSTR_sDMD,2);


  //////////////////////////////////////////////////////////////////////////////
  ////////////                                                       ///////////
  ////////////                      EDM's                           ////////////
  /////////////               Reviewer: Ramon Ruiz                    //////////
  ////////////                                                      ////////////
  //////////////////////////////////////////////////////////////////////////////
  
  /////////////////////////
  // Gluino contribution //
  /////////////////////////

  double xQ3 = 1/xgL;
  double xd3 = 1/xgR;
  double xu3 = muR2/Mg2; 

  double gc1 = -alpha_s*m_b*mu*tanb*fsg_d(xQ3,xd3)/(4*M_PI*Mg*Mg2*tbcorrY); 
  double gc1_2 = -alpha_s*m_b*mu*tanb*fsg_d2(xQ3,xd3)/(4*M_PI*Mg*Mg2*tbcorrY); 
  
  //RR:  only needed d_d, d_u, d_cd, d_cu -> Commenting rest for speed it up
  edm_obs[e0]     = gc1*pycuda::imag(ddLL_13*ddRR_31);            // d_edm
  // edm_obs[e0 + 1] = gc1*pycuda::imag(ddLL_23*ddRR_32);        // s_edm
  // edm_obs[e0 + 2] = gc1*pycuda::imag(ddLL_33*ddRR_33);        // b_edm

  edm_obs[e0 + 2] = gc1_2*pycuda::imag(ddLL_13*ddRR_31);      // d_cedm
  // edm_obs[e0 + 7] = gc1_2*pycuda::imag(ddLL_23*ddRR_32);      // s_cedm
  // edm_obs[e0 + 8] = gc1_2*pycuda::imag(ddLL_33*ddRR_33);      // b_cedm

  //Old RR correction -> C.1
  double gc2 = -alpha_s*(-2*fsg_d(xQ3, xu3))/(4*M_PI*Mg2*Mg); //RR -2 to translate fsg_d to fsg_u
  double gc2_2 = -alpha_s*fsg_d2(xQ3, xu3)/(4*M_PI*Mg2*Mg);
  
  // edm_obs[e0 + 3] = gc2*(m_c*Ac*pycuda::imag(duLL_12*duRR_21) + mt*At*pycuda::imag(duLL_13*duRR_31)) ; //u_edm
  // edm_obs[e0 + 4] = gc2*(m_c*Ac*pycuda::imag(duLL_21*duRR_12) + mt*At*pycuda::imag(duLL_23*duRR_32)) ; //c_edm
  // edm_obs[e0 + 5] = gc2*(m_u*Au*pycuda::imag(duLL_31*duRR_13) + m_c*Ac*pycuda::imag(duLL_31*duRR_13)); //t_edm
  // edm_obs[e0 + 9] = gc2_2*(m_c*Ac*pycuda::imag(duLL_12*duRR_21) + mt*At*pycuda::imag(duLL_13*duRR_31)) ; //u_cedm
  // edm_obs[e0 + 10] = gc2_2*(m_c*Ac*pycuda::imag(duLL_21*duRR_12) + mt*At*pycuda::imag(duLL_23*duRR_32)) ; //c_cedm
  // edm_obs[e0 + 11] = gc2_2*(m_u*Au*pycuda::imag(duLL_31*duRR_13) + m_c*Ac*pycuda::imag(duLL_31*duRR_13)); //t_cedm
  edm_obs[e0+1] = gc2*mt*At*pycuda::imag(duLL_13*duRR_31);
  edm_obs[e0+3] = gc2_2*mt*At*pycuda::imag(duLL_13*duRR_31);


  //////////////////////////////// 
  //Chargino-Contribution       //
  ///////////////////////////////
  
  double epsR = -2*alpha_s*mu*Mg*LoopG(xgR,xdLR)/(M_PI*mdR2); // RR correction
  double char_1 = alpha_2*m_b*mu*M_2*epsR*tanb2 /(8*M_PI*mtilde4*3*tbcorrY2);

  double hwc = char_1*fhwc(x_2L, xmuL);
  double hwc_2 = char_1*fhwc_2(x_2L, xmuL);

  edm_obs[e0] += hwc*pycuda::imag(duLL_13*ddRR_31); //d_edm
  edm_obs[e0+2] += hwc_2*pycuda::imag(duLL_13*ddRR_31); //d_cedm

  double char_2 = alpha_2*m_b*At*mt2*epsR*tanb2/(16*M_PI*mW2*mu2*mu*3*tbcorrY2);

  double llc = char_2*fchar(xLmu, xumu); 
  double llc_2 = char_2*fchar2(xLmu, xumu);

  edm_obs[e0] += llc * pycuda::imag(pycuda::conj(Vtd)*ddRR_31); //d_edm
  edm_obs[e0+2] += llc_2 * pycuda::imag(pycuda::conj(Vtd)*ddRR_31); //d_cedm


  //////////////////////////////// 
  //Charged-Higgs Contribution  //
  ///////////////////////////////

  double mbmt2 = m_b*mt2;

  double chc1   = -alpha_2*mbmt2*tbcorrn_prime*epsR*tanb*fH(yt)/(16*M_PI*massH2*mW2*3*tbcorrY2); //RR correction
  double chc1_2 = -alpha_2*mbmt2*tbcorrn_prime*epsR*tanb*fH2(yt)/(16*M_PI*massH2*mW2*3*tbcorrY2); //RR correction
  // C.4 first term
  edm_obs[e0] += chc1*pycuda::imag(pycuda::conj(Vtd)*ddRR_31);//d_edm
  // edm_obs[e0 + 1] += chc1*pycuda::imag(pycuda::conj(Vts)*ddRR_32);// s_edm
  // edm_obs[e0 + 2] += chc1*pycuda::imag(pycuda::conj(Vtb)*ddRR_33);// b_edm
  edm_obs[e0 + 2] += chc1_2*pycuda::imag(pycuda::conj(Vtd)*ddRR_31); // d_cedm
  // edm_obs[e0 + 7] += chc1_2*pycuda::imag(pycuda::conj(Vts)*ddRR_32);// s_cedm
  // edm_obs[e0 + 8] += chc1_2*pycuda::imag(pycuda::conj(Vtb)*ddRR_33);// b_cedm
  
  //Old based Buras paper
  // double chc2 =  alpha_2*mbmt2*At*mu*epsR*tanb*fsH(xmu)/(16*M_PI*mtilde4*mW2*tbcorr2);
  // double chc2_2 =  alpha_2*mbmt2*At*mu*epsR*tanb*fsH2(xmu)/(16*M_PI*mtilde4*mW2*tbcorr2);

  // RR C.4 second term
  double epsL_prime = -2*alpha_s*mu*Mg*LoopG(xgL,xuRL)/(M_PI*mtilde2); 

  double chc2   =  -alpha_2*mbmt2*epsL_prime*epsR*tanb2*fH(yt)/(16*M_PI*massH2*mW2*9*tbcorrY2);
  double chc2_2 =  -alpha_2*mbmt2*epsL_prime*epsR*tanb2*fH2(yt)/(16*M_PI*massH2*mW2*9*tbcorrY2);

  edm_obs[e0] += chc2*pycuda::imag(ddLL_13*ddRR_31);  //d_edm
  // edm_obs[e0 + 1] += chc2*pycuda::imag(ddLL_23*ddRR_32);
  // edm_obs[e0 + 2] += chc2*pycuda::imag(ddLL_33*ddRR_33);
  edm_obs[e0 + 2] += chc2_2*pycuda::imag(ddLL_13*ddRR_31);  //d_cedm
  // edm_obs[e0 + 7] += chc2_2*pycuda::imag(ddLL_23*ddRR_32);
  // edm_obs[e0 + 8] += chc2_2*pycuda::imag(ddLL_33*ddRR_33);

  //Old Based paper Buras 
  // double llc = alpha_2*mt2*At*mu*tanb*gsH(xmu)/(16*M_PI*mtilde4*mW2*tbcorr);
  // double llc_2 = alpha_2*mt2*At*mu*tanb*gsH2(xmu)/(16*M_PI*mtilde4*mW2*tbcorr);
  // Purely left-handed currents:
  // edm_obs[e0] += llc*m_d*pycuda::imag(pycuda::conj(Vtd)*duLL_13);//d_edm
  // edm_obs[e0 + 1] += llc*m_s*pycuda::imag(pycuda::conj(Vts)*duLL_23);//s_edm
  // edm_obs[e0 + 2] += 0.;//llc*m_b*pycuda::imag(pycuda::conj(Vtb)*duLL_33);//b_edm
  // edm_obs[e0 + 6] += llc_2*m_d*pycuda::imag(pycuda::conj(Vtd)*duLL_13);//d_cedm
  // edm_obs[e0 + 7] += llc_2*m_s*pycuda::imag(pycuda::conj(Vts)*duLL_23);//s_cedm
  // edm_obs[e0 + 8] += 0.;//llc_2*m_b*pycuda::imag(pycuda::conj(Vtb)*duLL_33);//t_cedm

  //dn
  //edm_obs[e0+12] = 1.97e-14*(1.4*(edm_obs[e0] - 0.25*edm_obs[e0 + 3]) + 0.83*(edm_obs[e0 + 9] + edm_obs[e0 + 6]) 
  //- 0.27*(edm_obs[e0 + 9] - edm_obs[e0 + 6])); //NU to e*cm

  //update RR end 2023
  double d_dout[2];
  double d_uout[2];
  double d_din[2] = {edm_obs[e0], edm_obs[e0 + 2]};
  double d_uin[2] = {edm_obs[e0+1], edm_obs[e0 + 3]};
  edm_RGE(d_din, d_dout, alpha_m2[2], alpha_mb[2], alpha_mt[2], alpha_MS[2], 1==1); //LO QCD only included
  edm_RGE(d_uin, d_uout, alpha_m2[2], alpha_mb[2], alpha_mt[2], alpha_MS[2], 1==0);


  // if(row%1000==0){
  //   printf("LoopG(1,1) = %.4f\n", LoopG(1., 1.));
  //   printf("LoopG(x, 1) = %.4f\n", LoopG(1.5, 1.));
  //   printf("LoopG(1, x) = %.4f\n", LoopG(1., 1.5));
  //   printf("LoopG(x, y) = %.4f\n", LoopG(xgL, 1./xdLR));
  //   printf("fsg(1, 1) = %.4f\n", fsg_d(1., 1.));
  //   printf("fsg2(1, 1) = %.4f\n", fsg_d2(1., 1.));
  //   printf("fsg(1, 3) = %.4f\n", fsg_d(1., 3.));
  //   printf("fsg2(1, 3) = %.4f\n", fsg_d2(1., 3.));
  //   printf("fsg(3, 1) = %.4f\n", fsg_d(3., 1.));
  //   printf("fsg2(3, 1) = %.4f\n", fsg_d2(3., 1.));
  //   printf("fsg(x, y) = %.4f\n", fsg_d(xQ3, xQ3));
  //   printf("fsg2(x, y) = %.4f\n", fsg_d2(xd3, xd3));
  //   printf("fsg(1, 3) = %.4f\n", fsg_d(xQ3, xd3));
  //   printf("fsg2(1, 3) = %.4f\n", fsg_d2(xQ3, xd3));
  //   printf("fhwc(1, 1) = %.4f\n", fhwc(1., 1.));
  //   printf("fhwc2(1, 1) = %.4f\n", fhwc_2(1., 1.));
  //   printf("fhwc(1, 1) = %.4f\n", fhwc(1., 3.));
  //   printf("fhwc2(1, 1) = %.4f\n", fhwc_2(1., 3.));
  //   printf("fhwc(3, 1) = %.4f\n", fhwc(3., 1.));
  //   printf("fhwc2(3, 1) = %.4f\n", fhwc_2(3., 1.));
  //   printf("fhwc(x, y) = %.4f\n", fhwc(x_2L, xmuL));
  //   printf("fhwc2(x, y) = %.4f\n", fhwc_2(x_2L, xmuL));
  //   printf("fchar(1, 1) = %.4f\n", fchar(1., 1.));
  //   printf("fchar2(1, 1) = %.4f\n", fchar2(1.,1.));
  //   printf("fchar(1, 3) = %.4f\n", fchar(1., 3.));
  //   printf("fchar2(1, 3) = %.4f\n", fchar2(1.,3.));
  //   printf("fchar(3, 1) = %.4f\n", fchar(3., 1.));
  //   printf("fchar2(3, 1) = %.4f\n", fchar2(3.,1.));
  //   printf("fchar(x, y) = %.4f\n", fchar(xLmu, xumu));
  //   printf("fchar2(x, y) = %.4f\n", fchar2(xLmu, xumu));
  //   printf("fH(1) = %.4f\n", fH(1.));
  //   printf("fH2(1) = %.4f\n", fH2(1.));
  //   printf("fH(x) = %.4f\n", fH(yt));
  //   printf("fH2(x) = %.4f\n", fH2(yt));
  // }

  // double dn_test;
  // double dp_test;
  // double dHg_test;
  
  //dn 
  edm_obs[e0 + 4] = 1.9732698e-14*(0.73*d_dout[0] - 0.18*d_uout[0]
                         + 0.20*d_dout[1] + 0.10 * d_uout[1]); //Gev^-1 to cm
  
  //Old
  // double dn_test;
  // double dp_test;
  // double dHg_test;
  // dn_test = 1.97e-14*(1.4*(edm_obs[e0] - 0.25*edm_obs[e0 + 3]) + 0.83*(edm_obs[e0 + 9] + edm_obs[e0 + 6])
  //                         - 0.27*(edm_obs[e0 + 9] - edm_obs[e0 + 6])); //NU to e*cm
  //
  // dp_test = 1.97e-14*(1.4*(edm_obs[e0 + 3] - 0.25*edm_obs[e0]) + 0.83*(edm_obs[e0 + 9] + edm_obs[e0 + 6])
  //            + 0.27*(edm_obs[e0 + 9] - edm_obs[e0 + 6]) ); //NU to e*cm

  

  
  //dp
  // edm_obs[e0+13] = 1.97e-14*(1.4*(edm_obs[e0 + 3] - 0.25*edm_obs[e0]) + 0.83*(edm_obs[e0 + 9] + edm_obs[e0 + 6]) 
  //            + 0.27*(edm_obs[e0 + 9] - edm_obs[e0 + 6]) ); //NU to e*cm
  //
  //update RR end 2023
  edm_obs[e0 + 5] = 1.9732698e-14*(0.73*d_uout[0] - 0.18*d_dout[0]
                          - 0.40*d_uout[1] - 0.049 * d_dout[1]); //Gev^-1 to cm


  //dHg
  // edm_obs[e0+14] = -2.4e-4*(1.9*edm_obs[e0+12] + 0.2*edm_obs[e0+13]);
  //update RR end 2023
  edm_obs[e0+6] = -2.1e-4*(1.9*edm_obs[e0+4] + 0.2*edm_obs[e0+5]);
  //TODO: Apparently dn and dn_test sign of diff

//   dHg_test = -2.4e-4*(1.9*dn_test + 0.2*dp_test);
//   printf("dHg %.3e dHg_test %.3e ", edm_obs[e0+14], dHg_test);
//   printf("dn %.3e dn_test %.3e ", edm_obs[e0+12], dn_test);
//   printf("dp %.3e dp_test %.3e \n", edm_obs[e0+13], dp_test);

  chi2[row] += pow((edm_obs[e0+4] - CONSTR_dn)/CONSTR_sdn, 2);
  chi2_dn[row] = pow((edm_obs[e0+4] - CONSTR_dn)/CONSTR_sdn, 2);

  chi2[row] += pow((edm_obs[e0+6] - CONSTR_dHg)/CONSTR_sdHg, 2);
  chi2_dHg[row] = pow((edm_obs[e0+6] - CONSTR_dHg)/CONSTR_sdHg, 2);


  //////////////////////////////////////////////////////////////////////////////
  ///                                                                        ///
  ///                     Phenomenological masses                            ///
  ///                                                                        ///
  //////////////////////////////////////////////////////////////////////////////
   
  pycuda::complex<double> XMN[4]  = {1000.,1000.,1000.,1000};
  pycuda::complex<double> XMch[2] = {1000,1000};
  pycuda::complex<double> XMstop[2] = {1000,1000};

  neutralino_guts(M_1,M_2,mu,beta,XMN);//1393.4091,2475.85984,3431.50481,1.54804100796,MN,XMN);
  chargino_guts(M_2,mu,beta, XMch);
  stop_guts(At,mtL2,mtR2,beta,mu,XMstop);
  //printf("XMstop[0], XMstop[1] \t%.3e,%.3e\n",pycuda::abs(XMstop[0]),pycuda::abs(XMstop[1]));

  ////// Added by Miriam : one-loop contribution to the CP-even Higgs boson (1601.01890v1) ///////
  double tan2alpha = tan(2*beta)*(MA2-MZ2)/(MA2+MZ2);
  double alpha = beta - 0.5*M_PI;//0.5*atan(tan2alpha); // TODO : check if alpha is used somwhere else (I think not)
  double cotbeta = 1/tan(beta);
  double cosalpha = cos(alpha);
  double sinalpha = sin(alpha);
  
  double mst12 = pycuda::abs(XMstop[0])*pycuda::abs(XMstop[0]); // TODO : Check
  double mst22 = pycuda::abs(XMstop[1])*pycuda::abs(XMstop[1]); // TODO : Check
  double higgshit = (mst12- mst22);
  double higgshit2 = higgshit*higgshit;
  double sinb_2 = sin(beta)*sin(beta);
  
  double higgs1 = 3*g2_2*mt4/(8*M_PI*M_PI*mW2*sinb_2);
  double higgs21 = 0.5*cosalpha*cosalpha*log(mst12*mst22/mt4);
  double higgs22,higgs23,higgs24,  higgs2324;
  
  if(mst12 == mst22){
  // if((mst12 - mst22) < 0.005){
      
    higgs22 = -cosalpha*(cosalpha*At + mu*sinalpha)*(At - mu*cotbeta)*1.0/mst22;
    higgs23 = 0.0;
    higgs24 = 0.0;
    higgs2324 = -1.0L/12.0L*(pow(At, 4)*pow(cosalpha, 2) - 2*pow(At, 3)*pow(cosalpha, 2)*cotbeta*mu + 
                             2*pow(At, 3)*cosalpha*mu*sinalpha + pow(At, 2)*pow(cosalpha, 2)*pow(cotbeta, 2)*pow(mu, 2) - 
                             4*pow(At, 2)*cosalpha*cotbeta*pow(mu, 2)*sinalpha + pow(At, 2)*pow(mu, 2)*pow(sinalpha, 2) + 
                             2*At*cosalpha*pow(cotbeta, 2)*pow(mu, 3)*sinalpha - 2*At*cotbeta*pow(mu, 3)*pow(sinalpha, 2) + 
                             pow(cotbeta, 2)*pow(mu, 4)*pow(sinalpha, 2))/pow(mst22, 2);
    
  }
  else 
  {
    higgs22 =  -cosalpha*(cosalpha*At + mu*sinalpha)*(At - mu*cotbeta)*log(mst12/mst22)/higgshit;
    higgs23 =  (At - mu*cotbeta)*(At - mu*cotbeta)*(cosalpha*At + mu*sinalpha)*(cosalpha*At + mu*sinalpha)/higgshit2;
    higgs24 =  1.0 - ((mst12 + mst22)*log(mst12/mst22)/(2*higgshit));
    higgs2324 = higgs23*higgs24;
    
  }
  
  //double DMh2 = higgs1*(higgs21 + higgs22 + higgs23*higgs24);
  double DMh2 = higgs1*(higgs21 + higgs22 + higgs2324);
  double Mh02 = 0.5*(MA2 + MZ2 - sqrt( (MA2-MZ2)*(MA2-MZ2) +4*MA2*MZ2*sin(2*beta)*sin(2*beta) ) );
  mass_obs[m0] =  sqrt(Mh02 + DMh2);
  // mass_obs[m0] =  sqrt(0.5*(MA2 + MZ2 + epsh/sinb_2 - sqrt( (MA2+MZ2)*(MA2+MZ2)*sin(2*beta)*sin(2*beta)+  CRAP1*CRAP1)));
  
  //WR : We are having problems here
  // chi2[row] += pow((mass_obs[m0]-CONSTR_mH)/CONSTR_smH,2); 
  // chi2_mH[row] = pow((mass_obs[m0]-CONSTR_mH)/CONSTR_smH,2); 


  //////////////////////////////////////////////////////////////////////////////
  ///                                                                        ///
  ///                     Wilson Coefficients                                ///
  ///               Reviewer: Ramon Ruiz, checked (falta C9, C10)            ///
  ///                                                                        ///
  //////////////////////////////////////////////////////////////////////////////


  //////////////////////////////
  ///     C7,C8,C7p,C8p       //
  // Based Teppei calculation //
  //  Author: Ramon Ruiz 2023 //
  //    RGE to 3 GeV and 2GeV //
  /////////////////////////////
  
  
  //1. Gluino Contribution
  pycuda::complex<double> C7_NP;
  pycuda::complex<double> C8_NP;
  pycuda::complex<double> C7p_NP;
  pycuda::complex<double> C8p_NP;

  double wc_gluino = M_PI*alpha_s*mu*tanb/(sq2*GF*Mg2*Mg*tbcorrY);
  // wc_obs[wc0] = wc_gluino*(ddLL_32)/(VtbVts)*fsg(xQ3, xd3); //C7
  // wc_obs[wc0+2] = wc_gluino*(ddLL_32)/(VtbVts)*fsg_2(xQ3,xd3); //C8
  C7_NP = wc_gluino*(ddLL_32)/(VtbVts)*fsg(xQ3, xd3); //C7
  C8_NP = wc_gluino*(ddLL_32)/(VtbVts)*fsg_2(xQ3,xd3); //C8
  

  // wc_obs[wc0+1] = wc_gluino*(ddRR_32)/(VtbVts)*fsg(xd3, xQ3); //C7p
  // wc_obs[wc0+3] = wc_gluino*(ddRR_32)/(VtbVts)*fsg_2(xd3, xQ3); //C7p
  C7p_NP = wc_gluino*(ddRR_32)/(VtbVts)*fsg(xd3, xQ3); //C7p
  C8p_NP = wc_gluino*(ddRR_32)/(VtbVts)*fsg_2(xd3, xQ3); //C8p

  //2. Chargino Contribution
  pycuda::complex<double> wc_charg_1 = mW2*mu*M_2*duLL_32/(mtilde4*pycuda::conj(Vts));
  pycuda::complex<double> wc_charg_2 = mt2*At/(2*pow(mu,3));
  
  // wc_obs[wc0]   += -tanb/(2*tbcorrY)*(wc_charg_1*fhwc(x_2L, xmuL) + wc_charg_2*fchar(xLmu, xumu));
  // wc_obs[wc0+2] += -tanb/(2*tbcorrY)*(wc_charg_1*fhwc_2(x_2L, xmuL) + wc_charg_2*fchar2(xLmu, xumu));
  C7_NP   += -tanb/(2*tbcorrY)*(wc_charg_1*fhwc(x_2L, xmuL) + wc_charg_2*fchar(xLmu, xumu));
  C8_NP += -tanb/(2*tbcorrY)*(wc_charg_1*fhwc_2(x_2L, xmuL) + wc_charg_2*fchar2(xLmu, xumu));
  

  //3. Charged-Higgs Contribution
  pycuda::complex<double> wc_hc = tbcorrn_prime/tbcorrY + (epsL_prime*tanb*ddLL_32)/(3*tbcorrY*pycuda::conj(Vts));
  wc_hc *= mt2/(8*massH2);

  // wc_obs[wc0]   += wc_hc*fH(yt); //C7
  // wc_obs[wc0+2] += wc_hc*fH2(yt); //C8
  C7_NP   += wc_hc*fH(yt); //C7
  C8_NP += wc_hc*fH2(yt); //C8

  
  // if (row%1000 == 0){
  //   printf("C7_NP = %e C8_NP =%e Mg = %.4f \n", pycuda::real(C7_NP), pycuda::real(C8_NP), Mg);
  // }

  pycuda::complex<double> cs_out_2gev[2];
  pycuda::complex<double> csp_out_2gev[2];
  pycuda::complex<double> cs_out_bmass[2];
  pycuda::complex<double> csp_out_bmass[2];
  pycuda::complex<double> cs_in[2] = {C7_NP, C8_NP};
  pycuda::complex<double> csp_in[2] = {C7p_NP, C8p_NP};

  //TO-DO Check this formulas w/ SuperISO

  c7c8_RGE(cs_in, cs_out_2gev, alpha_m2[2], alpha_mb[2], alpha_mt[2], alpha_MS[2]);
  c7c8_RGE(csp_in, csp_out_2gev, alpha_m2[2], alpha_mb[2], alpha_mt[2], alpha_MS[2]);

  c7c8_RGE(cs_in, cs_out_bmass, alpha_mb[2], alpha_mb[2], alpha_mt[2], alpha_MS[2]);
  c7c8_RGE(csp_in, csp_out_bmass, alpha_mb[2], alpha_mb[2], alpha_mt[2], alpha_MS[2]);
  
  //Saving \delta C7, \delta C8 at mb scale
  wc_obs[wc0] = cs_out_bmass[0];
  wc_obs[wc0 + 2] = cs_out_bmass[1];

  wc_obs[wc0 + 1] = csp_out_bmass[0];
  wc_obs[wc0 + 3] = csp_out_bmass[1];


  double Lambda78 = 0.089; //GeV i think it is 101 actually bcs of info below not really important
  double sLambda78 = 0.09; //1712.04959 btw 12 - 190

  pycuda::complex<double> C7_2 = cs_out_2gev[0] + C7_2_SM;
  pycuda::complex<double> C8_2 = cs_out_2gev[1] + C8_2_SM;

  pycuda::complex<double> C7p_2 = csp_out_2gev[0] + C7p_2_SM;
  pycuda::complex<double> C8p_2 = csp_out_2gev[1] + C8p_2_SM;

  pycuda::complex<double> num_wcfun = pycuda::conj(C7_2)*C8_2 + pycuda::conj(C7p_2)*C8p_2;
  pycuda::complex<double> den_wcfun = pycuda::abs(C7_2)*pycuda::abs(C7_2) + pycuda::abs(C7p_2)*pycuda::abs(C7p_2);
  pycuda::complex<double> wc_fun = pycuda::imag(num_wcfun/den_wcfun);

  wc_obs[wc0+4] = 4*M_PI*M_PI*alpha_m2[2]*Lambda78/m_b*wc_fun; //DACP
  

  double s_DACP_teo = sLambda78 * pycuda::abs(wc_obs[wc0+4])/Lambda78;

  //RR comment:
  //This is not good as is proportional to ACP so big ACP big error low chi2
  //maybe better to fix to 0 (?)
  
  // if (row%1000 == 0){
  //   printf("s_DACP_teo = %e \n", s_DACP_teo);
  // }

  //b -> s\gamma ACP to chi2
  chi2[row] += pow((pycuda::real(wc_obs[wc0 + 4]) - CONSTR_DACP)/ sqrt(pow(CONSTR_sDACP,2)+pow(s_DACP_teo,2)),2);
  chi2_DACP[row] = pow((pycuda::real(wc_obs[wc0 + 4]) - CONSTR_DACP)/ sqrt(pow(CONSTR_sDACP,2) + pow(s_DACP_teo,2)),2);

  //\Delta C7 evaluated at mb scale
  chi2[row] += pow((pycuda::abs(wc_obs[wc0]) - fabs(CONSTR_C7))/ CONSTR_sC7,2);
  chi2_C7[row] = pow((pycuda::abs(wc_obs[wc0]) - fabs(CONSTR_C7))/ CONSTR_sC7,2); 

  //OLD -> To implement C9 - C10
  //TODO (RR Q?) -> g3 in which scale (?) 
  //RR correction
  // wc_obs[wc0] = sqrt(2.0)/(4.*GF)*g3_2/mtilde2*(Mg*ddRL_32*g7_1(xg)/(m_b*VtbVts)
  //                             + Mg*mu*tanb*ddLL_32*g7_2(xg)/(mtilde2*tbcorr*VtbVts)); //C7 RR correction VtbVts + 4Gf/sqrt(2) (TODO: Check)
  // wc_obs[wc0 + 1] = sqrt(2.0)/(4.*GF)*g3_2/mtilde2*(Mg*ddLR_32*g7_1(xg)/(m_b*VtbVts)
  //                                 + Mg*mu*tanb*ddRR_32*g7_2(xg)/(mtilde2*tbcorr*VtbVts));//C7'
  // wc_obs[wc0 + 2] = sqrt(2.0)/(4.*GF)*g3_2/mtilde2*(Mg*ddRL_32*g8_1(xg)/(m_b*VtbVts)
  //                                 + Mg*mu*tanb*ddLL_32*g8_2(xg)/(mtilde2*tbcorr*VtbVts));//C8
  // wc_obs[wc0 + 3] = sqrt(2.0)/(4.*GF)*g3_2/mtilde2*(Mg*ddLR_32*g8_1(xg)/(m_b*VtbVts)
  //                                 + Mg*mu*tanb*ddRR_32*g8_2(xg)/(mtilde2*tbcorr*VtbVts));//C8'

  // wc_obs[wc0] += tbcorrn/(tbcorr)*1.0/2.*h7(yt);//C7 charged Higgs contribution

  // wc_obs[wc0 + 2] += tbcorrn/(tbcorr)*1.0/2.*h8(yt);//C8 charged Higgs contribution
  
  // wc_obs[wc0] += sqrt(2.0)*g2_2*tanb/(4.0*GF*mtilde2*tbcorr)*(duLL_32*mu*M_2/(VtbVts*mtilde2)*f7_1(x_2, xmu)
  //                                                           + mt2*At*mu/(mW2*mtilde2)*f7_2(xmu));//C7 chargino contribution

  // wc_obs[wc0 + 2] += sqrt(2.0)/(4.0*GF)*g2_2/(mtilde2)*tanb/tbcorr*(duLL_32*mu*M_2*f8_1(x_2, xmu)/(VtbVts*mtilde2)
  //                                                          + mt2*At*mu*f8_2(xmu)/(mW2*mtilde2));//C8 chargino contribution


  //////////////////////////
  ///        C9           //
  // 1205.1500 2201.04659 //
  /////////////////////////


  // C9_gluino_gamma (31)
  double Msq = (1./8.)*(4*mtilde + 2*sqrt(mdR2) + 2*sqrt(muR2));
  double Msq2 = Msq*Msq;

  //A.15 -> alpha_s (mw)
  pycuda::complex<double> preC9_gluino_gamma = -(4./9.)*sq2*M_PI*alpha_in[2]/(Msq2*VtbVts*GF);

  //C9 gluino Z (32) -> Proportional to LR_23 = 0
  // TODO: mstop_R input(?) -> try diagonalization (?)
  // pycuda::complex<double> preC9_gluino_Z_2MI = ddLR_33*ddLR_23*(4.*SW2-1.)/(Vtb*pycuda::conj(Vts))*(4./3.);

  //C9 WW\gamma & C9 HW\gamma & C9_HWZ & C9 WWZ & W box
  //TODO: Check for yukawa´s  (2 types superISO)
  //TODO: There is a order of the eigenvalues, check that V, U is syncronithed w/ this
  //TODO: Check contributions Masiero vs 1205.1500
  pycuda::complex<double> XMchar[2] = {0., 0.}; //Masses
  pycuda::complex<double> XMtest[2] = {0., 0.}; //Masses
  pycuda::complex<double> U[2][2] = {
                                       {0., 0.},
                                       {0., 0.}
                                      };
  pycuda::complex<double> V[2][2] = {
                                       {0., 0.},
                                       {0., 0.}
  }; 
  //Matrix Chargino sector
  
  chargino_matrix(M_2,mu,beta, XMchar, U, V);
  //Constants needed calculations C7, C9, C10
  //TODO: Account for imaginary part of x´s and mchar
  double x[2] = {pow(pycuda::abs(XMchar[0]),2)/Msq2, pow(pycuda::abs(XMchar[1]),2)/Msq2};
  double inv_x[2] = {Msq2/(pow(pycuda::abs(XMchar[0]),2)), Msq2/(pow(pycuda::abs(XMchar[1]),2))};
  double m_char[2] = {pycuda::abs(XMchar[0]), pycuda::abs(XMchar[1])};
  // double x_tr = mtR2/Msq2;
  // double inv_x_tr = Msq2/mtR2;

  //Yukawas
  // double g2 = sqrt(g2_2);
  // double yut = mt*g2/(sq2*sin(beta)*mW);
  // double yub = m_b*g2/(sq2*cos(beta)*mW);

  //Prefactors EWK sector contributions.
  pycuda::complex<double> sum_WWgamma = 0.;
  pycuda::complex<double> sum_WWZ = 0.;
  pycuda::complex<double> sum_boxW = 0.;

  // pycuda::complex<double> sum_WWZ_2MI = 0.; 
  //For the moment some C7 here to take advantage of the loop
  // pycuda::complex<double> sumC7_WWgamma = 0.;
  // pycuda::complex<double> sumC7_HWgamma = 0.;
  // pycuda::complex<double> sumC7_duLR23 = 0.; 
  //Following line commented
  // pycuda::complex<double> sum_HWgamma, sum_HWZ, boxHW;
  // pycuda::complex<double> sum_HWgamma_2 = 0.;
  // pycuda::complex<double> sum_HWZ_2 = 0.; 
  // pycuda::complex<double> sum_boxHW_2 = 0.; 

  pycuda::complex<double> preC9_WWgamma, preC9_WWZ, preC9_boxW;
  // pycuda::complex<double> preC9_WWgamma, preC9_WWZ, preC9_boxW, preC9_WWZ_2MI;
  // pycuda::complex<double> preC9_HWgamma, preC9_HWZ, preC9_boxHW;
  // pycuda::complex<double> preC9_HWgamma_2, preC9_HWZ_2, preC9_boxHW_2;

  preC9_WWgamma = -duLL_23*mW2/Msq2*(2./3.)*pycuda::conj(Vcs)/(pycuda::conj(Vts));

  //Proportional to duLR_23 commmented to speed up
  // preC9_HWgamma = duLR_23*mW2/Msq2*(2./3.)*yut/g2*pycuda::conj(Vcs)/pycuda::conj(Vts);
  // preC9_HWgamma_2 = duLR_23*pycuda::conj(Vcs)/Vts*yut/g2*mW2/Msq2;
  // preC9_HWZ = (4.*SW2-1.)*duLR_23*yut/g2*pycuda::conj(Vcs)/pycuda::conj(Vts)*(1./(4.*SW2));
  // preC9_HWZ_2 = duLR_23*pycuda::conj(Vcs)/pycuda::conj(Vts)*yut/g2*(1./(4.*SW2))*(4.*SW2-1.);
  // preC9_WWZ_2MI = (1-4.*SW2)*duLR_23*duLR_33*pycuda::conj(Vcs)/pycuda::conj(Vts)*(1./4.*SW2);
  // preC9_boxHW = - duLR_23 * pycuda::conj(Vcs)/pycuda::conj(Vts) * yut/(g2*4.*SW2)*mW2/Msq2;
  // preC9_boxHW_2 = - duLR_23 * pycuda::conj(Vcs)/pycuda::conj(Vts) * yut/(g2*4.*SW2)*mW2/Msq2;

  preC9_WWZ = (1. - 4.*SW2)*duLL_23*pycuda::conj(Vcs)/pycuda::conj(Vts)*(1./(4.*SW2));
  preC9_boxW = duLL_23*pycuda::conj(Vcs)/(pycuda::conj(Vts)) * mW2/(Msq2)*(1./SW2);



  for (int i=0; i<2; i++)
  {
    sum_WWgamma += V[i][0]*pycuda::conj(V[i][0])*(P_loop(3,1,2,x[i], x[i]) - 1./3.*P_loop(0,4,2,x[i], x[i]) + x[i]*P_loop(3,1,3,x[i], x[i]));

    //Following line commented -> C7 Teppei calculations
    // sumC7_WWgamma += V[i][0]*pycuda::conj(V[i][0])*( (3./2.)*P_loop(2,2,2,x[i],x[i]) + P_loop(1,3,2,x[i], x[i]));
    // sumC7_HWgamma += duLL_23*(U[i][1]*V[i][0]*(m_char[i]/m_b)*yub/g2) * (P_loop(2,1,2, x[i], x[i])+(2./3.)*P_loop(1,2,2, x[i], x[i]));
    // sumC7_duLR23 += V[i][0]*V[i][1]* (pow(inv_x[i],2.)* (f_1(inv_x[i]/inv_x_tr)-f_1(inv_x[i]))/((inv_x[i]/inv_x_tr)-inv_x[i]) ) ;
    
    // sum_HWgamma += pycuda::conj(V[i][1]*V[i][0])* (P_loop(3,1,2, x[i], x[i])- (1./3.)*P_loop(0,4,2, x[i], x[i]) +x[i]*P_loop(3,1,3, x[i], x[i]));
    for (int j=0; j<2; j++){
      //Following line commented
      // sum_HWZ += V[i][0]*pycuda::conj(V[j][1])*((pycuda::conj(U[i][0])*U[j][0])*sqrt(x[i]*x[j])*P_loop(1,1,2,x[i], x[j])+pycuda::conj(V[i][0])*V[j][0]*P_loop(1,1,1,x[i],x[j])-0.5*Krodelta(i,j)*P_loop(0,2,1,x[i],x[j]));
      sum_WWZ += V[i][0]*pycuda::conj(V[j][0])*( (pycuda::conj(U[i][0])*U[j][0])*sqrt(x[i]*x[j])*P_loop(1,1,2,x[i], x[j]) + pycuda::conj(V[i][0])*V[j][0]*P_loop(1,1,1,x[i],x[j])-Krodelta(i,j)*P_loop(0,2,1, x[i], x[j]));
      // printf("sum_WWZ_2MI=%.4f\n", pycuda::real(sum_WWZ));
      sum_boxW += pycuda::conj(V[i][0])*V[j][0]*V[i][0]*pycuda::conj(V[j][1])*f_loop(x[i], x[j], xsnu);
      //Following line commented Proportional to duLR23
      // sum_boxHW += pycuda::conj(V[i][0])*V[j][0]*V[i][0]*pycuda::conj(V[j][1])*f_loop(x[i], x[j], xsnu);
      // sum_WWZ_2MI += V[i][0]*pycuda::conj(V[j][0])*
                    //  (pycuda::conj(U[i][0])*U[j][0]*sqrt(x[i]*x[j])*P_loop(1,2,3,x[i], x[j])+
                      //  0.5*pycuda::conj(V[i][0])*V[j][0]*P_loop(1,2,2,x[i],x[j])
                      //  -(1./3.)*Krodelta(i,j)*P_loop(0,3,2, x[i], x[j])
                      // );
    // printf("sum_WWZ_2MI=%.4f\n", pycuda::real(sum_WWZ_2MI));

    }
  }

  pycuda::complex<double> C9_gluino_gamma = (1./3.)*preC9_gluino_gamma*P_loop(0,4,2, xg, xg)*ddLL_23;
  // pycuda::complex<double> C9_gluino_Z_2MI = preC9_gluino_Z_2MI*alpha_in[2]*inv_alpha_MZ/12.*P_loop(0,3,2, xg, xg);
  pycuda::complex<double> C9_WWgamma = preC9_WWgamma*sum_WWgamma;

  //Following line commented
  // pycuda::complex<double> C9_HWgamma = preC9_HWgamma*sum_HWgamma; //TODO: Check differences w/ 1205.1500
  // F_gammap(U, V, x, mtR2/Msq2, sum_HWgamma_2);
  // pycuda::complex<double> C9_HWgamma_2 = preC9_HWgamma_2*sum_HWgamma_2;

  //Following line commented
  // pycuda::complex<double> C9_HWZ = preC9_HWZ*sum_HWZ;  //TODO Check differences w/1205.1500
  // F_Zp(U, V, x, mtR2/Msq2, sum_HWZ_2);
  // pycuda::complex<double> C9_HWZ_2 = preC9_HWZ_2*sum_HWZ_2;
  
  //
  pycuda::complex<double> C9_WWZ = preC9_WWZ*sum_WWZ;
  pycuda::complex<double> C9_boxW = preC9_boxW*sum_boxW;

  //Following line commented
  // pycuda::complex<double> C9_boxHW = preC9_boxHW*sum_boxHW; //TODO: Check differences w/ 1205.1500
  // F_box(U, V, x, mtR2/Msq2, xsnu, sum_boxHW_2);
  // pycuda::complex<double> C9_boxHW_2 = preC9_boxHW_2*sum_boxHW_2;

  // pycuda::complex<double> C9_WWZ_2MI = preC9_WWZ_2MI*sum_WWZ_2MI;

  //Following line commented
  // wc_obs[wc0+5] = C9_gluino_gamma + C9_gluino_Z_2MI + C9_WWgamma + C9_HWgamma_2
  //          + C9_HWZ_2 + C9_WWZ + C9_boxW + C9_boxHW_2 + C9_WWZ_2MI;

  //LR_23 = 0
  pycuda::complex<double> C9_NMFV = C9_gluino_gamma + C9_WWgamma + C9_WWZ + C9_boxW;
  wc_obs[wc0+5] = C9_NMFV;
           

  // printf("C9_gluino_gamma=%4.2f\n", pycuda::real(C9_gluino_gamma));
  // printf("C9_gluino_Z_2MI=%4.2f\n", pycuda::real(C9_gluino_Z_2MI));
  // printf("C9_WWgamma=%4.2f\n", pycuda::real(C9_WWgamma));
  // printf("C9_WWZ=%4.2f\n", pycuda::real(C9_WWZ));
  // printf("C9_boxW=%4.2f\n", pycuda::real(C9_boxW));
  // printf("C9_HWgamma_2=%4.2f\n", pycuda::real(C9_HWgamma_2));
  // printf("C9_HWZ_2=%4.2f\n", pycuda::real(C9_HWZ_2));
  // printf("C9_NMFV=%4.2f\n", pycuda::real(C9_NMFV));

  //////////////////////////
  ///        C10          //
  //9900.6286, 1205.1500 //
  /////////////////////////

  // pycuda::complex<double> preC10_HWZ = 0.;
  // pycuda::complex<double> preC10_boxHW = 0.;
  // pycuda::complex<double> C10_gluino_2MI = 0.;
  // pycuda::complex<double> sumC10_HWZ = 0.; 
  // pycuda::complex<double> sumC10_boxHW = 0.;

  // preC10_HWZ = duLR_23*pycuda::conj(Vcs)/pycuda::conj(Vts) * (1./(4.*SW2)) * yut/g2 ;
  // preC10_boxHW = preC10_HWZ*mW2/Msq2;
  // C10_gluino_2MI = ((ddLR_33*ddRL_23)/(Vtb*pycuda::conj(Vts)))
  //                     *(4./3.)*P_loop(0,3,2,xg, xg)*alpha_in[2]/(12.*inv_alpha_MZ);


  // pycuda::complex<double> C10p_gluino_2MI = ((ddRL_33*ddLR_23)/(Vtb*pycuda::conj(Vts)))
  //                     *(4./3.)*P_loop(1,2,2,xg, xg)*alpha_in[2]/(12.*inv_alpha_MZ);
       

  // F_Zp(U, V, x, mtR2/Msq2, sumC10_HWZ);
  // F_box(U, V, x, mtR2/Msq2, xsnu, sumC10_boxHW);

  // pycuda::complex<double> C10_HWZ = preC10_HWZ*sumC10_HWZ;
  // pycuda::complex<double> C10_boxHW = preC10_boxHW*sumC10_boxHW;


  //LR_23 = 0
  pycuda::complex<double> C10_NMFV= -1.*C9_WWZ/(1. - 4.*SW2) - 1.*C9_boxW;
  wc_obs[wc0+6] = C10_NMFV;

  // wc_obs[wc0+6] = C10_gluino_2MI 
  //                  + C10_HWZ + C10_boxHW;
  //                 -1.* ( (C9_WWZ+ C9_WWZ_2MI)/(1.-4.*SW2) + C9_boxW) ;

  //DEBUG
  // if (row%1000 == 0){
  //   printf("C10_gluino_2MI = %e\n", pycuda::real(C10_gluino_2MI));
  //   printf("C10_HWZ = %e\n", pycuda::real(C10_HWZ));
  //   printf("C10_HW = %e\n", pycuda::real(C10_boxHW));
  //   printf("C10p = %e\n", pycuda::real(C10p_gluino_2MI));
    // printf("imag C10p = %e\n", pycuda::imag(C10p_gluino_2MI));
    // printf("C10_gluino_2MI = %e\n", pycuda::imag(C10_gluino_2MI));
    // printf("C10_HWZ = %e\n", pycuda::imag(C10_HWZ));
    // printf("C10_HW = %e\n", pycuda::imag(C10_boxHW));
    // printf("C9_WWZ=%e\n", pycuda::real(C9_WWZ));
    // printf("C9_WWZ_2MI=%e\n", pycuda::real(C9_WWZ_2MI));
    // printf("C9_boxW=%e\n", pycuda::real(C9_boxW));
    // printf("imag C9_WWZ=%e\n", pycuda::imag(C9_WWZ));
    // printf("imag C9_WWZ_2MI=%e\n", pycuda::imag(C9_WWZ_2MI));
    // printf("imag C9_boxW=%e\n", pycuda::imag(C9_boxW));
  // }
  //Calculate chi2 contribution taking into account correlation
  // double diff[4] = {
  //                   pycuda::real(wc_obs[wc0+5]) - C9_C10_nominal[0],
  //                   pycuda::imag(wc_obs[wc0+5]) - C9_C10_nominal[1],
  //                   pycuda::real(wc_obs[wc0+6]) - C9_C10_nominal[2],
  //                   pycuda::imag(wc_obs[wc0+6]) - C9_C10_nominal[3]
  // };
  
  // double aux[4] = {0., 0., 0., 0.};
  // for (int i=0; i<4; i++){
  //     for (int j=0; j<4; j++){
  //       aux[i] += diff[j]*C9_C10_covinv[j][i];
  //     }
  //     chi2_c9c10[row] += aux[i]*diff[i];
  // }
  // chi2[row] += chi2_c9c10[row];

  ////////////////////////////////
  ///        C7                 //
  //9900.6286, 1205.1500       //
  //OLD now Teppei calculation //
  /////////////////////////
  

  //Following line commented
  // not used only duLR_23 contribution -> RR (Q) why different (?)
  
  //Following line commented
  // pycuda::complex<double> preC7_gluino_gamma1, preC7_gluino_gamma2;

  //Following line commented
  // preC7_gluino_gamma1 = sq2/Msq2*GF*(1./3.)*(4./3.)*M_PI*alpha_in[2]/(pycuda::conj(Vts)*Vtb);
  //Following line commented
  // preC7_gluino_gamma2 =( (2.*ddLL_23 + m_s/m_b*ddRR_23)*(1./4.) * P_loop(1,3,2,xg,xg))
                        // - ddRL_23*P_loop(1,2,2,xg,xg)*(Mg/m_b);

  //Following line commented
  // pycuda::complex<double> C7_gluino_gamma = preC7_gluino_gamma1*preC7_gluino_gamma2;
  
  //Following line commented
  // pycuda::complex<double> C7_WWgamma = preC9_WWgamma/2.*sumC7_WWgamma;
  //Following line commented
  // pycuda::complex<double> C7_HWgamma = mW2/Msq2*pycuda::conj(Vcs)/(pycuda::conj(Vts))*sumC7_HWgamma;

  // pycuda::complex<double> preC7_duLR23 = duLR_23*pycuda::conj(Vcs)/pycuda::conj(Vts)*yut/g2
  //                                         *mW2/Msq2*(1./6.)*(1./6.)*(5+(mtR2/Msq2));


  // wc_obs[wc0] += preC7_duLR23*sumC7_duLR23;

  // double Lambda78 = 89*0.001; //1712.04959.pdf
  // double C8_SM = -0.095;
  // double C7_SM = -0.189;
  // pycuda::complex<double> wcprod = pycuda::conj(wc_obs[wc0]+C7_SM)*(wc_obs[wc0+2]+C8_SM);
  // wcprod += pycuda::conj(wc_obs[wc0+1])*wc_obs[wc0+3];
  // wcprod = wcprod/(pycuda::abs(wc_obs[wc0]+C7_SM)*pycuda::abs(wc_obs[wc0]+C7_SM) 
  //                  + pycuda::abs(wc_obs[wc0+1])*pycuda::abs(wc_obs[wc0+1]));

  //TODO: RR correction alpha scale mb -> check
  // wc_obs[wc0 + 4] = 4*M_PI*M_PI*alpha_mb[2]*Lambda78*pycuda::imag(wcprod)/m_b;
  
  // chi2[row] += pow((pycuda::abs(wc_obs[wc0 + 4]) - fabs(CONSTR_DACP))/ CONSTR_sDACP,2);
  // chi2_DACP[row] = pow(pycuda::abs(wc_obs[wc0 + 4]) - fabs(CONSTR_DACP)/ CONSTR_sDACP,2); // CHECK THIS 

  // chi2[row] += pow((pycuda::abs(wc_obs[wc0] + C7_SM) - fabs(C7_SM + CONSTR_C7))/ sCONSTR_C7,2);
  // chi2_C7[row] = pow((pycuda::abs(wc_obs[wc0] + C7_SM) - fabs(C7_SM + CONSTR_C7))/ sCONSTR_C7,2); // CHECK THIS

  //Following line commented
  // pycuda::complex<double> C7_NMFV;
  
  //TODO: Warning faltan contribuciones -> Amine/Nazilla style 
  //Following line commented
  // C7_NMFV = C7_gluino_gamma + C7_WWgamma;
                // + C7_HWgamma + C7_duLR23 ;

  // if (row%100 == 0){
    // printf("C7_new=%e, C7_old=%e\n", pycuda::real(C7_NMFV), pycuda::real(wc_obs[wc0]));
  // }
  


  //////////////////////////////////////////////////////////////////////////////
  ///                                                                        ///
  ///                              P--> ll                                   ///
  ///         Reviewer: Ramon Ruiz, checked (falta C10 correction)           ///
  ///                                                                        ///
  //////////////////////////////////////////////////////////////////////////////

  //Applying correction to Pll (C10)
  //Bsmm

  ///////////////////////
  ///////////////////////
  /////   P--> ll  //////
  ///////////////////////
  ///////////////////////

  double preCS0 = -alpha2_2*m_mu*tanb3/(MA2*4*mW2*tbcorr*tbcorrY*tbcorrlep); //RR correction
  double preCSR = preCS0/mdR2;
  double preCSL = preCS0/mdL2;

  double aCS = mt2*At*mu/mW2;
  double bCS = M_2*mu;
  double cCS = alpha_s/alpha_2*Mg*mu;

  //TODO: Warning -> old (0909.1333) as constraint
  //Check, why with new one one gets 3.3 alpha_s 0.1185 -> 0.1179
  // pycuda::complex<double> CA_SM = sgCA * g2_2*4*GF*(VtbVts*Y0_xt+ VcbVcs*YNL)/(16*pi2*sq2);
  //Ramon Update 2023
  // double pre_ca = .4802*pow(162.6/163.5, 1.5)*pow(0.1179/0.1184, 0.015) - 0.0112*pow(162.6/163.5, 0.86)*pow(0.1179/0.1184,-0.031);
  // double pre_ca = .4802*pow(162.6/163.5, 1.5)*pow(0.1179/0.1184, 0.015) - 0.0112*pow(162.6/163.5, 0.86)*pow(0.1179/0.1184,-0.031);
  // double pre_ca = 0.469*pow(162.6/163.5,1.5)*pow(0.1179/0.1184, 0.016);
  // pycuda::complex<double> CA_SM = sgCA * 2 * GF*GF*mW2*VtbVts*pre_ca/pi2;
  //Taken form Bsmm https://arxiv.org/pdf/1311.0903.pdf not from CA
  double Bs_SM1 = 3.65e-9*pow(mt/163.5, 3.02)*pow(alpha_in[2]/0.1184, 0.032)*pow(f_Bs/0.2277, 2)*pow(pycuda::abs(Vcs)/0.0424, 2)*pow((pycuda::abs(Vtb)*pycuda::abs(Vts))/pycuda::abs(Vcs)/0.980,2)*(1.616/1.615);
  double shit = sqrt(1 - 4*m_mu*m_mu/pow(m_Bs,2));
  pycuda::complex<double> CA_SM = sgCA*m_Bs/(2*m_mu)*sqrt(32*M_PI*Bs_SM1/(tau_Bs*f_Bs*f_Bs*pow(m_Bs,3)*shit));



  //Conversion from 3.46 of 0909.1333. and 137 http://superiso.in2p3.fr/superiso3.4.pdf
  
  // pycuda::complex<double> CA_NMFV = SW2*g2_2*4*GF*VtbVts/(16*pi2*sq2)*(wc_obs[wc0+6] - C10p_gluino_2MI) ;
  //C10p dependent on dulr23
  pycuda::complex<double> CA_NMFV = SW2*g2_2*4*GF*VtbVts/(16*pi2*sq2)*(wc_obs[wc0+6]) ;


  ///////////////
  // Bs -> mm  //
  ///////////////


  pycuda::complex<double> CSp = 2./3*4.*preCSR*cCS*(ddRR_32*LoopG(xgR,xdLR));
  // + xdLR * ddRR_13*ddLL_32*LoopH(xgR,xdLR)*tbcorr/tbcorrY);
  pycuda::complex<double> CS = 4.*preCSL*(2./3*cCS*ddLL_32*LoopG(xgL,1./xdLR) 
                                         - 1./8.*aCS*VtbVts*LoopF(xmuL, xuRL) 
                                         - 1./4.*bCS*duLL_32*LoopG(x_2L,xmuL));

  pycuda::complex<double> A = m_Bs*m_b/(m_s+m_b)* (-CS - CSp) + 2*m_mu/m_Bs*(CA_NMFV + CA_SM) ;
  double absA = pycuda::abs(A);

  pycuda::complex<double> B = m_Bs*m_b/(m_s+m_b) * (CS - CSp);
  double absB = pycuda::abs(B);

  //RR coment -> y_s se anula al dividir
  // Ratio Bsmm
  pll_obs[i0] = Pll_master(tau_Bs, f_Bs, m_Bs, m_mu, absA, absB)
    /Pll_master(tau_Bs, f_Bs, m_Bs, m_mu, 2*m_mu/m_Bs*abs(CA_SM), 0.);

  // double Bs_SM = Pll_master(tau_Bs, f_Bs, m_Bs, m_mu, 2*m_mu/m_Bs*abs(CA_SM), 0.);
  // double Bs_MSSM = Pll_master(tau_Bs, f_Bs, m_Bs, m_mu, absA, absB);
  // if (row%1000 == 0){
  //   printf("Bs_MSSM=%e\n", Bs_MSSM);
  //   printf("Bs_SM=%e\n", Bs_SM);
  //   printf("Bs_SM1 =%e\n", Bs_SM1);
  //   printf("CA =%.4f\n", pycuda::real(CA_SM));
  // //   printf("ratio=%e\n", pll_obs[i0]);
  // } 
  
  chi2[row]      += pow((pll_obs[i0] - CONSTR_Rsmm)/CONSTR_sRsmm, 2);
  chi2_Rsmm[row]  = pow((pll_obs[i0] - CONSTR_Rsmm)/CONSTR_sRsmm, 2);
                   
  
  
  //////////////
  // Bd -> mm //
  //////////////

  CSp = 2./3*4.*preCSR*cCS*(ddRR_31*LoopG(xgR,xdLR));
  CS = 4.*preCSL*(2./3*cCS*ddLL_31*LoopG(xgL,1./xdLR)
                 - 1./8.*aCS*VtbVtd*LoopF(xmuL,xuRL)
                 - 1./4.*bCS*duLL_31*LoopG(x_2L,xmuL));


  //TODO: Warning -> old (0909.1333) as constraint
  // CA_SM =  sgCA* g2_2*4*GF*(VtbVtd*Y0_xt + VcbVcd*YNL)/(16*pi2*sq2);
  double Bd_SM1 = 0.106e-9*pow(mt/163.5, 3.02)*pow(alpha_in[2]/0.1184, 0.032)*pow(f_Bd/0.1905, 2)*pow(pycuda::abs(VtbVtd)/.0088,2);
  shit = sqrt(1 - 4*m_mu*m_mu/pow(m_Bd,2));
  CA_SM = sgCA*m_Bd/(2*m_mu)*sqrt(32*M_PI*Bd_SM1/(tau_Bd*f_Bd*f_Bd*pow(m_Bd,3)*shit));

  CA_NMFV = SW2*g2_2*4*GF*VtbVtd/(16*pi2*sq2)*(wc_obs[wc0+6]) ;

  A = m_Bd*m_b/(m_d+m_b)* (-CS - CSp) + 2*m_mu/m_Bd*(CA_NMFV + CA_SM) ;
  absA = pycuda::abs(A);


  B = - m_Bd*m_b/(m_d+m_b) * (CS - CSp);
  absB = pycuda::abs(B);

  //RR coment -> y_d se anula al dividir
  pll_obs[i0+1] = Pll_master(tau_Bd, f_Bd, m_Bd, m_mu, absA, absB)
  /Pll_master(tau_Bd, f_Bd, m_Bd, m_mu, 2*m_mu/m_Bd*pycuda::abs(CA_SM), 0.);

  chi2[row]      += pow((pll_obs[i0+1] - CONSTR_Rdmm)/CONSTR_sRdmm, 2);
  chi2_Rdmm[row]  = pow((pll_obs[i0+1] - CONSTR_Rdmm)/CONSTR_sRdmm, 2);

  // double Bd_SM = Pll_master(tau_Bd, f_Bd, m_Bd, m_mu, 2*m_mu/m_Bd*abs(CA_SM), 0.);
  // double Bd_MSSM = Pll_master(tau_Bd, f_Bd, m_Bd, m_mu, absA, absB);
  // if (row%1000 == 0){
  //   printf("Bd_MSSM=%e\n", Bd_MSSM);
  //   printf("Bd_SM=%e\n", Bd_SM);
  //   printf("Bd_SM1 =%e\n", Bd_SM1);
  //   printf("CA =%.4f\n", pycuda::real(CA_SM));
  //   printf("ratio=%e\n", pll_obs[i0]);
  // } 

  /////////////////////
  // B+ -> \tau \nu //
  ////////////////////
  //WR check i0 + 2 if one wants more indexes

  double epsL_tau = - 2.*alpha_s*mu*Mg/(M_PI*mtilde2)*LoopG(xgL, 1./xdLR);
  pycuda::complex<double> rlnu  = 1.- m_Bu*m_Bu*tanb2*(tbcorrY + (epsL_tau*tanb*duLL_31)/(3.*Vub))/(massH2*tbcorr*tbcorrY*tbcorrlep);
  pll_obs[i0 + 2 ] = pycuda::abs(rlnu)*pycuda::abs(rlnu);

  chi2[row] += pow((pll_obs[i0+4] - CONSTR_RTAUNU)/CONSTR_sRTAUNU,2);
  chi2_RTAUNU[row] = pow((pll_obs[i0+4] - CONSTR_RTAUNU)/CONSTR_sRTAUNU,2);

  // rlnu  = 1- m_Kp*m_Kp*tanb2/((massH2)*tbcorr*tbcorrlep);
  
  // pll_obs[i0 + 5 ] = rlnu*rlnu;
  //TODO not included in chi2 for the moment
  // chi2[row] += pow((pll_obs[i0+5] - CONSTR_KMUNU)/CONSTR_sKMUNU,2);
  // chi2_KMUNU[row] = pow((pll_obs[i0+5] - CONSTR_KMUNU)/CONSTR_sKMUNU,2);

  // double preCS0 = -alpha_2*alpha_2*m_mu*tanb3/(MA2*4*mW2*tbcorr2*tbcorrlep);
  // double preCSR = preCS0/mdR2;
  // double preCSL = preCS0/mdL2;
  //
  // double aCS = mt2*At*mu/mW2;
  // double bCS = M_2*mu;
  // double cCS = alpha_s/alpha_2*Mg*mu;
  // pycuda::complex<double> CA_SM = sgCA*2*m_mu/m_Bs * g2_2*4*GF*(VtbVts*Y0_xt+ VcbVcs*YNL)/(16*pi2*sq2);

  // 0909.1333 style
  // Bs -> mm
  // pycuda::complex<double> CS = preCS*(aCS * VtbVts*h3mu + bCS*duLL_32*h42mu - cCS*ddLL_32*h1g);
  // pycuda::complex<double> CSp = -preCS*cCS*ddRR_32*h1g;

  
  // Bd -> mm
  // CS = preCS*(aCS * VtbVtd*h3mu + bCS*duLL_31*h42mu - cCS*ddLL_31*h1g);
  // CSp = -preCS*cCS*ddRR_31*h1g;
  // CA_SM =  sgCA*2*m_mu/m_Bd * g2_2*4*GF*(VtbVtd*Y0_xt + VcbVcd*YNL)/(16*pi2*sq2);  
  // absA = pycuda::abs(m_Bd*(-CS-CSp)/rh + CA_SM );
  // absB = pycuda::abs(m_Bs*(CS - CSp));
  // pll_obs[i0+1] = Pll_master(tau_Bd, f_Bd, m_Bd, m_mu, absA, absB)
    // /Pll_master(tau_Bd, f_Bd, m_Bd, m_mu, pycuda::abs(CA_SM), 0.);
  
  //chi2[row] += (pll_obs[i0 +1 ] - CONSTR_Rdmm)*(pll_obs[i0 +1 ] - CONSTR_Rdmm) 
  //  /(CONSTR_sRdmm_TH*CONSTR_sRdmm_TH + CONSTR_sRdmm_EXP*CONSTR_sRdmm_EXP);
  //
  
  // Teppei style /// mu real

  ///////////////
  // Bs -> mm  //
  ///////////////
  // pycuda::complex<double> CSp = 2./3*4.*preCSR*cCS*(ddRR_32*LoopG(xgR,xdLR)); 
  // + xdLR * ddRR_13*ddLL_32*LoopH(xgR,xdLR)*tbcorr/tbcorrY);
  // pycuda::complex<double> CS = 4.*preCSL*(2./3*cCS*ddLL_32*LoopG(xgL,1./xdLR) 
                                         // - 1./8*aCS*VtbVts*tbcorrY*tbcorrY/tbcorr2*LoopF(xmuL,xtRdL) 
                                         // + .25*bCS*duLL_32*LoopG(x_2L,xmuL));

  //TODO: Warning -> C10 correction 
  // double absA = pycuda::abs(m_Bs*(-CS-CSp)/rh + CA_SM);
  //TODO: Warning check c10 correction is well applied
  // double absB = pycuda::abs(m_Bs*(CS - CSp));
  // pll_obs[i0] = Pll_master(tau_Bs, f_Bs, m_Bs, m_mu, absA, absB)
    // /Pll_master(tau_Bs, f_Bs, m_Bs, m_mu, pycuda::abs(CA_SM), 0.);
  // chi2[row] += (pll_obs[i0] - CONSTR_Rsmm)*(pll_obs[i0] - CONSTR_Rsmm)
   // /(CONSTR_sRsmm_TH*CONSTR_sRsmm_TH + CONSTR_sRsmm_EXP*CONSTR_sRsmm_EXP);
  
  //////////////
  // Bd -> mm //
  //////////////

  // CSp = 2./3*4.*preCSR*cCS*(ddRR_31*LoopG(xgR,xdLR));
  // CS = 4.*preCSL*(2./3*cCS*ddLL_31*LoopG(xgL,1./xdLR)
                 // - 1./8*aCS*VtbVtd*tbcorrY*tbcorrY/tbcorr2*LoopF(xmuL,xtRdL)
                 // + .25*bCS*duLL_31*LoopG(x_2L,xmuL));
  // CA_SM =  sgCA*2*m_mu/m_Bd * g2_2*4*GF*(VtbVtd*Y0_xt + VcbVcd*YNL)/(16*pi2*sq2);

  //TODO: Warning check c10 correction
  // absA = pycuda::abs(m_Bd*(-CS-CSp)/rh + CA_SM );
  // absB = pycuda::abs(m_Bs*(CS - CSp));
  // pll_obs[i0+1] = Pll_master(tau_Bd, f_Bd, m_Bd, m_mu, absA, absB)
    // /Pll_master(tau_Bd, f_Bd, m_Bd, m_mu, pycuda::abs(CA_SM), 0.);

  // chi2[row] += (pll_obs[i0 +1 ] - CONSTR_Rdmm)*(pll_obs[i0 +1 ] - CONSTR_Rdmm)
   // /(CONSTR_sRdmm_TH*CONSTR_sRdmm_TH + CONSTR_sRdmm_EXP*CONSTR_sRdmm_EXP);

  ///////////////
  // K0 -> mm  //
  ///////////////
  //TODO: Warning -> Commented from here -> Kaons
  // /* 
  // CS = 2./3*4.*preCSR*cCS*(ddRR_12*LoopG(xgR,xdLR)
                          // + m_b/m_s* xdLR * ddRR_13*ddLL_32*LoopH(xgR,xdLR)*tbcorr/tbcorrY);

  // CSp = 4.*preCSL*(2./3*cCS*(ddLL_12*LoopG(xgL,1./xdLR)
                            // + m_b/m_s* (1.0/xdLR) * ddLL_13*ddRR_32*LoopH(xgL,1./xdLR)*tbcorr/tbcorrY) 
                  // - 1./8*aCS*VtdVts*tbcorrY*tbcorrY/tbcorr2*LoopF(xmuL,xtRdL) + .25*bCS*duLL_12*LoopG(x_2L,xmuL));
  // CA_SM =  sgCA*2*m_mu/m_K * g2_2*4*GF*(VtdVts*Y0_xt+ VcdVcs*YNL)/(16*pi2*sq2);
 
  //  printf("%e %e %e %e %e\n", 
  //      pycuda::real(CS-CSp),pycuda::imag(CS-CSp) , pycuda::real(-CS-CSp),pycuda::imag(-CS-CSp), pycuda::abs(CA_SM));
  
  // KS
  // prea1 = pycuda::imag(m_K*(-CS-CSp)/rh + CA_SM);// A_KS, just recycling memory positions thats why the name is weir
  // C1K = pycuda::real(-m_K*(CS - CSp))  + ASgg; // B_KS, just recycling memory positions thats why the name is weird
 
  // absA = pycuda::imag(m_K*(-CS-CSp)/rh + CA_SM);//prea1;
  // absB = pycuda::abs( pycuda::real(-m_K*(CS - CSp))  + ASgg); // Bgg + BBSM
  // pll_obs[i0+2]  = 2* Pll_master(tau_KS, f_K, m_K, m_mu, absA, absB);// factor 2 w.r.t Bs,d according to Teppei
  //printf ("KS %e %e\n", absA, absB);
  
  // KL 
  // C1pK = pycuda::real(-m_K*(-CS-CSp)/rh - CA_SM) + sign_Agg*ALgg;// A_KL, just recycling memory positions
  // prea2 = pycuda::imag(m_K*(CS - CSp));

  //WR: Take care of sign_Agg now included in mu
  // absA = pycuda::abs( pycuda::real(-m_K*(-CS-CSp)/rh - CA_SM) + sign_Agg*ALgg );
  // absB = pycuda::imag(m_K*(CS - CSp));
  // printf ("KL absA  %e absB %e coeff %e %e %e %e\n", absA, absB, 
  //      pycuda::real(CS-CSp),pycuda::imag(CS-CSp) , pycuda::real(-CS-CSp),pycuda::imag(-CS-CSp), pycuda::abs(CA_SM));

  // pll_obs[i0+3]  = 2* Pll_master(tau_KL, f_K, m_K, m_mu, absA, absB);// factor 2 w.r.t Bs,d according to Teppei
  
  //WR: Take care of sign_Agg now included in mu
  // if (sign_Agg > 0) {
    //TODO: Not included in global chi2 for the moment
    // chi2[row] += (pll_obs[i0 + 3] - CONSTR_KLmm_EXP)* (pll_obs[i0 +3 ] - CONSTR_KLmm_EXP)
                      // /(CONSTR_sKLmm_TH_plus*CONSTR_sKLmm_TH_plus + CONSTR_sKLmm_EXP*CONSTR_sKLmm_EXP);

    // chi2_KLmm[row] = (pll_obs[i0 + 3] - CONSTR_KLmm_EXP)* (pll_obs[i0 +3 ] - CONSTR_KLmm_EXP)
      // /(CONSTR_sKLmm_TH_plus*CONSTR_sKLmm_TH_plus + CONSTR_sKLmm_EXP*CONSTR_sKLmm_EXP);
  // }
  // else{ 
    //TODO Not included in chi2 for the moment
    // chi2[row] += (pll_obs[i0 +3 ] - CONSTR_KLmm_EXP)*(pll_obs[i0 +3 ] - CONSTR_KLmm_EXP)
         // /(CONSTR_sKLmm_TH_minus*CONSTR_sKLmm_TH_minus + CONSTR_sKLmm_EXP*CONSTR_sKLmm_EXP);            
    // chi2_KLmm[row] = (pll_obs[i0 +3 ] - CONSTR_KLmm_EXP)*(pll_obs[i0 +3 ] - CONSTR_KLmm_EXP)
      // /(CONSTR_sKLmm_TH_minus*CONSTR_sKLmm_TH_minus + CONSTR_sKLmm_EXP*CONSTR_sKLmm_EXP);
  // }
  
  // Interference KS-KL
  
  // C4K = f_K*f_K * m_K*m_K*m_K * sqrt(1 - 4*m_mu*m_mu/(m_K*m_K)) *I/ (8*M_PI)*
    // (prea1*C1pK - (1 - 4*m_mu*m_mu/(m_K*m_K))*pycuda::conj(C1K)*prea2  ) ;
  
  
  // pll_obs[i0+6] = pycuda::real(C4K);
  // pll_obs[i0+7] = pycuda::imag(C4K);
  // Rtaunu, R(K->mu nu) (3.47 0909.1333)
  // double rlnu  = 1- m_Bu*m_Bu*tanb2/((massH2)*tbcorr*tbcorrlep);
  // WR check dimensions first [ ]


  //////////////////////////////////////////////////////////////////////////////
  //                                                                          //
  //                      Constraints and Cost Penalties                      //
  //                                                                          //
  //////////////////////////////////////////////////////////////////////////////
  
  mass_obs[m0+1] = pycuda::abs(XMch[0]);
  mass_obs[m0+2] = pycuda::abs(XMch[1]);
  mass_obs[m0+3] = pycuda::abs(XMN[0]); // TODO : account for imaginary values of the masses
  mass_obs[m0+4] = pycuda::abs(XMN[1]);
  mass_obs[m0+5] = pycuda::abs(XMN[2]);
  mass_obs[m0+6] = pycuda::abs(XMN[3]);
  mass_obs[m0+7] = M_1;
  mass_obs[m0+8] = M_2;

  // Adding Constraint for duLR_23 from Vacuum stability
  // Also important for duLR_13 
  // double constraint = 2.* mt/Msq; // Factor 2 from Masiero paper
  // double mod2 = pycuda::real(duLR_23)*pycuda::real(duLR_23) + pycuda::imag(duLR_23)*pycuda::imag(duLR_23);

  // //neutral LSP
  if ((mass_obs[m0+1] + 0.16) < mass_obs[m0+3]) chi2[row] += 1e9;
  // else if (sqrt(mod2) > constraint) chi2[row] += 1e9;
  //TODO Warning -> wo cost penalties
  // else if (mtilde < mass_obs[m0+3]) chi2[row] += 1e9; // ML: check this
  // else if (mdR2 < mass_obs[m0+3]* mass_obs[m0+3]) chi2[row] += 1e9; // ML: check this 
  // else if (mtR2 < mass_obs[m0+3]* mass_obs[m0+3]) chi2[row] += 1e9; // ML: check this 
  // else if (mdL2 < mass_obs[m0+3]* mass_obs[m0+3]) chi2[row] += 1e9; // ML: check this 
  // else if (mtL2 < mass_obs[m0+3]* mass_obs[m0+3]) chi2[row] += 1e9; // ML: check this 
  //WR: Update on this?
  else if (tanb > 4.4 + exp(MA/415.))    chi2[row] += 1e9; //MA tanb plane
  // else if (mdR2 < 81e04 ) chi2[row] += 1e9;
  // else if (mdL2 < 81e04 ) chi2[row] += 1e9;
  else if (MA < 450 ) chi2[row] += 1e9;
  // else if (fabs(rh -1) > 0.8 ) chi2[row] += 1e9;
  if (pycuda::abs(ddRR_13) > 0.85) chi2[row] += 1e4*(pycuda::abs(ddRR_13)) ;
  if (pycuda::abs(ddRR_23) > 0.85) chi2[row] += 1e4*(pycuda::abs(ddRR_23)) ;
  if (pycuda::abs(duRR_13) > 0.85) chi2[row] += 1e4*(pycuda::abs(duRR_13)) ;
  if (pycuda::abs(duRR_23) > 0.85) chi2[row] += 1e4*(pycuda::abs(duRR_23)) ;

  if (pycuda::abs(ddLL_13) > 0.85) chi2[row] += 1e4*(pycuda::abs(ddLL_13)) ;
  if (pycuda::abs(ddLL_23) > 0.85) chi2[row] += 1e4*(pycuda::abs(ddLL_23)) ;
  if (pycuda::abs(duLL_13) > 0.85) chi2[row] += 1e4*(pycuda::abs(duLL_13)) ;
  if (pycuda::abs(duLL_23) > 0.85) chi2[row] += 1e4*(pycuda::abs(duLL_23)) ;

  // if (pycuda::imag(ddRR_12) > 0.85 || pycuda::imag(ddLL_12) > 0.85) chi2[row] += 1000*(pycuda::imag(ddRR_12) + pycuda::imag(ddLL_12));
  // if (pycuda::imag(ddRR_13) > 0.85 ||pycuda::imag(ddLL_13) > 0.85) chi2[row] += 1000*(pycuda::imag(ddRR_13) + pycuda::imag(ddLL_13));
  // if (pycuda::imag(ddRR_23) > 0.85 ||pycuda::imag(ddLL_23) > 0.85) chi2[row] += 1000*(pycuda::imag(ddRR_23) + pycuda::imag(ddLL_23));
  // if (pycuda::imag(duRR_12) > 0.85 ||pycuda::imag(duLL_12) > 0.85) chi2[row] += 1000*(pycuda::imag(duRR_12) + pycuda::imag(duLL_12));
  // if (pycuda::imag(duRR_13) > 0.85 ||pycuda::imag(duLL_13) > 0.85) chi2[row] += 1000*(pycuda::imag(duRR_13) + pycuda::imag(duLL_13));
  // if (pycuda::imag(duRR_23) > 0.85 ||pycuda::imag(duLL_23) > 0.85) chi2[row] += 1000*(pycuda::imag(duRR_23) + pycuda::imag(duLL_23));

  // if (pycuda::real(duLR_23) > 0.85 || pycuda::real(duLR_33) > 0.85) chi2[row] += 1000*(pycuda::real(duLR_23) + pycuda::real(duLR_33));
  // if (pycuda::real(ddLR_23) > 0.85 || pycuda::real(ddLR_33) > 0.85 || pycuda::real(ddLR_21) > 0.85) chi2[row] += 1000*(pycuda::real(ddLR_23) + pycuda::real(ddLR_33) + pycuda::real(ddLR_21));
  // if (pycuda::real(ddRL_23) > 0.85 || pycuda::real(ddRL_33) > 0.85) chi2[row] += 1000*(pycuda::real(ddRL_23) + pycuda::real(ddRL_33) );
  // if (pycuda::imag(duLR_23) > 0.85 || pycuda::imag(duLR_33) > 0.85) chi2[row] += 1000*(pycuda::imag(duLR_23) + pycuda::imag(duLR_33));
  // if (pycuda::imag(ddLR_23) > 0.85 || pycuda::imag(ddLR_33) > 0.85 || pycuda::imag(ddLR_21) > 0.85) chi2[row] += 1000*(pycuda::imag(ddLR_23) + pycuda::imag(ddLR_33) + pycuda::imag(ddLR_21));
  // if (pycuda::imag(ddRL_23) > 0.85 || pycuda::imag(ddRL_33) > 0.85) chi2[row] += 1000*(pycuda::imag(ddRL_23) + pycuda::imag(ddRL_33) );

  // additional costs not really chi2, to be used for sampling
  cost[row] = chi2[row];

  /// Large cost penalty for events outside delta boundaries
  
  //if (pycuda::abs(duRR_12) > 0.5 ||pycuda::abs(duLL_12) > 0.5  ) 
  //cost[row] += 1000*(pycuda::abs(duRR_12) + pycuda::abs(duLL_12));
  
  //if (pll_obs[i0+2] < 1e-10) cost[row] += pow((pll_obs[i0+2]-1e-10)/2e-12,2);
  //if (pll_obs[i0+3] > 1e-08 ) cost[row] += 1e11* pll_obs[i0+3];

  return;
  //Miriam penalties
  // if ((mass_obs[m0+1] + 0.16) < mass_obs[m0+3]) chi2[row] += 1e9;
  // else if (mtilde < mass_obs[m0+3]) chi2[row] += 1e9; // ML: check this
  // else if (mdR2 < mass_obs[m0+3]* mass_obs[m0+3]) chi2[row] += 1e9; // ML: check this
  // else if (mtR2 < mass_obs[m0+3]* mass_obs[m0+3]) chi2[row] += 1e9; // ML: check this
  // else if (mdL2 < mass_obs[m0+3]* mass_obs[m0+3]) chi2[row] += 1e9; // ML: check this
  // else if (mtL2 < mass_obs[m0+3]* mass_obs[m0+3]) chi2[row] += 1e9; // ML: check this
  // else if (tanb > 4.4 + exp(MA/415.))    chi2[row] += 1e9; //MA tanb plane
  // else if (mdR2 < 81e04 ) chi2[row] += 1e9;
  // else if (mdL2 < 81e04 ) chi2[row] += 1e9;
  // else if (MA < 450 ) chi2[row] += 1e9;
  // else if (fabs(rh -1) > 0.8 ) chi2[row] += 1e9;
  // if (pycuda::abs(ddRR_12) > 0.3 ||pycuda::abs(ddLL_12) > 0.3) chi2[row] += 1000*(pycuda::abs(ddRR_12) + pycuda::abs(ddLL_12));
  // if (pycuda::abs(ddRR_13) > 0.3 ||pycuda::abs(ddLL_13) > 0.3) chi2[row] += 1000*(pycuda::abs(ddRR_13) + pycuda::abs(ddLL_13));
  // if (pycuda::abs(ddRR_23) > 0.3 ||pycuda::abs(ddLL_23) > 0.3) chi2[row] += 1000*(pycuda::abs(ddRR_23) + pycuda::abs(ddLL_23));
  // if (pycuda::abs(duRR_12) > 0.3 ||pycuda::abs(duLL_12) > 0.3) chi2[row] += 1000*(pycuda::abs(duRR_12) + pycuda::abs(duLL_12));
  // if (pycuda::abs(duRR_13) > 0.3 ||pycuda::abs(duLL_13) > 0.3) chi2[row] += 1000*(pycuda::abs(duRR_13) + pycuda::abs(duLL_13));
  // if (pycuda::abs(duRR_23) > 0.3 ||pycuda::abs(duLL_23) > 0.3) chi2[row] += 1000*(pycuda::abs(duRR_23) + pycuda::abs(duLL_23));
  
}

