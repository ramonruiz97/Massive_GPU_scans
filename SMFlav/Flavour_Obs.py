# -*- coding: utf-8 -*-
############################################################################################
#                                       FLAVOUR OBSERVABLES                                #
#                       authors = D. Martinez Santos, R.A. Ruiz Fernandez                  #
############################################################################################


### Modules needed
import numpy as np
import sympy as syp
import cPickle
from Flavin import *
from CKMfit import *

# from Inami_Lim import S0_xt, S0_xt_xc 

############################################################################################
#                                  Experimental Inputs                                     #
############################################################################################

# 1. Oscillation Frequencies
DMsExp = Mag('DMsExp', 17.7650, 0.0057) #Own Calc. w/ 2104.04421 
DMdExp = Mag('DMdExp', 0.5065, 0.0019)  #HFLAV2021
DGs_SM = Mag("DGs_SM", 0.091, 0.013) #ps-1 (1912.0762)
asl_SM = Mag("Asl_SM", 2.06e-5, 0.18e-5) #1912.0762
DGs_Exp = Mag("DGs_Exp", 0.085, 0.005) #ps-1  HFLAV 2023
asl_Exp = Mag("Asl_Exp", -0.0006, 0.0028)  #HFLAV 2023
Betas_Exp = Mag("Betas_Exp", -0.039, 0.016) #including latest BsJpsiphi prelim comb hflav
Beta_Exp = Mag("Beta_Exp", 0.787, 0.016) #Including latest B0JpsiKs0 prelim comb hflav

# 2. B decays
Bsmm_c = Mag('Bsmm_c', 3.660, 0.051) #1908.07011
Bdmm_c = Mag('Bdmm_c', 0.1027, 0.0013) #1908.07011
Bsmm_exp = Mag("Bsmm_Exp", 3.52, 0.33) #Own calc
Bdmm_exp = Mag("Bdmm_Exp", 0.41, 0.46) #Own calc
# BR_Xsll0 = Mag('BR_Xsll0', 1.62e-6, np.sqrt(0.09**2-0.005**2)*1e-6) #1503.04849 five contribution not worth for the moment 1806.11521
BtaunuExp = Mag('BtaunuExp', 1.09e-4, 0.24e-4) #PDG2023 
# BR_Xsll0Exp = Mag('BR_Xsll0Exp', 1.6e-6, 0.46e-6) #PhysRevLett.112.211802
# BXsgExp = Mag('BXsgExp',  3.49, 0.19) #PDG2020
# BXsgTeo =  Mag('BXsgTeo', 3.40, 0.17) # JHEP06(2020)175

#3. K decays 
# Kpnn = Mag('Kpnn', 8.39e-11, 0.3e-11) #1503.02693 (3.10) 
# KpnnExp = Mag('KpnnExp', 1.139e-10, 3.71e-11) #Own Calc. w/ 2103.15389
# KmunuExp = Mag('KmunuExp', 0.6356, 0.0011) # PDG21
# epsilon_Exp = Mag('epsilon_Exp', 2.228e-3, 0.011e-3) #PDG21





############################################################################################
#                                  SM Predictions                                          #
############################################################################################

conv1 = 1./6.5821195e-13  # conversion factor for changing units t
conv2 = 1./(6.5821195e-25)
Abs = syp.Abs
re = syp.re
im = syp.im
# 1. Oscillation Frequencies
SMDMs = conv1*(BBs*GF**2*mBs*mW**2*St0*Abs(Vts)**2*(1-Abs(Ucb)-Abs(Uub))*etaB*fBs**2)/(6*syp.pi**2)
SMDMdDMs = BBdBBs*fBdfBs**2*mBd*Abs(Vtd)**2/(mBs*Abs(Vts)**2)
Betas_teo = syp.arg((Vtb*Vts.conjugate())**2/(Vcb*Vcs.conjugate())**2)
Beta_teo = syp.arg((Vcd*Vcb.conjugate())**2/(Vtd*Vtb.conjugate())**2)
CPobs_teo = SMDMs * asl_SM / DGs_SM
CPobs_exp = DMsExp * asl_Exp/ DGs_Exp

# 2. B decays
# A.  Bu --> tau+ nu
Btaunu = (GF**2*(syp.Abs(Vub))**2*tauBu*fBu**2*mBu*mTau**2*(1-mTau**2/mBu**2)**2)/(8*syp.pi)
# B. B -> Xs gamma
# BXsg = BXsgTeo*syp.Abs((Vts.conjugate()*Vtb)/Vcb)**2/0.9626
# C. Bs --> mu+ mu-
Bsmm = Bsmm_c*(mt/173.1)**3.06*(alpha_s/0.1181)**(-0.18)*(fBs/0.2303)**2*(syp.Abs(Vts)/0.04200)**2*(Vtb/0.982)**2*(tauH/1.615e-12) #FORMULA 2+1+1 (CAMBIAN PESOS fBs)
# D. Bd --> mu+mu-
Bdmm = Bdmm_c * (mt/173.1)**3.06*(alpha_s/0.1181)**(-0.18)*(fBs*fBdfBs/0.1900)**2*(Vtb*syp.Abs(Vtd)/0.0087)**2*tauBd/1.520e-12

R_Bs = Bsmm_exp/Bsmm
R_Bd = Bdmm_exp/Bdmm
# E. B --> Xs l l
# BR_Xsll = (syp.Abs(Vts)*Vtb/(.999146*.04043))*BR_Xsll0

# 3. K decays
# A.  K+ --> pi+ nu nu
# Kpnn_c =Kpnn*(Vcb/40.7e-3)**2.8*(phi/73.2*180/syp.pi)**0.74
# B. K- --> mu+ nu
# BRKmunu = tauK/tauPi*BRpimunu*(Vus/Vud)**2*rfK**2*(mK/mPi)*(1-mMu**2/mK**2)**2/((1-mMu**2/mPi**2)**2) * rKl2

# epsilon k 
#Parametrization based on 0805.3887 (better result w/ 1911.06822)
# Rt = syp.sqrt(syp.re((Vtd*Vtb.conjugate())/(Vcd*Vcb.conjugate()))**2+syp.im((Vtd*Vtb.conjugate())/(Vcd*Vcb.conjugate()))**2)
# beta =syp.atan(syp.im(-Vcd*Vcb.conjugate()/(Vtd*Vtb.conjugate()))/syp.re(-Vcd*Vcb.conjugate()/(Vtd*Vtb.conjugate())))
# const = (Keps*mK0*BK*(mW*FK*GF)**2)/(6*np.sqrt(2)*np.pi**2*DmK0)*Vcb*Vcb.conjugate()*Vus*Vus.conjugate()*Rt*syp.sin(beta)
# sum1 = Vcb*Vcb.conjugate()*Rt*syp.cos(beta)*etatt2*S0_xt((MSmt/mW)**2)
# sum2 = etact2*S0_xt_xc((MSmt/mW)**2, (MSmc/mW)**2)
# sum3 = -etacc2*((MSmc/mW)**2)
# epsilon = const*(sum1+sum2+sum3)

#Parametrization based on 1911.06822 
#epsilon_K To uncomment these line -> Take care of Inami-Lim
# Rt = syp.sqrt(syp.re((Vtd*Vtb.conjugate())/(Vcd*Vcb.conjugate()))**2+syp.im((Vtd*Vtb.conjugate())/(Vcd*Vcb.conjugate()))**2)
# beta =syp.atan(syp.im(-Vcd*Vcb.conjugate()/(Vtd*Vtb.conjugate()))/syp.re(-Vcd*Vcb.conjugate()/(Vtd*Vtb.conjugate())))
# const = (Keps*GF**2*mW**2*mK0*FK**2*BK)/(6*syp.sqrt(2)*syp.pi**2*DmK0)*Vcb*Vcb.conjugate()*Vus*Vus.conjugate()*Rt*syp.sin(beta)
# rhoxc = (MSmc/mW)**2-S0_xt_xc((MSmt/mW)**2, (MSmc/mW)**2)
# rhoxt = S0_xt((MSmt/mW)**2)+(MSmc/mW)**2-2*S0_xt_xc((MSmt/mW)**2, (MSmc/mW)**2)
# corr = Vcb*Vcb.conjugate()*Rt*syp.cos(beta)*etatt*rhoxt-etaut*rhoxc
# epsilon_Teo = const*corr

# Get job done!
mysubs = []
myvars = []
for var in vars().values():
    if isinstance(var, Mag):
        myvars.append(var)
        mysubs.append((var,var.val))
# print epsilon_Teo.subs(mysubs).n(), propagate_error(epsilon_Teo, myvars)

############################################################################################
#                             1. Oscillation Frequencies                                   #
############################################################################################
SMDMs1 = SMDMs.subs(mysubs).n()
eSMDMs1 = propagate_error(SMDMs,myvars)
SMDMdDMs1 = SMDMdDMs.subs(mysubs).n()
eSMDMdDMs1 = propagate_error( SMDMdDMs, myvars)

Betas_teo1 = Betas_teo.subs(mysubs).n()
eBetas_teo = propagate_error(Betas_teo, myvars)
diffBetas = Betas_Exp - Betas_teo
diffBetas_1 = diffBetas.subs(mysubs).n()
ediffBetas = propagate_error(diffBetas, myvars)

Beta_teo1 = Beta_teo.subs(mysubs).n()
eBeta_teo = propagate_error(Beta_teo, myvars)
diffBeta = Beta_Exp - Beta_teo
diffBeta_1 = diffBeta.subs(mysubs).n()
ediffBeta = propagate_error(diffBeta, myvars)

CPobs_teo1 = CPobs_teo.subs(mysubs).n()
eCPobs_teo = propagate_error(CPobs_teo, myvars)
CPobs_exp1 = CPobs_exp.subs(mysubs).n()
eCPobs_exp = propagate_error(CPobs_exp, myvars)

# #ratios & cp constraint
RDMsNew = DMsExp/SMDMs
RDMsDMdNew = (DMsExp/DMdExp)*SMDMdDMs
cpconstraint = CPobs_exp - CPobs_teo


RDMsNew1 = RDMsNew.subs(mysubs).n()
eRDMsNew = propagate_error(RDMsNew, myvars)
RDMsDMdNew1 = RDMsDMdNew.subs(mysubs).n()
eRDMsDMdNew = propagate_error(RDMsDMdNew, myvars)

cpconstraint1 = cpconstraint.subs(mysubs).n()
ecpconstraint = propagate_error(cpconstraint, myvars)

print "Oscillation Frequencies"
print 'SM predictions for DMs and DMd/DMs: ' 
print 'SMDMs = ', re(SMDMs1), ' +- ',  eSMDMs1
print 'DeltaMd/DeltaMs = ', re(SMDMdDMs1), ' +- ', eSMDMdDMs1
print "\n"

print "CP violation phase"
print "SM prediction"
print '-2Beta_s = ', Betas_teo1, ' +- ', eBetas_teo
print 'Diff 2Beta_s = ', diffBetas_1, ' +- ', ediffBetas
print '-2Beta = ', Beta_teo1, ' +- ', eBeta_teo
print 'Diff 2Beta = ', diffBeta_1, ' +- ', ediffBeta

print "CP violation constraints"
print "SM prediction CP obs"
print "CP_SM", re(CPobs_teo1), ' +- ', eCPobs_teo
print "CP_Exp", re(CPobs_exp1), ' +- ', eCPobs_exp
print "\n"

print 'Ratios EXP/SM for DeltaM:' 
print 'RDMsNew', re(RDMsNew1), ' +- ', eRDMsNew
print 'RDMsDMdNew', re(RDMsDMdNew1), ' +-  ', eRDMsDMdNew
print "\n"

print "CP constraint Bs system"
print "Cp constraint", re(cpconstraint1), ' +- ', ecpconstraint



############################################################################################
#                                       2. B decays                                        #
############################################################################################

# A. B --> tau nu
Btaunu1 = Btaunu.subs(mysubs).n()*conv2
eBtaunu = propagate_error(Btaunu, myvars)*conv2

# B. B -> Xs gamma
# BXsg = BXsgTeo*syp.Abs((Vts.conjugate()*Vtb)/Vcb)**2/0.9626
# BXsg1 = BXsg.subs(mysubs).n()
# eBXsg = propagate_error(BXsg, myvars)

# C. B_{s,d} -> mu^+ mu^-
Bsmm1 = Bsmm.subs(mysubs).n()
Bdmm1 = Bdmm.subs(mysubs).n()

R_Bs1 = R_Bs.subs(mysubs).n()
R_Bd1 = R_Bd.subs(mysubs).n()

eBsmm = propagate_error(Bsmm, myvars)
eBdmm = propagate_error(Bdmm, myvars)
eR_Bs = propagate_error(R_Bs, myvars)
eR_Bd = propagate_error(R_Bd, myvars)

# D. B --> Xs ll
# BR_Xsll1 = BR_Xsll.subs(mysubs).n()
# eBR_Xsll = propagate_error(BR_Xsll, myvars)

# Ratios
Bsdmm = Bdmm/Bsmm
Bsdmm1 = Bdmm1/Bsmm1
eBsdmm = propagate_error(Bsdmm, myvars)
R_Bs1 = R_Bs.subs(mysubs).n()
R_Bd1 = R_Bd.subs(mysubs).n()

RBtaunu = BtaunuExp/(Btaunu*conv2)
RBtaunu1 = RBtaunu.subs(mysubs).n()
eRBtaunu = propagate_error(RBtaunu, myvars)

# RBXsg = BXsgExp/BXsg
# RBXsg1 = RBXsg.subs(mysubs).n()
# eRBXsg = propagate_error(RBXsg, myvars)

# RBX = BR_Xsll0Exp/BR_Xsll
# RBX1 = RBX.subs(mysubs).n()
# eRBX = propagate_error(RBX, myvars)

print "B decays"
print "SM predictions for B decays:"
print 'B_taunu = ', re(Btaunu1), ' +- ', eBtaunu
# print 'B_Xsg = ', re(BXsg1), ' +- ', eBXsg
print 'Bsmm = ', re(Bsmm1), ' +- ', eBsmm
print 'Bdmm = ', re(Bdmm1), ' +- ', eBdmm

# print 'BR_Xsll', re(BR_Xsll1), ' +- ', eBR_Xsll
# print "\n"



# Ratio Bs(mumu)/Bd(mumu)
print "Ratios Bs/Bd and EXP/SM for B decays: "
print 'Bsdmm =', re(Bsdmm1), ' +- ', eBsdmm
print 'RBsmm = ', re(R_Bs1), ' +- ', eR_Bs
print 'RBdmm = ', re(R_Bd1), ' +- ', eR_Bd

# Ratios EXP/SM
print 'RBtaunu =', re(RBtaunu1), ' +- ', eRBtaunu
# print 'RBXsg =', re(RBXsg1), ' +- ', eRBXsg
# print 'RBX = ', re(RBX1), ' +- ', eRBX
# print "\n"

############################################################################################
#                                       3. K decays                                        #
############################################################################################

# A. K+ --> pi+ nu nu
# Kpnn_c1 = Kpnn_c.subs(mysubs).n()
# eKpnn = propagate_error(Kpnn_c, myvars)

# B. K- --> mu+ nu
# BRKmunu1 = BRKmunu.subs(mysubs).n()
# eBRKmunu = propagate_error(BRKmunu, myvars)

#ratios
# BRKpnn = KpnnExp/Kpnn_c
# BRKpnn1 = BRKpnn.subs(mysubs).n()
# eBRKpnn = propagate_error(BRKpnn, myvars)

# RKl2  = KmunuExp/BRKmunu
# RKl21 = RKl2.subs(mysubs).n()
# eRKl2 = propagate_error(RKl2, myvars)

# ratio = epsilon_Exp/epsilon_Teo
# ratio1 = ratio.subs(mysubs).n()
# eratio = propagate_error(ratio, myvars)

# print "K decays"
# print "SM predictios for K decays"
# print 'Kpnn_c = ', Kpnn_c1, ' +- ', eKpnn  
# print 'BRKmunu = ', BRKmunu1, ' +- ', eBRKmunu
# print "\n"


# Ratios (EXP/SM)
# print "Ratios (EXP/SM) for K decays"
# print 'BRKpnn =', re(BRKpnn1), ' +- ', eBRKpnn  
# print 'RKl2 = ', re(RKl21), ' +- ', eRKl2
# print "\n"

# Epsilon_K
# print "Ratio epsilon_K (EXP/SM)"
# print 'epsilon_k = ', ratio1, '  +-  ', eratio
# print "\n"

