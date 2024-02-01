##############################################################################
#                             Flavour Inputs 23                              #
#                                                      author: Ramon Ruiz    #
##############################################################################

from magnitude import Mag

##############################################################################
# Decay Constants and Bag parameters -> Lattice FLAG Nf = 2+1+1 2111.09849   #
##############################################################################

# B meson system
fBs = Mag("fBs", 0.2303, 0.0013, positive= True) #GeV FLAG21
fBsfBd = Mag('fBsfBd', 1.207, 0.0035,     positive = True) #FLAG21, pay attention fB in this paper is not fB0
fBdfBs = 1/fBsfBd
fBu = Mag('fBu',       0.1894,   0.0014,  positive = True) #FLAG2021

#ection used in renormalization (eta, \hat{B})
# etaB = Mag("etaB",    0.55210,   0.00062,  positive = True) #qcd correction factor eta2b, PhysRevD.100.094508
# BBs = Mag("BBs",      1.232,    0.053,    positive = True) #FLAG21
# BBsBBd = Mag('BBsBBd', 1.008, 0.025, positive = True)    #FLAG21 
# BBdBBs = 1/BBsBBd

#Convention used (\hat{eta}, B) -> I only found this for 1 to 5 operators
etaB = Mag("etaB", 0.8393, 0.0034, positive=True) #/scratch15/diego/stuff I havent found anything newer
BBs = Mag("BBs", 0.849, 0.023, positive=True) #Lenz average https://arxiv.org/pdf/1909.11087.pdf
BBd = Mag("BBd", 0.835, 0.028, positive=True) #Lenz average https://arxiv.org/pdf/1909.11087.pdf 
BBsBBd = BBs/BBd
BBdBBs = BBd/BBs

#Inami Lim
St0 = Mag("St0",      2.313,    0.008,   positive = True) #Inami-Lim function xt=4.116(19), PhysRevD.100.094508, checked own calculation InamiLim.py

# K system
rfK = Mag('rfK',        1.1932,  0.0021, positive = True) # FLAG2021 (fk^\pm/fpi^\pm) 
rKl2 = Mag('rKl2',      0.9930,  0.0035, positive = True) # PhysRevLett.93.231803 
FK = Mag('FK',      0.1557,  0.0003, positive = True) #GeV FLAG2021
Keps = Mag('Keps', 0.94, 0.02, positive=True) #1911.06822
BK = Mag('BK', 0.717, 0.024, positive=True) #FLAG21 2+1+1 \hat{B_K}

#############################################################################
#                         Other Constants needed                             #
##############################################################################
GF = Mag("G_F",     1.1663787e-5, 6e-12, positive = True) #GeV^-2   PDG2021
# alpha_s = Mag('alpha_s', 0.1187, 0.0016,  positive = True) # \alpha_s(M_Z) PDG2020
alpha_s = Mag('alpha_s', 0.1179, 0.0016,  positive = True) # \alpha_s(M_Z) PDG2023 This has changed caution in case something is affected

# Mesons
mBs = Mag("mBs",      5.36688,  0.00014, positive = True) #GeV  PDG2021
tauH = Mag('tauH',     1.616e-12,    0.010e-12,   positive = True) #seconds HFLAV2021
mBd = Mag("mBd",      5.27965,  0.00012, positive = True) #GeV  PDG2021
tauBd = Mag('tauBd',    1.519e-12,0.004e-12, positive = True) #seconds PDG2021
mBu = Mag('mBu',       5.27934,  0.00012, positive = True) #GeV   PDG2021 
tauBu = Mag('tauBu',   1.638e-12,0.004e-12, positive = True)  # seconds  PDG2021
mK = Mag('mK',          0.493677, 0.000016, positive = True) #GeV PDG2021
mK0 = Mag('mK0',          0.497611, 0.000013, positive = True) #GeV PDG2021
DmK0 = Mag('DmK0',          3.484e-15, 0.006e-15, positive = True) #GeV PDG2021
tauK = Mag('tauK',      12380e-12, 20e-12, positive = True) #seconds PDG2021
mPi = Mag('mPi',        0.13957039, 0.00000018, positive = True) #GeV PDG2021
tauPi = Mag('tauPi',    26033e-12, 5e-12, positive = True) # seconds PDG2021
BRpimunu = Mag('BRpimunu', 99.98770e-2, 0.00004e-2, positive = True) # PDG2021

# Elementary particles
mt = Mag('mt',         172.76,    0.3,     positive = True) #GeV PDG2021 (PDG 2018, MC input 172.9 \pm 0.4)
mTau = Mag('mTau',     1.77686,  0.00012, positive = True) #GeV PDG2021 
mMu = Mag('mMu',        0.1056583745, 0.0000000024, positive = True) #GeV PDG2021
mW = Mag("mW",        80.379,   0.012,   positive = True) #GeV  PDG2021

#Other:
etatt = Mag("etatt", 0.55, 0.024, positive=True) #1911.06822
etaut = Mag("etaut", 0.402, 0.0068, positive=True) #1911.06822
MSmt = Mag("MSmt", 163.48, 0.86, positive = True) #1911.06822
MSmc = Mag("MSmc", 1.27, 0.02, positive = True) #PDG2021
#For epsilon_K based on 0805.3887 (not used)
# etatt2 = Mag("etatt2", 0.5765, 0.0065, positive=True) #0805.3887
# etact2 = Mag("etact2", 0.47, 0.04, positive=True) 
# etacc2 = Mag("etacc2", 1.43, 0.23, positive=True) 


