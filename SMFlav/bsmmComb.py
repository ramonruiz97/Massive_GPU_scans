## DMS Oct 2018 -> Update Ramon Ruiz 22
from iminuit import *
from math import isnan, sqrt, log
import numpy as np
import sys
sys.path.append("/scratch28/diego/urania/")
LAT = "211"
fdfs = .252
sfdfs = .012
BRu = 1.010 * 5.961 ## BR(B->JpsiK+)*BR(J/psi --> mumu) in units of 1e-05
sBRu = BRu*(sqrt( (.029/1.010)**2 + (.033/5.961)**2)) #2108.09283

#Update Ramon 211 2022
# if LAT == "211":
#     BRd_SM = 0.10772
#     BRs_SM = 3.6702#3.6288
#     sBRs_SM = 0.1516
#     smfv_val = 0.00202397942566/0.0293493173878771 
# ### 2+1
# elif LAT == "21":
#     BRs_SM =  3.6288
#     BRd_SM = 0.10218
#     sBRs_SM = 0.1868 
#     smfv_val = 0.003247/0.028159

def chi2(var, mpv, sm, sp, type = "junk"):
    if sm == sp: LL = -0.5* (( var - mpv)/sm)**2
    elif type == "junk" :
        P = var
        s1 = sp
        s2 = -sm
        m = mpv
        try: LL =  -1./2*(1/(2*(s1/m+s2/m)) *((s2/m-s1/m) + sqrt((s2/m-s1/m)**2 -8*(s1/m+s2/m)*(1-P/m)  ) ))**2
        except ValueError: LL = -1e9
    else:
        #Ramon implementation of asym likelihood based on: https://indico.cern.ch/event/798971/contributions/3414211/attachments/1907656/3150969/Statistics-2.pdf)
        # Ask Diego, for a reference for his LL
        # num = var - mpv
        # sigma0 = 2*sp*sm/(sp+sm)
        # sigmap = (sp-sm)/(sp+sm)
        # den = sigma0+sigmap*(var - mpv)
        # try: LL =  -1./2* (num/den)**2
        #Diego LL
        beta = sm/sp
        gamma = sm*sp/(sp-sm)
        try: LL = -1./2*( log(1. + ((var-mpv)/gamma))/log(beta)) **2
        except ValueError: LL = -1e9
    return -2*LL
## fdfs = 0.256, .013
        
class Experiment:
    def __init__(self, type = "other"):
        self.unc_type = type
         
    def setBs(self, m,sm,sp):
        self.ms = m
        self.sms = sm
        self.sps = sp
    def setBd(self, m,sm,sp):
        self.md = m
        self.smd = sm
        self.spd = sp
    def setfdfs(self, f = 0.256,sf = 0.013, subs = 1):
        self.f = f
        self.sf = sf
        if subs:
            self.sms = abs(self.ms*sqrt(self.sms**2/self.ms**2 - self.sf**2/self.f**2))
            self.sps = abs(self.ms*sqrt(self.sps**2/self.ms**2 - self.sf**2/self.f**2))

    def updatefdfs(self, f = fdfs, sf = sfdfs, subs = 0):
        sc = f*1./self.f
        self.ms *= sc
        self.sms *= sc
        self.sps *= sc
        self.setfdfs(f,sf, subs)
    def setBRu(self, Bu = BRu, sBu = sBRu, subs = 1):
        self.BRu = Bu
        self.sBRu = sBu
        if subs:
            self.sms = abs(self.ms*sqrt(self.sms**2/self.ms**2 - self.sBRu**2/self.BRu**2))
            self.sps = abs(self.ms*sqrt(self.sps**2/self.ms**2 - self.sBRu**2/self.BRu**2))
            self.smd = abs(self.md*sqrt(self.smd**2/self.md**2 - self.sBRu**2/self.BRu**2))
            self.spd = abs(self.md*sqrt(self.spd**2/self.md**2 - self.sBRu**2/self.BRu**2))
        
    def setRho(self, rho): self.rho =rho
    def chi2s(self, brs): return chi2(brs, self.ms, self.sms, self.sps, self.unc_type)
    def chi2d(self, brd): return chi2(brd, self.md, self.smd, self.spd, self.unc_type)
    def __call__(self, brs,brd, fdfs, bru):
        chi2s = self.chi2s(brs*fdfs/self.f*bru/self.BRu)
        chi2d = self.chi2d(brd*bru/self.BRu)
        rho = self.rho
        chis = sqrt(chi2s)*np.sign(brs-self.ms)
        chid = sqrt(chi2d)*np.sign(brd-self.md)

        out = (chi2s + chi2d - 2*rho*chis*chid)*1./(1-rho*rho)
        if isnan(out): return 1e9
        else: return out

# def getPoints(fname):
#     l0 = file(fname)
#     lines = l0.readlines()
#     out = []
#     for line in lines:
#         out.append(map(float, line.split(",")))
#     return out
# AtlasR2 = { 2.3: getPoints("/scratch15/diego/stuff/ATR21s.txt"), 11.8: getPoints("/scratch15/diego/stuff/ATR23s.txt") }

class GetParametersFromPaint:
    def __init__(self, chi2map = {}):
        self.chi2map = chi2map
    def getPars(self):
        
        def FCN(ms, sms, sps, md, smd, spd, rho):
            shit = Experiment()
            shit.setBs(ms, sms,sps)
            shit.setBd(md, smd, spd)
            shit.setRho(rho)
            shit.setfdfs( subs = 0)
            shit.setBRu( subs = 0)
            chi2 = 0
            for x2 in self.chi2map.keys():
                points = self.chi2map[x2]
                for point in points: chi2  += (shit(point[0],point[1],fdfs, BRu) - x2)**2
            return chi2
        fit = Minuit(FCN, limit_rho = (-1,1), limit_sms = (.1,10),limit_smd = (.1,10),limit_sps = (.1,10), limit_spd = (.1,10), limit_ms = (2,4), limit_md = (-2,2), md = -1.3 )
        fit.migrad()
        fit.hesse()
            
#BREAK
ATLAS = Experiment()
ATLAS.setBs(2.8, 0.7, 0.8)
ATLAS.setBd(-1.9,1.6, 1.6)
ATLAS.setRho(-.55)
ATLAS.setfdfs()
ATLAS.setBRu()
ATLAS.updatefdfs()

#Update Diego 2022
CMS = Experiment()
CMS.setBs(3.83, 0.39, 0.42)
CMS.setBd(0.37, 0.68, 0.75)
CMS.setRho(-0.2)
CMS.setfdfs(.231,.008, subs = 0)
CMS.setBRu()
CMS.updatefdfs()

#Update Ramon 2022 -> Run1 + Run2 simultaneous
LHCb = Experiment()
LHCb.setBs(3.09, .44, .48)
LHCb.setBd(1.2, .75, .84)
LHCb.setRho(-0.11)
LHCb.setfdfs(.249,.008) #fs/fd run1 + run2 averaging w/ luminosities
LHCb.setBRu()
LHCb.updatefdfs()

def BRFCN(brs, brd, f, Bu):
    chi2 = 0.
    chi2 += LHCb(brs,brd, f,Bu)
    chi2 += CMS(brs,brd, f, Bu)
    chi2 += ATLAS(brs,brd, f,Bu)
    chi2 += ((f-fdfs)/sfdfs)**2 + ((Bu-BRu)/sBRu)**2
    return chi2

def RmumuFCN(Rmm, f, Bu, mfv_val, Bs_SM):
    brd = Bs_SM*mfv_val*Rmm*BRd_SM/BRs_SM
    brs = Bs_SM * Rmm
    chi2 = 0.
    chi2 += LHCb(brs,brd, f,Bu)
    chi2 += CMS(brs,brd, f, Bu)
    chi2 += ATLAS(brs,brd, f,Bu)
    chi2 += ((f-fdfs)/sfdfs)**2 + ((Bu-BRu)/sBRu)**2 + (Bs_SM-BRs_SM)**2/(sBRs_SM)**2 + (mfv_val - 1)**2/smfv_val**2
    return chi2


def contFCN(pair): return BRFCN(pair[0],pair[1], fdfs,BRu)
fit = Minuit(BRFCN, limit_brs = (0.1,10), limit_brd = (-1,10))
fit.migrad()
fit.hesse()
fit.minos()
print(fit.values)
print(fit.errors)
print(fit.merrors)
# fit.draw_mnprofile
exit()
#BREAK
# fit2 = Minuit(RmumuFCN)
# fit2.migrad()
# fit2.hesse()
# fit2.minos()
#BREAK
#x = np.arange(0, 7, .1)
#y = np.arange(-2.5, 13, .1)
#X,Y = np.meshgrid(x,y)
def makeZ(zlist):
    Z = X*0
    k = 0
    for i in range (len(x)):
        for j in range( len(y)) :
            Z[j][i] = zlist[k]
            k += 1
    return Z
#pairs = []
#for thing in x:
#    for crap in y: pairs.append([thing,crap])
#chi2list = map(contFCN, pairs)
#Z = makeZ(chi2list)
#Zlist = list(Z)
#v = []
#for shit in Zlist:
#    v.append(min(shit))
#minchi2 = min(v)
#print minchi2
#if minchi2 > 4: print "look at minchi2"
#from matplotlib import pyplot as plt
#cs = plt.contour(X,Y,Z, levels = [minchi2 +  2*1.15, minchi2 + 5.99,minchi2 + 11.8 ]) ## 4 is 1.5 sigma

#plt.clabel(cs, inline=1, fontsize=10)
#plt.xlabel("BRs")
#plt.ylabel("BRd")
#rect = plt.Rectangle((3.65-.23, 1.06-.09), 2*.23, 2*.09, facecolor = "k", edgecolor = "k")
## from matplotlib.collections import PatchCollection
## pc = PatchCollection([rect])
## shit = plt.subplots(1)
#plt.gca().add_artist(rect)
#plt.show()


def RmmProfile():
    from XTuple import XTuple
    fit = Minuit(RmumuFCN )
    fit.migrad()
    minchi2 = fit.get_fmin()['fval']
    tup  = XTuple("Rmm" + LAT, ["Rmm/F","chi2/F"])
    #g = TGraph()
    for i in range(1000):
        x = .1 + .001*i
        fit = Minuit(RmumuFCN, Rmm = x, fix_Rmm = True )
        fit.migrad()
        tup.fillItem("Rmm", x)
        tup.fillItem("chi2", fit.get_fmin()['fval'] - minchi2)
        tup.fill()
    tup.close()

# print("Ey")
fit = Minuit(BRFCN)
fit.migrad()
fit.hesse()
fit.minos()
# print(fit)
# print(dir(fit))


# RmmProfile()
