#Fit CKM Bjorken parametrization
import numpy as np
import sympy as syp
from iminuit import Minuit
import cPickle
import CKMin as inps ## metin aqui os inputs, sinon toleaba con tantas definicios de Vij
from magnitude import Mag, propagate_error
I = syp.I
#The ones we know

def FCN(Uus, Uub, Ucb, phi): ## Comment: phi non fai falla, diria eu
    Vus1 = np.sqrt(Uus)
    Vcb1 = np.sqrt(Ucb)
    Vud1 = np.sqrt(1 - Uus - Uub)
    #aterm = -Vtd*Vtb.conjugate()/(Vud*Vub.conjugate())
    Vtb = syp.sqrt(1-Ucb-Uub)
    Vcs = (1./(1-Uub))*(-syp.sqrt(Uus*Ucb*Uub)*syp.cos(phi) + syp.sqrt(1-Uus-Ucb + Uus*Ucb - 2*Uub + Uus*Uub + Ucb*Uub + Uub**2 - Uus*Ucb*Uub*(syp.sin(phi))**2)) ## penso q  formula q tinhas tu taba mal, faltaba un parentesis en (1-Uub)
    Vud = syp.sqrt(1-Uus-Uub)
    Vus = syp.sqrt(Uus)
    Vcb = syp.sqrt(Ucb)

                                                                                                                                              ## Complex
    Vub = syp.sqrt(Uub)*syp.exp(-I*phi)

    Vcd = -(Vcs*Vus.conjugate() + Vcb*Vub.conjugate())/Vud
    Vts = -(Vus*Vub.conjugate() + Vcs*Vcb.conjugate())/Vtb
    Vtd = -(Vts*Vus.conjugate() + Vtb*Vub.conjugate())/Vud       
    aterm = -Vtd*Vtb.conjugate()/(Vud*Vub.conjugate())
    bterm = -Vcd*Vcb.conjugate()/(Vtd*Vtb.conjugate())
    alpha = syp.arg(aterm)
    beta = syp.arg(bterm)
    chi2 = ((Vud1-inps.Vud)/inps.eVud)**2 + ((Vcb1-inps.Vcb)/inps.eVcb)**2 + ((np.sqrt(Uub)-inps.Vub)/inps.eVub)**2 + ((Vus1-inps.Vus)/inps.eVus)**2 + ((phi - inps.gamma)/inps.egamma)**2  #+ ((alpha-1.508)/.077)**2 #+ ((beta-0.386)/.012)**2#+ ((alpha-1.52)/.033)**2

    return chi2

m = Minuit(FCN, pedantic=False, limit_Uus=(0,1), limit_Uub=(0,1), limit_Ucb=(0,1), limit_phi=(0,3))

m.migrad()
m.hesse()
m.minos()

Uub = Mag("Uub", m.values["Uub"], m.errors["Uub"], positive = True)
Uus = Mag("Uus", m.values["Uus"], m.errors["Uus"], positive = True)
Ucb = Mag("Ucb", m.values["Ucb"], m.errors["Ucb"], positive = True)
phi = Mag("phi", m.values["phi"], m.errors["phi"], positive = True)


Vtb = syp.sqrt(1-Ucb-Uub)
Vcs = (1./(1-Uub))*(-syp.sqrt(Uus*Ucb*Uub)*syp.cos(phi) + syp.sqrt(1-Uus-Ucb + Uus*Ucb - 2*Uub + Uus*Uub + Ucb*Uub + Uub**2 - Uus*Ucb*Uub*(syp.sin(phi))**2)) ## penso q  formula q tinhas tu taba mal, faltaba un parentesis en (1-Uub)
Vud = syp.sqrt(1-Uus-Uub)
Vus = syp.sqrt(Uus)
Vcb = syp.sqrt(Ucb)

## Complex
Vub = syp.sqrt(Uub)*syp.exp(-I*phi)

Vcd = -(Vcs*Vus.conjugate() + Vcb*Vub.conjugate())/Vud
Vts = -(Vus*Vub.conjugate() + Vcs*Vcb.conjugate())/Vtb
Vtd = -(Vts*Vus.conjugate() + Vtb*Vub.conjugate())/Vud
VARS = vars()
def printCKM():
    cousas = ["Vud","Vus","Vub"] + ["Vcd","Vcs","Vcb"] + ["Vtd","Vts","Vtb"]

    print '============================================='
    print '              FIT RESULTS'
    print '============================================='
    mysubs = [(Ucb, Ucb.val), (Uus,Uus.val) , (Uub,Uub.val), (phi,phi.val) ]
    for cousa in cousas:
        V = syp.Abs(VARS[cousa])
        err = propagate_error(V, [Uub,Uus,Ucb,phi])
        print "|", cousa , "|",  " =  " , syp.re(V.subs( mysubs ).n()) ,"+/-", err

