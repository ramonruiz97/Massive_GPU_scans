from CKMfit import *
import numpy as np
import sympy as syp
from math import pi
replacements = [(Uus, Uus.val), (Uub, Uub.val), (Ucb, Ucb.val), (phi, phi.val)]

CKM = [[Vud.subs(replacements), Vus.subs(replacements), Vub.subs(replacements)], [Vcd.subs(replacements), Vcs.subs(replacements), Vcb.subs(replacements)], [Vtd.subs(replacements), Vts.subs(replacements), Vtb.subs(replacements)]]

Vtd1 = Vtd.subs(replacements)
Vts1 = Vts.subs(replacements)
Vcd1 = Vcd.subs(replacements)
Vcs1 = Vcs.subs(replacements)
Vub1 = Vub.subs(replacements)

aterm = -Vtd*Vtb.conjugate()/(Vud*Vub.conjugate())
bterm = -Vcd*Vcb.conjugate()/(Vtd*Vtb.conjugate())
vars = [Vtd, Vts, Vcd, Vcs, Vub, aterm, bterm]
theta = []
etheta = []
names = ['Vtd', 'Vts', 'Vcd', 'Vcs', 'Vub', 'a','b']
for var in vars:
    Re = syp.re(var)
    Im = syp.im(var)
    theta.append(syp.atan(Im/Re)*180./pi)
    etheta.append(propagate_error(syp.arg(var), [Uub, Uus, Ucb, phi]) *180./pi)

angles = {}
    
for i in xrange(len(names)):
    angles[names[i]] = theta[i].subs(replacements) 
    print "phase", names[i], '=', theta[i].subs(replacements), '+/-', etheta[i]

#Added by Ramon
####PRINT RESULTS
names = ['Vud', 'Vus', 'Vub', 'Vcd', 'Vcs', 'Vcb', 'Vtd', 'Vts', 'Vtb']
vars = [Vud, Vus, Vub, Vcd, Vcs, Vcb, Vtd, Vts, Vtb]
for i, var in enumerate(vars):
     print '#define '+names[i]+'= ('+str(syp.re(var).subs(replacements))+'+'+str(syp.im(var).subs(replacements))+'*I)'
