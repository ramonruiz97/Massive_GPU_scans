from iminuit import * 
from math import sqrt

def chi2(var, mpv, sm, sp, type='junk'):
    if sm==sp: LL = -0.5*((var-mpv)/sm)**2
    elif type == 'junk':
        P = var
        s1 = sp
        s2 = -sm
        m = mpv
        try: LL = -1./2*(1/(2*(s1/m+s2/m)) *((s2/m-s1/m) + sqrt((s2/m-s1/m)**2-8*(s1/m+s2/m)*(1-P/m))))**2
        except ValueError:LL=-1e-9
    else: 
        beta  = sm/sp
        gamma = sm*sp/(sp-sm)
        try: LL = -1./2*(log(1.+((var-mpv)/gamma))/log(beta))**2
        except ValueError: LL= -1e-9
        
    return -2*LL


def na62(BR):
    return chi2(BR, 1.06, 0.35, 0.41)

def e949(BR):
    return chi2(BR, 1.73, 1.05, 1.15)

def world(BR):
    return na62(BR)+e949(BR)


m = Minuit(world)
m.migrad()
m.hesse()
m.minos()
