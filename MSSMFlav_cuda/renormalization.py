#Renormalization coefficients

import sympy as sp
from CppFunctionsMaker import *
import numpy as np
import matplotlib.pyplot as plt


XX = np.linspace(1, 22,1000)
X = 10**XX
def g2(x, t):

    if t == 1:

        return 5./3*0.1277/(1 - 11./(8*np.pi**2)*0.1277*np.log(x/91.187))
  
    if t == 2:

        return 0.424/(1 - 1./(8*np.pi**2)*0.424*np.log(x/91.187))

    if t == 3:
        return 1.495/(1 + 3./(8*np.pi**2)*1.495*np.log(x/91.187))


plt.figure(1)
plt.grid(True)
plt.semilogx(X, g2(X, 1))
plt.semilogy(X, g2(X, 2))
plt.semilogy(X, g2(X, 3))
#plt.show()


#SUSY scale

def alpha_3(x):

     return 0.1182/(1 + 3./(2*np.pi)*0.1182*np.log(x/91.1876))


#Fuction to renormalize
#x is the coefficient
#y is the number of the coefficient.
#Exalmple, C1_tilde x = C1_tilde y = 1, C5 is x = C5 y = 5
mb = 4.4
mt = 172

alpha_mb = alpha_3(mb)
alpha_mt = alpha_3(mt)
alpha_mu = alpha_3(3)
alpha_muNP = alpha_3(5000)

eta1_K = (alpha_mb/alpha_mu)**(6./25)*(alpha_mt/alpha_mb)**(6./23)*(alpha_muNP/alpha_mt)**(6./21)

eta23_K = np.matrix([[eta1_K**(1./6*(1 - np.sqrt(241))), 0], [0, eta1_K**(1./6*(1 + np.sqrt(241)))]])

eta45_K = np.matrix([[eta1_K**(-4), 0], [0, eta1_K**0.5]])

X23 = np.matrix([[0.5*(-15 - np.sqrt(241)), 0.5*(-15 + np.sqrt(241))], [1, 1]])

X45 = np.matrix([[1, -1], [0, 3]])

R23 = X23*eta23_K*np.linalg.inv(X23)

R45 = X45*eta45_K*np.linalg.inv(X45)


def renorm(x, y):

    mb = 4.4
    mt = 172

    alpha_mb = alpha_3(mb)
    alpha_mt = alpha_3(mt)
    alpha_mu = alpha_3(3)
    alpha_muNP = alpha_3(5000)

    eta1_K = (alpha_mb/alpha_mu)**(6./25)*(alpha_mt/alpha_mb)**(6./23)*(alpha_muNP/alpha_mt)**(6./21)

    eta23_K = np.matrix([[eta1_K**(1./6*(1 - np.sqrt(241))), 0], [0, eta1_K**(1./6*(1 + np.sqrt(241)))]])

    eta45_K = np.matrix([[eta1_K**(-4.), 0], [0, eta1_K**0.5]])

    X23 = np.matrix([[0.5*(-15 - np.sqrt(241)), 0.5*(-15 + np.sqrt(241))], [1, 1]])

    X45 = np.matrix([[1, -1], [0, 3]])

    R23 = X23*eta23_K*np.linalg.inv(X23)

    R45 = X45*eta45_K*np.linalg.inv(X45)


    if y == 1:

        return float(eta1_K*x)

    if y == 2:

        A23 = R23*x
        return float(A23[0])

    if y == 3:

        A23 = R23*x
        return float(A23[1])

    if y == 4:

        A45 = R45*x
        return float(A45[0])

    if y == 5:

        A45 = R45*x
        return float(A45[1])
