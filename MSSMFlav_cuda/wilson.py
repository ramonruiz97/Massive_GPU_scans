#Wilson coefficients

import sympy as sp
from CppFunctionsMaker import *

x = sp.Symbol('x')
y = sp.Symbol('y')

alpha_s, ddLL_21, ddLL_23, ddLL_31, g1_1, g1_2, g1_3, ddRR_21, ddRR_23, ddRR_31, g4_1, g4_2, g4_3, M_dL, M3, g5_1, g5_2, g5_3 = sp.symbols('alpha_s ddLL_21 ddLL_23 ddLLL_31 g1_1 g1_2 g1_3 ddRR_21 ddRR_23 ddRR_31 g4_1 g4_2 g4_3 M_dL M3 g5_1 g5_2 g5_3')


#Gluino box contribution

C1_1 = -(alpha_s)**2/(M_dL**2)*(ddLL_21**2*g1_1 + ddLL_21*ddLL_23*ddLL_31*g1_2 + (ddLL_23*ddLL_31)**2*g1_3)

C2_1 = 0.

C3_1 = 0.

C4_1 = - alpha_s**2/M3**2*(ddLL_21*ddRR_21*g4_1 + 1./2*(ddLL_21*ddRR_23*ddRR_31 + ddLL_23*ddLL_31*ddRR_21)*g4_2 + ddLL_23*ddLL_31*ddRR_23*ddRR_31*g4_3)
C5_1 = - alpha_s**2/M3**2*(ddLL_21*ddRR_21*g5_1 + 1./2*(ddLL_21*ddRR_23*ddRR_31 + ddLL_23*ddLL_31*ddRR_21)*g5_2 + ddLL_23*ddLL_31*ddRR_23*ddRR_31*g5_3)

C1_1_tilde = - alpha_s**2/M_dL**2*(ddRR_21**2*g1_1 + ddRR_21*ddRR_23*ddRR_31*g1_2 + (ddRR_23*ddRR_31)**2*g1_3)

C2_1_tilde = 0.

C3_1_tilde = 0.


#Wino and Higgsino contributions

alpha_2, ggw_1, ggw_2, ggw_3, M2, gw_1, gw_2, gw_3, M_tR, Mw = sp.symbols('alpha_2 ggw_1 ggw_2 ggw_3 M2 gw_1 gw_2 gw_3 M_tR Mw')


C1_2 = - alpha_s*alpha_2/(6*M_dL**2)*(ddLL_21**2*ggw_1 + ddLL_21*ddLL_23*ddLL_31*ggw_2 + (ddLL_23*ddLL_31)**2*ggw_3)

C2_2 = 0.

C3_2 = 0.

C4_2 = 0.

C5_2 = 0.

C1_2_tilde = 0.

C2_2_tilde = 0.

V_td, V_ts, m_s, tanB, m_t, mu, A_t, eps_g, M_tL, f_3, m_b, H, M_A, y_t, eps_y, M_dR = sp.symbols('V_td V_ts m_s tanB m_t mu A_t eps_g M_tL f_3 m_b H M_A y_t eps_y M_dR')

C3_2_tilde = - alpha_2**2/8*(V_ts*V_td)**2*m_s**2*tanB**2*m_t**4*mu**2*A_t**2*f_3/((1 + eps_g*tanB)**2*Mw**4*M_tL**4*M_tR**4)

#Neutral Higgses exchange contribution

C1_3 = 0.

C2_3 = 0.

C3_3 = 0.

C4_3 = - 8*alpha_s**2*alpha_2*m_b**2*tanB**4*mu**2*M3**2/(9*sp.pi*Mw**2*(1+eps_g*tanB)**2*(1 + (eps_g + eps_y*y_t**2)*tanB)**2*M_A**2*M_dL**2*M_dR**2)*ddLL_23*ddLL_31*ddRR_23*ddRR_31*H*H

C5_3 = 0.

C1_3_tilde = 0.

C2_3_tilde = 0.

C3_3_tilde = 0.

#Summing all the contributions

C1 = C1_1 + C1_2 + C1_3

C2 = C2_1 + C2_2 + C2_3

C3 = C3_1 + C3_2 + C3_3

C4 = C4_1 + C4_2 + C4_3

C5 = C5_1 + C5_2 + C5_3

C1_tilde = C1_1_tilde + C1_2_tilde + C1_3_tilde

C2_tilde = C2_1_tilde + C2_2_tilde + C2_3_tilde

C3_tilde = C3_1_tilde + C3_2_tilde + C3_3_tilde


#Converting the expressions to C++

## C1 = FunctionPrinter(C1_2, (x,y), "C1", "double") 
## C1.make()
## C2 = FunctionPrinter(C2, (x,y), "C2", "double") 
## C2.make()
## C3 = FunctionPrinter(C3, (x,y), "C3", "double") 
## C3.make()
## C4 = FunctionPrinter(C4_3, (x,y), "C4", "double") 
## C4.make()
## C5 = FunctionPrinter(C5_1, (x,y), "C5", "double") 
## C5.make()
## C1_tilde = FunctionPrinter(C1_tilde, (x,y), "C1_tilde", "double") 
## C1_tilde.make()
## C2_tilde = FunctionPrinter(C2_tilde, (x,y), "C2_tilde", "double") 
## C2_tilde.make()
## C3_tilde = FunctionPrinter(C3_tilde, (x,y), "C3_tilde", "double") 
## C3_tilde.make()


f3 = - x**2*(x*(1+x+y) - 3*y)*sp.log(x)/((x-y)**3*(1-x)**3) - y**2*(y*(1+x+y) - 3*x)*sp.log(y)/((y-x)**3*(1-y)**3) - 2*(x**2 + y**2 - x*y - x**2*y - x*y**2 + x**2*y**2)/((1-x)**2*(1-y)**2*(x-y)**2)

## A1 = sp.limit(f3, x, 1)

## A2 = sp.limit(f3, y, 1)

## A3 = sp.limit(f3, y, x)

## A4 = sp.limit(A3, x, 1)

## F3 = FunctionPrinter(f3, (x,y), "f3", "double")
## F3.make()
## a1 = FunctionPrinter(A1, [y], "a1", "double")
## a1.make()
## a2 = FunctionPrinter(A2, [x], "a2", "double")
## a2.make()
## a3 = FunctionPrinter(A3, [x], "a3", "double")
## a3.make()

mZ = sp.Symbol('mZ')

alpha = 0.1182/(1 + 3./(2*sp.pi)*0.1182*sp.log(x/mZ))


import matplotlib.pyplot as plt
import numpy as np

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


gw_1 = (-5 - 67*x - 13*x**2 + x**3)/(12*(1-x)**4) - x*(3+4*x)*sp.log(x)/(1-x)**5

B1 = sp.limit(gw_1, x, 1)

gw_1  = FunctionPrinter(gw_1, [x], "gw_1", 'double')
gw_1.make()

gw_2 = (15 + 309*x + 113*x**2 - 19*x**3 + 2*x**4)/(24*(1-x)**5) + x*(12 + 23*x)*sp.log(x)/(2*(1-x)**6)

B2 = sp.limit(gw_2, x, 1)

Gw_2 = FunctionPrinter(gw_2, [x], "gw_2", 'double')
Gw_2.make()

gw_3 = - (10 + 284*x + 159*x**2 - 41*x**3 + 9*x**4 - x**5)/(40*(1-x)**6) - 3*x*(2 + 5*x)*sp.log(x)/(2*(1-x)**7)

B3 = sp.limit(gw_3, x, 1)

Gw_3 = FunctionPrinter(gw_3, [x], "gw_3", 'double')
Gw_3.make()



def alpha_3(x):

     return 0.1182/(1 + 3./(2*np.pi)*0.1182*np.log(x/91.1876))

mb = 164
mt = 172

alpha_mb = alpha_3(mb)
alpha_mt = alpha_3(mt)
alpha_mu = alpha_3(3)
alpha_muNP = alpha_3(5000)

eta1_K = (alpha_mb/alpha_mu)**(6./25)*(alpha_mt/alpha_mb)**(6./23)*(alpha_muNP/alpha_mt)**(6.21)

eta23_K = np.matrix([[eta1_K**(1./6*(1 - np.sqrt(241))), 0], [0, eta1_K**(1./6*(1 + np.sqrt(241)))]])

eta45_K = np.matrix([[eta1_K**(-4), 0], [0, eta1_K**0.5]])

X23 = np.matrix([[0.5*(-15 - np.sqrt(241)), 0.5*(-15 + np.sqrt(241))], [1, 1]])

X45 = np.matrix([[1, -1], [0, 3]])




