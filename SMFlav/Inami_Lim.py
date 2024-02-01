# Inami_Lim_function Computation
# author = Ramon Angel Ruiz Fernandez
#This reference is taken here just in case: Based on: https://inspirehep.net/files/686af1f5d681f50da1573aaf131bf50a
#Really based on 9806471.pdf

import numpy as np
import sympy as syp
from magnitude import Mag, propagate_error


def S0_xt(x):
  term1 = (4*x-11*x**2+x**3)/(4*(1-x)**2)
  term2 = (-3*x**3*syp.log(x))/(2*(1-x)**3) 
  result = term1+term2
  if isinstance(x, Mag):
    result = Mag("S0_xt", result.subs(x, x.val).n(), syp.re(propagate_error(result, [x])))  
  return result

def S0_xt_xc(x1, x2):
  term1 = syp.log(x1/x2)
  term2 = -3*x1/(4*(1-x1))
  term3 = -3*x1**2*syp.log(x1)/(4*(1-x1)**2)
  result = x2*(term1+term2+term3) 
  if isinstance(x1, Mag) and isinstance(x2, Mag):
      result = Mag("S0_xt_xc", result.subs([(x1, x1.val),(x2, x2.val)]).n(), syp.re(propagate_error(result, [x1, x2])))
  return result

  
  







