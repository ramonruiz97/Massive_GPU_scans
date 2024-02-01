#Magnitude class
from sympy import Symbol, re
import math

def propagate_error(f, mags):
    mysubs = []
    for mag in mags: mysubs.append( (mag, mag.val) )
    s = 0
    for x in mags: s += (x.err*f.diff(x).subs(mysubs).n())**2
    

    try: out = math.sqrt(s)
    except TypeError :
        #print "CANCER", f, "gives complex values: Uncertainty squared:", s
        out = math.sqrt(re(s))
    return out

class _Mag:
    def __init__(self, val,err):
        self.val = val
        self.err = err
class Mag(Symbol):

    def __init__(self, name,  val = 0., err = 0., latexname = "", **assumptions):
        self.first_name = name
        self.texname = name
        self.first_name = name
        if latexname :    self.texname = latexname
        self.val = val
        self.err = err
        Symbol.__init__(self, name, **assumptions)
    def __new__(cls, name, val = 0. , err = 0., latexname = "",  **assumptions):
        cls.first_name = name
        cls.texname = name
        cls.first_name = name
        if latexname :    cls.texname = latexname
        cls.val = val
        cls.err = err
        return Symbol.__new__(cls, name, **assumptions)
        
        
    def useLatexName(self): self.name = self.texname
    def useFirstName(self): self.name = self.first_name
    def setTex(self, texname): self.texname = texname
    def info(self): print self.name , " = ", self.val , " +/- ", self.err
    
    def setVal(self, num):
        'Magnitude'
        self.val = num

    def setError(self, error):
        'Error'
        self.err = error

    #Asymmetric error
    def setPos(self, EPos):
        'Positive error'
        self.errP = EPos
        
    def setNeg(self, ENeg):
        'Negative error'
        self.errM = ENeg
    def set(self, val, err = 0, EP = 0, EM =0 ):
        self.setVal(val)
        if err:
            self.setError(err)
            self.setPos(err)
            self.setNeg(err)
            
        if EP: self.setPos(EP)
        if EM: self.setNeg(EM)
    
