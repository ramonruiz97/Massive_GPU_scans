import numpy as np
import pycuda.autoinit
import pycuda.cumath
import pycuda.driver as cudriver
import pycuda.gpuarray as gpuarray
import toyrand
from ModelBricks import cuRead
from timeit import default_timer as timer
import importlib
# from scan_ranges import *
import sys,os
sys.path.append("./Scenarios/")
# from scenarioA import *

from RTuple import *

BLOCK_SIZE=256


#Number of deltas
# Ncol_dels = np.int32(18) #(21, 31, 32) * (LL+RR) * (U+D) + 4 RLs = 12 + 4 +2 (LR_33, d, u)= 18
Ncol_dels = np.int32(14) #(12, 13, 23) * (U + D) (RR + LL) + 2 (LR_33, d, u)

#Number of inputs (different from deltas)
# Ncol_reals = np.int32(len(Rmin))
Ncol_reals = np.int32(17)

#Constant input 
#WR not used (?)
ddLL33 = np.float64(1.67/4.04)**2 #WR

#Compiling KERNELs
mod = cuRead("mammoth.c")
MIA = mod.get_function("MIA_observables")
X = cuRead(os.environ["IPAPATH"] + "/cuda/genetic.c", no_extern_c = True)

cx_mutate = X.get_function("cx_mutate")
cx_select = X.get_function("cx_select")
re_mutate = X.get_function("re_mutate")
re_select = X.get_function("re_select")

rand = toyrand.XORWOWRandomNumberGenerator()

#Vars that will be filled for the tuple:
#Inputs:
tupvars =  [
            "tb/F","msq/F","mg/F","mu/F", "signmu/F",
            "MA/F",
            "rGUT1/F","rGUT2/F","Au/F", "Ac/F", "At/F", 
             "rh/F", "BG/F", "msnu/F"
            ]

#Deltas D LL:
tupvars += [
            "r_ddLL12/F", "r_ddLL13/F", "r_ddLL23/F",
            "i_ddLL12/F", "i_ddLL13/F", "i_ddLL23/F"
           ]

#Deltas D RR:
tupvars += [
            "r_ddRR12/F", "r_ddRR13/F", "r_ddRR23/F",
            "i_ddRR12/F", "i_ddRR13/F", "i_ddRR23/F"
           ]

#Deltas U LL : Not really inputs
tupvars += [
            "r_duLL12/F", "r_duLL13/F", "r_duLL23/F",
            "i_duLL12/F", "i_duLL13/F", "i_duLL23/F"
           ]

#Deltas U RR
tupvars += [
             "r_duRR12/F", "r_duRR13/F", "r_duRR23/F",
             "i_duRR12/F", "i_duRR13/F", "i_duRR23/F"
            ]

#Deltas LRs 
#Not really inputs but to be filled -> In scan ranges must be 0
tupvars += [
             "r_ddLR33/F", "r_duLR33/F", 
             "i_ddLR33/F", "i_duLR33/F"
            ]

#For the moment set to 0, not important for the tuple
# tupvars += [
#              "r_ddLR21/F", "r_ddLR23/F", "r_ddLR33/F",
#              "i_ddLR21/F", "i_ddLR23/F", "i_ddLR33/F"
#             ] 

# tupvars += [
#              "r_ddRL23/F", "r_ddRL33/F",
#              "i_ddRL23/F", "i_ddRL33/F" 
#             ] 

#duLR_33 is a function of At -> input to check w/ Amine
# tupvars += [
#              "r_duLR23/F", "r_duLR33/F",
#              "i_duLR23/F", "i_duLR33/F", 
#             ] 

#Observables calculated

#M12
tupvars += [ 
            # "phis/F","RDMs/F","RDMd/F","Dphis/F","Dphid/F", "Asl/F",
            "Dphis/F","RDMs/F","Asl/F",
            "Dphid/F","RDMd/F",
            # "RepsK/F", "RDMK/F", "epek/F","r_M12D/F", "i_M12D/F"
            ]

#P -> ll
tupvars += [
            "Bsmm/F", "Bdmm/F", 
            # "KSmm/F", "KLmm/F", 
            "Rtaunu/F", 
            # "Rl2/F", "RK1K2/F", "IK1K2/F"
             ]

#Wilson Coefficients
tupvars += ["C7_mb_real/F", 
            "C7_mb_imag/F",
            "C7p_mb_real/F", 
            "C8_mb_real/F", 
            "C8p_mb_real/F",
            "ACP/F", 
            "C9_real/F", 
            "C10_real/F",
            "C9_imag/F", 
            "C10_imag/F"
            ]

#EDM s 
tupvars += [
            "d_edm/F", #"edm_s/F", "edm_b/F", 
            "u_edm/F", #"edm_c/F", "edm_t/F"
            "d_cedm/F", #"cedm_s/F", "cedm_b/F", 
            "u_cedm/F", #"cedm_c/F", "cedm_t/F", 
            "dn/F",
            "dp/F", 
            "dHg/F"
           ]

#Masses related observables
tupvars += [
            "mh/F", 
            "mcha1/F","mcha2/F",
            "mneu1/F", "mneu2/F","mneu3/F", "mneu4/F",
            "M1/F", "M2/F",
            "sgAgg/F", 
            "chi2/F", "cost/F"
            ]
#For debugging
tupvars += [
            "chi2_dphis/F", #phis
            "chi2_RDMs/F", #M12_Bs
            "chi2_Asl/F",   #Semileptonic
            "chi2_dphid/F", #phis
            "chi2_RDMd/F", #M12_Bd
            "chi2_Rds/F",  #Ratio
]

tupvars += [
             "chi2_C7/F",   #C7
             "chi2_DACP/F",
             "chi2_RTAUNU/F",
             "chi2_Rsmm/F",  #Bs mm
             "chi2_Rdmm/F",  #Bd mm
             "chi2_dn/F",
             "chi2_dHg/F",   #EDM
]


# tupvars += ["maxdelta/F"]
            # ,
            # "C5g/F","C4g/F","C4H/F",
            # "C4/F","phC5g/F","phC4g/F","phC4H/F","phC4/F"] ## for plotting issues

# tupvars += ["ReC4/F","ImC4/F"] ## for plotting issues too

tupvars += [
           "xdLR/F", "xuRL/F"
        #    "xtRdL/F", "xtLdL/F"
           ]

def fillTup(tup,sig, 
            reals_, deltas_, 
            pll_obs_, m12_obs_, 
            edm_obs_, mass_obs_, 
            wc_obs_, 
            chi2_, 
            chi2_RDMs_, chi2_RDMd_, chi2_Rds_, 
            chi2_dphis_, chi2_dphid_, chi2_Asl_,
            chi2_C7_, chi2_DACP_, 
            chi2_RTAUNU_, 
            chi2_Rsmm_, chi2_Rdmm_, #chi2_mH_, 
            chi2_dHg_, chi2_dn_, #chi2_c9c10_,
            cost_, db=0):
            # maxdelta_, C5g_, C4g_, C4H_,C4_, phC5g_, phC4g_, phC4H_, phC4_, ReC4_, ImC4_, db = 0):
    print "Filling tuple"
    start = timer()
    Nrow = len(reals_)
    for i in range(Nrow):
        
        #Inputs
        tup.fillItem("msq",reals_[i][0])
        tup.fillItem("tb", reals_[i][1])
        tup.fillItem("mg", reals_[i][2])
        tup.fillItem("MA", reals_[i][3])
        tup.fillItem("mu", reals_[i][4])
        tup.fillItem("signmu", sig)
        tup.fillItem("rGUT1", reals_[i][5])
        tup.fillItem("rGUT2", reals_[i][6])

        tup.fillItem("Au", reals_[i][7])
        tup.fillItem("Ac", reals_[i][8])
        tup.fillItem("At", reals_[i][9])
        tup.fillItem("xdLR", reals_[i][10])
        tup.fillItem("xuRL", reals_[i][11])
        # tup.fillItem("xtRdL", reals_[i][12])
        # tup.fillItem("xtLdL", reals_[i][13])
        tup.fillItem("rh", reals_[i][14])
        tup.fillItem("BG", reals_[i][15])
        tup.fillItem("msnu", reals_[i][16])
        
        #Deltas 
        # tup.fillItem("r_ddLL12",np.real(deltas_[i][0]))
        tup.fillItem("r_ddLL13",np.real(deltas_[i][1]))
        tup.fillItem("r_ddLL23",np.real(deltas_[i][2]))
        
        # tup.fillItem("i_ddLL12",np.imag(deltas_[i][0]))
        tup.fillItem("i_ddLL13",np.imag(deltas_[i][1]))
        tup.fillItem("i_ddLL23",np.imag(deltas_[i][2]))

        # tup.fillItem("r_duLL12",np.real(deltas_[i][3]))
        tup.fillItem("r_duLL13",np.real(deltas_[i][4]))
        tup.fillItem("r_duLL23",np.real(deltas_[i][5]))
        
        # tup.fillItem("i_duLL12",np.imag(deltas_[i][3]))
        tup.fillItem("i_duLL13",np.imag(deltas_[i][4]))
        tup.fillItem("i_duLL23",np.imag(deltas_[i][5]))

        # tup.fillItem("r_ddRR12",np.real(deltas_[i][6]))
        tup.fillItem("r_ddRR13",np.real(deltas_[i][7]))
        tup.fillItem("r_ddRR23",np.real(deltas_[i][8]))
        
        # tup.fillItem("i_ddRR12",np.imag(deltas_[i][6]))
        tup.fillItem("i_ddRR13",np.imag(deltas_[i][7]))
        tup.fillItem("i_ddRR23",np.imag(deltas_[i][8]))

        # tup.fillItem("r_duRR12",np.real(deltas_[i][9]))
        tup.fillItem("r_duRR13",np.real(deltas_[i][10]))
        tup.fillItem("r_duRR23",np.real(deltas_[i][11]))
        
        # tup.fillItem("i_duRR12",np.imag(deltas_[i][9]))
        tup.fillItem("i_duRR13",np.imag(deltas_[i][10]))
        tup.fillItem("i_duRR23",np.imag(deltas_[i][11]))
        
        #Not important for the moment set to 0
        # tup.fillItem("r_ddLR21",np.real(deltas_[i][12]))
        # tup.fillItem("r_ddLR23",np.real(deltas_[i][13]))
        # tup.fillItem("r_ddLR33",np.real(deltas_[i][14]))
        
        # tup.fillItem("i_ddLR21",np.imag(deltas_[i][12]))
        # tup.fillItem("i_ddLR23",np.imag(deltas_[i][13]))
        # tup.fillItem("i_ddLR33",np.imag(deltas_[i][14]))

        # tup.fillItem("r_ddRL23",np.real(deltas_[i][15]))
        # tup.fillItem("r_ddRL33",np.real(deltas_[i][16]))

        # tup.fillItem("i_ddRL23",np.imag(deltas_[i][15]))
        # tup.fillItem("i_ddRL33",np.imag(deltas_[i][16]))

        # tup.fillItem("r_duLR23",np.real(deltas_[i][17]))

        tup.fillItem("r_ddLR33",np.real(deltas_[i][12]))
        tup.fillItem("r_duLR33",np.real(deltas_[i][13]))

        # tup.fillItem("i_duLR23",np.imag(deltas_[i][17]))
        tup.fillItem("i_ddLR33",np.imag(deltas_[i][12]))
        tup.fillItem("i_duLR33",np.imag(deltas_[i][13]))

        #Observables

        #M12
        tup.fillItem("Dphis",m12_obs_[i][0])
        tup.fillItem("RDMs",m12_obs_[i][1])
        tup.fillItem("Asl", m12_obs_[i][2])
        tup.fillItem("Dphid",m12_obs_[i][3])
        tup.fillItem("RDMd",m12_obs_[i][4])

        #P -> ll
        tup.fillItem("Bsmm",pll_obs_[i][0])
        tup.fillItem("Bdmm",pll_obs_[i][1])
        tup.fillItem("Rtaunu",pll_obs_[i][2])
        
        #EDMs 
        tup.fillItem("d_edm",1.97e-14*edm_obs_[i][0])
        tup.fillItem("u_edm",1.97e-14*edm_obs_[i][1])
        tup.fillItem("d_cedm",1.97e-14*edm_obs_[i][2])
        tup.fillItem("u_cedm",1.97e-14*edm_obs_[i][3])
        tup.fillItem("dn",edm_obs_[i][4])
        tup.fillItem("dp",edm_obs_[i][5])
        tup.fillItem("dHg",edm_obs_[i][6])


        # WCs
        tup.fillItem("C7_mb_real", np.real(wc_obs_[i][0]))
        tup.fillItem("C7_mb_imag", np.imag(wc_obs_[i][0]))
        tup.fillItem("C8_mb_real", wc_obs_[i][1])
        tup.fillItem("C7p_mb_real",wc_obs_[i][2])
        tup.fillItem("C8p_mb_real",wc_obs_[i][3])
        tup.fillItem("ACP",        wc_obs_[i][4])
        tup.fillItem("C9_real", np.real(wc_obs_[i][5]))
        tup.fillItem("C10_real", np.real(wc_obs_[i][6]))
        tup.fillItem("C9_imag", np.imag(wc_obs_[i][5]))
        tup.fillItem("C10_imag", np.imag(wc_obs_[i][6]))

        
        #Masses
        tup.fillItem("mh",mass_obs_[i][0])
        tup.fillItem("mcha1", mass_obs_[i][1])
        tup.fillItem("mcha2", mass_obs_[i][2])
        tup.fillItem("mneu1", mass_obs_[i][3])
        tup.fillItem("mneu2", mass_obs_[i][4])
        tup.fillItem("mneu3", mass_obs_[i][5])
        tup.fillItem("mneu4", mass_obs_[i][6])
        tup.fillItem("M1", mass_obs_[i][7])
        tup.fillItem("M2", mass_obs_[i][8])
        
        #Statistics
        tup.fillItem("chi2", chi2_[i])
        tup.fillItem("cost", cost_[i])
        #M12 chi2
        tup.fillItem("chi2_RDMs", chi2_RDMs_[i])
        tup.fillItem("chi2_RDMd", chi2_RDMd_[i])
        tup.fillItem("chi2_Rds", chi2_Rds_[i])

        tup.fillItem("chi2_dphis", chi2_dphis_[i])
        tup.fillItem("chi2_dphid", chi2_dphid_[i])
        tup.fillItem("chi2_Asl", chi2_Asl_[i])

        tup.fillItem("chi2_C7", chi2_C7_[i])
        tup.fillItem("chi2_DACP", chi2_DACP_[i])

        tup.fillItem("chi2_RTAUNU", chi2_RTAUNU_[i])

        tup.fillItem("chi2_Rsmm", chi2_Rsmm_[i])
        tup.fillItem("chi2_Rdmm", chi2_Rdmm_[i])

        tup.fillItem("chi2_dHg", chi2_dHg_[i])
        tup.fillItem("chi2_dn", chi2_dn_[i])

        tup.fillItem("cost", cost_[i])

        if db:
            tup.fillItem("sufla_RDMs",db["RDMs"][i])
            tup.fillItem("sufla_RDMd",db["RDMd"][i])
            tup.fillItem("sufla_Psll",db["Psll"][i])
            tup.fillItem("sufla_Pdll",db["Pdll"][i])
        tup.fill()
    print "filled, ", timer() - start

class MammothScanner:
    def __init__(self, Nrow, scenario=None):
        self.Nrow = np.int32(Nrow)
        if scenario != None:
          sm = importlib.import_module(scenario)
        else:
          sm = importlib.import_module("scenarioA")

        Rmin =  np.float64([
                    sm.msqmin, sm.tanbmin, sm.Mgmin, sm.MAmin,  #Main parameters
                    sm.r_mumin, sm.r_M1min,  sm.r_M2min,       #GUT
                    sm.Aumin,   sm.Acmin,   sm.Atmin,        #Trilinear couplings
                    sm.xdLRmin, sm.xuRLmin, sm.xtRdLmin, sm.xtLdLmin,    #Masses
                    1 - sm.rhv,  sm.BG_min,               #Other -> rh related charged higgs mass, BG Ksmm
                    sm.msnumin                         #Leptons
                   ])

        Rmax =  np.float64([
                    sm.msqmax, sm.tanbmax, sm.Mgmax, sm.MAmax,  #Main parameters
                    sm.r_mumax, sm.r_M1max,  sm.r_M2max,       #GUT
                    sm.Aumax,   sm.Acmax,   sm.Atmax,        #Trilinear couplings
                    sm.xdLRmax, sm.xuRLmax, sm.xtRdLmax, sm.xtLdLmax,    #Masses
                    1 - sm.rhv,  sm.BG_max,               #Other -> rh related charged higgs mass, BG Ksmm
                    sm.msnumax                         #Leptons
                   ])


        Ncol_dels = np.int32(14) #(12, 13, 23) * (U + D) (RR + LL) + 2 (LR_33, d, u)

        Ncol_reals = np.int32(len(Rmin))
        crap1 = gpuarray.to_gpu(np.float64(Nrow*[Ncol_reals*[0.]]))
        crap4 = gpuarray.to_gpu(np.float64(Nrow*[Ncol_dels*[0.]]))
        crap5 = gpuarray.to_gpu(np.float64(Nrow*[Ncol_dels*[0.]]))
        
        rand.fill_uniform(crap1)
        rand.fill_uniform(crap4)
        rand.fill_uniform(crap5)
        self.reals0 = gpuarray.to_gpu(np.float64(Nrow*[Rmin]))
        self.dreals = crap1*gpuarray.to_gpu(np.float64(Nrow*[Rmax-Rmin]))
        self.Rmax = gpuarray.to_gpu(Rmax)
        self.Rmin = gpuarray.to_gpu(Rmin)

        self.offdiag_deltas_re = gpuarray.to_gpu(np.float64(Nrow*[[
                                                                   sm.r_dd12LLmin, sm.r_dd13LLmin, sm.r_dd23LLmin,
                                                                   sm.r_du12LLmin, sm.r_du13LLmin, sm.r_du23LLmin,
                                                                   sm.r_dd12RRmin, sm.r_dd13RRmin, sm.r_dd23RRmin,
                                                                   sm.r_du12RRmin, sm.r_du13RRmin, sm.r_du23RRmin,
                                                                   sm.r_dd33LRmin, sm.r_du33LRmin
                                                                #    r_dd21LRmin, r_dd23LRmin,
                                                                #    r_dd23RLmin, 
                                                                #    r_du23LRmin,

                                                                   ]])) 

        self.offdiag_deltas_re += crap4*gpuarray.to_gpu(np.float64(Nrow*[[
            sm.r_dd12LLmax - sm.r_dd12LLmin, sm.r_dd13LLmax - sm.r_dd13LLmin, sm.r_dd23LLmax - sm.r_dd23LLmin,
            sm.r_du12LLmax - sm.r_du12LLmin, sm.r_du13LLmax - sm.r_du13LLmin, sm.r_du23LLmax - sm.r_du23LLmin,
            sm.r_dd12RRmax - sm.r_dd12RRmin, sm.r_dd13RRmax - sm.r_dd13RRmin, sm.r_dd23RRmax - sm.r_dd23RRmin,
            sm.r_du12RRmax - sm.r_du12RRmin, sm.r_du13RRmax - sm.r_du13RRmin, sm.r_du23RRmax - sm.r_du23RRmin,
            sm.r_dd33LRmax - sm.r_dd33LRmin, sm.r_du33LRmax - sm.r_du33LRmin
            # r_dd21LRmax - r_dd21LRmin, r_dd23LRmax - r_dd23LRmin, 
            # r_dd23RLmax - r_dd23RLmin, 
            # r_du23LRmax - r_du23LRmin
                                                                   ]]))

        self.offdiag_deltas_im = gpuarray.to_gpu(np.float64(Nrow*[[
                                                              sm.i_dd12LLmin, sm.i_dd13LLmin, sm.i_dd23LLmin,
                                                              sm.i_du12LLmin, sm.i_du13LLmin, sm.i_du23LLmin,
                                                              sm.i_dd12RRmin, sm.i_dd13RRmin, sm.i_dd23RRmin,
                                                              sm.i_du12RRmin, sm.i_du13RRmin, sm.i_du23RRmin,
                                                              sm.i_dd33LRmin, sm.i_du33LRmin
                                                            #   i_dd21LRmin, i_dd23LRmin, 
                                                            #   i_dd23RLmin,
                                                            #   i_du23LRmin
                                                            ]]))
        
        self.offdiag_deltas_im += crap5*gpuarray.to_gpu(np.float64(Nrow*[[
            sm.i_dd12LLmax - sm.i_dd12LLmin, sm.i_dd13LLmax - sm.i_dd13LLmin, sm.i_dd23LLmax - sm.i_dd23LLmin,
            sm.i_du12LLmax - sm.i_du12LLmin, sm.i_du13LLmax - sm.i_du13LLmin, sm.i_du23LLmax - sm.i_du23LLmin,
            sm.i_dd12RRmax - sm.i_dd12RRmin, sm.i_dd13RRmax - sm.i_dd13RRmin, sm.i_dd23RRmax - sm.i_dd23RRmin,
            sm.i_du12RRmax - sm.i_du12RRmin, sm.i_du13RRmax - sm.i_du13RRmin, sm.i_du23RRmax - sm.i_du23RRmin,
            sm.i_dd33LRmax - sm.i_dd33LRmin, sm.i_du33LRmax - sm.i_du33LRmin
            # i_dd21LRmax - i_dd21LRmin, i_dd23LRmax - i_dd23LRmin, 
            # i_dd23RLmax - i_dd23RLmin, 
            # i_du23LRmax - i_du23LRmin 
            ]]))

        self.pll_obs = gpuarray.to_gpu(np.float64(Nrow*[3*[0.]]))
        self.m12_obs = gpuarray.to_gpu(np.float64(Nrow*[5*[0.]]))
        self.mass_obs = gpuarray.to_gpu(np.float64(Nrow*[9*[0.]]))
        self.wc_obs = gpuarray.to_gpu(np.complex128(Nrow*[7*[0.]]))
        self.edm_obs = gpuarray.to_gpu(np.float64(Nrow*[7*[0.]]))
        
        self.cost = gpuarray.to_gpu(np.float64(Nrow*[0.]))
        self.chi2 = 1.*self.cost

        self.chi2_RDMs = 1.*self.cost
        self.chi2_RDMd = 1.*self.cost
        self.chi2_Rds = 1.*self.cost
        self.chi2_dphis = 1.*self.cost
        self.chi2_dphid = 1.*self.cost
        self.chi2_Asl = 1.*self.cost

        self.chi2_C7 = 1.*self.cost
        self.chi2_DACP = 1.*self.cost
        self.chi2_RTAUNU = 1.*self.cost

        self.chi2_Rsmm = 1.*self.cost
        self.chi2_Rdmm = 1.*self.cost

        self.chi2_dn = 1.*self.cost
        self.chi2_dHg = 1.*self.cost
  

    def __call__(self, cost, reals, deltas, sgn):
        agg_s = np.float64(sgn)
        MIA(self.pll_obs, self.m12_obs,
            self.edm_obs,self.wc_obs, self.mass_obs, 
            self.chi2,
            self.chi2_RDMs, self.chi2_RDMd, self.chi2_Rds, 
            self.chi2_dphis, self.chi2_dphid, self.chi2_Asl,
            self.chi2_C7,  self.chi2_DACP,  
            self.chi2_RTAUNU, self.chi2_Rsmm, self.chi2_Rdmm, 
            self.chi2_dn, self.chi2_dHg, 
            cost, 
            reals, deltas, agg_s, self.Nrow, 
            block = (70,1,1),grid = (int(self.Nrow*1./70)+1,1,1) )
        

    def FlatScan(self, filename):
        tup = RTuple(filename, tupvars)
        Nrow = self.Nrow
        def do(sig):
            reals = self.reals0 + self.dreals
            offdiag_deltas = self.offdiag_deltas_re + 1j*self.offdiag_deltas_im
            self(self.cost, reals, offdiag_deltas, sig)

            reals_ = reals.get()
            deltas_ = offdiag_deltas.get()
            pll_obs_ = self.pll_obs.get()
            m12_obs_ = self.m12_obs.get()
            mass_obs_ = self.mass_obs.get()
            edm_obs_ = (self.edm_obs).get() 
            wc_obs_ = self.wc_obs.get()

            chi2_ = self.chi2.get()
            chi2_RDMs_ = self.chi2_RDMs.get()
            chi2_RDMd_ = self.chi2_RDMd.get()
            chi2_Rds_ = self.chi2_Rds.get()

            chi2_dphis_ = self.chi2_dphis.get()
            chi2_dphid_ = self.chi2_dphid.get()
            chi2_Asl_ = self.chi2_Asl.get()

            chi2_C7_ = self.chi2_C7.get()
            chi2_DACP_ = self.chi2_DACP.get()
            chi2_RTAUNU_ = self.chi2_RTAUNU.get()

            chi2_Rsmm_ = self.chi2_Rsmm.get()
            chi2_Rdmm_ = self.chi2_Rdmm.get()

            chi2_dHg_ = self.chi2_dHg.get() 
            chi2_dn_ = self.chi2_dn.get()
            cost_ = self.cost.get()
            
            fillTup(tup,sig, reals_, deltas_, 
                    pll_obs_, m12_obs_,
                    edm_obs_, mass_obs_,
                    wc_obs_, 
                    chi2_,
                    chi2_RDMs_, chi2_RDMd_, chi2_Rds_, 
                    chi2_dphis_, chi2_dphid_, chi2_Asl_,
                    chi2_C7_, chi2_DACP_, 
                    chi2_RTAUNU_, 
                    chi2_Rsmm_, chi2_Rdmm_, 
                    chi2_dHg_, chi2_dn_, 
                    cost_)
        do(1)
        do(-1)
        tup.close()
        
    def GeneticScan(self,filename, NG = 100, f=0.5, cr = .9 ):
        F = np.float64(f)
        CR = np.float64(cr)
        x_cost = 1.*self.cost
        x_reals = 0*self.reals0
        x_dels = 0j*self.offdiag_deltas_im
        tup = RTuple(filename, tupvars)
        Nrow = self.Nrow
        def do(sig):
            print "Doing sign ", sig
            start = timer()
            reals = self.reals0 + self.dreals
            offdiag_deltas = self.offdiag_deltas_re + 1j*self.offdiag_deltas_im
            self(self.cost, reals, offdiag_deltas, sig)

            for i in range(NG):
                re_mutate(reals, x_reals,self.Rmin, self.Rmax, F, CR, Ncol_reals, self.Nrow, block = (1000,1,1),grid = (int(Nrow*1./1000) + 1,1,1))
                #TODO: Warning truco cx_mutate para deltas
                cx_mutate(offdiag_deltas, x_dels, F, CR, Ncol_dels, Nrow, block = (1000,1,1),grid = (int(Nrow*1./1000) + 1,1,1))

                self(x_cost, x_reals,x_dels,sig)
                darwin = x_cost < self.cost
                re_select(reals,x_reals,darwin, Ncol_reals, Nrow, block = (1000,1,1),grid = (int(Nrow*1./1000) + 1,1,1))
                cx_select(offdiag_deltas,x_dels,darwin, Ncol_dels, Nrow, block = (1000,1,1),grid = (int(Nrow*1./1000) + 1,1,1))

                self(self.cost, reals, offdiag_deltas,sig)
            print "Genetic scan:" , timer()- start
            reals_ = reals.get()
            deltas_ = offdiag_deltas.get()
            pll_obs_ = self.pll_obs.get()
            m12_obs_ = self.m12_obs.get()
            mass_obs_ = self.mass_obs.get()
            edm_obs_ = self.edm_obs.get()
            wc_obs_ = self.wc_obs.get()

            chi2_ = self.chi2.get()
            chi2_RDMs_ = self.chi2_RDMs.get()
            chi2_RDMd_ = self.chi2_RDMd.get()
            chi2_Rds_ = self.chi2_Rds.get()

            chi2_dphis_ = self.chi2_dphis.get()
            chi2_dphid_ = self.chi2_dphid.get()
            chi2_Asl_ = self.chi2_Asl.get()

            chi2_C7_ = self.chi2_C7.get()
            chi2_DACP_ = self.chi2_DACP.get()
            chi2_RTAUNU_ = self.chi2_RTAUNU.get()

            chi2_Rsmm_ = self.chi2_Rsmm.get()
            chi2_Rdmm_ = self.chi2_Rdmm.get()
            chi2_dHg_ = self.chi2_dHg.get() 
            chi2_dn_ = self.chi2_dn.get()
            cost_ = self.cost.get()

            fillTup(tup,sig, 
                    reals_, deltas_, 
                    pll_obs_, m12_obs_,
                    edm_obs_, mass_obs_,
                    wc_obs_, 
                    chi2_,
                    chi2_RDMs_, chi2_RDMd_, chi2_Rds_, 
                    chi2_dphis_, chi2_dphid_, chi2_Asl_,
                    chi2_C7_, chi2_DACP_, 
                    chi2_RTAUNU_, 
                    chi2_Rsmm_, chi2_Rdmm_, 
                    chi2_dHg_, chi2_dn_, 
                    cost_)
            

        do(1)
        do(-1)
        tup.close()


