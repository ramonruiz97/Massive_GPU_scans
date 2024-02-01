# run_softsusy
#
#

__all__ = []
__author__ = ["Ramon Angel Ruiz Fernandez"]
__email__ = ["rruizfer@cern.ch"]

import uproot3 as uproot
import numpy as np
import subprocess
import os
import time
import re
import pandas as pd
import warnings
import multiprocessing
warnings.simplefilter(action='ignore', category=FutureWarning)



def get_string(
		            Q, M1, M2, M3,       #Classics
	              mu, mA, tanb,        #Higgs
		            M2Q, M2U, M2D,          #Squarks
		            TU,                  #Trilinear
	       ):
    # Define the input slha string
    slha_string =  ["Block MODSEL		         # Select model"]
    slha_string += ["1    0		             # General MSSM"]
    slha_string += ["6    1		             # FV          "]

    slha_string += ["Block SMINPUTS		     # Standard Model inputs "]
    slha_string += ["1   1.279400000e+02	     # alpha^(-1) SM MSbar(MZ) "]
    slha_string += ["2   1.166370000e-05	     # G_Fermi "]
    slha_string += ["3   1.185000000e-01	     # alpha_s(MZ) SM MSbar "]
    slha_string += ["4   91.00000e+00	       # MZ(pole) "]
    slha_string += ["5   4.180000000e+00	     # mb(mb) SM MSbar "]
    slha_string += ["6   172.69e+00	           # mtop(pole) "]
    slha_string += ["7   1.77686e+00	     # mtau(pole) "]

    slha_string += [f"Block VCKMin                 #CP violation SM "]
    slha_string += [f"1  0.224388098372811     #lambda  "]
    slha_string += [f"2  0.8313999273445006    #A  "]
    slha_string += [f"3  0.1325470288207307    #rho_bar "]
    slha_string += [f"4  0.3503302822256662    #eta_bar  "]
  
  
    slha_string += ["Block MINPAR "]
    slha_string += ["4  {}         #Mu sign ".format(np.sign(mu))]
  
  
    slha_string += ["Block EXTPAR                     # non-universal SUSY breaking parameters "]
    slha_string += ["0   {}                    # Input scale ".format(Q)]
    slha_string += ["1   {}                   # M_1(Q) ".format(M1)]
    slha_string += ["2   {}                   # M_2(Q) ".format(M2)]
    slha_string += ["3   {}                   # M_3(Q) ".format(M3)]
    
	  #To check if wo or with sign 
    slha_string += ["23  {}                   # mu(Q) ".format(np.abs(mu))]
    slha_string += ["25  {}                 # tanb(Q) ".format(tanb)]
    slha_string += ["26  {}                   # mA(Q) ".format(mA)]
	  

    slha_string += ["Block TUIN #Trilinear coupling "]
    slha_string += ["1  1  0.0                   #TU_11(MX)  "]
    slha_string += ["1  2  0.0                   #TU_12(MX)  "]
    slha_string += ["1  3  0.0                   #TU_13(MX)  "]
    slha_string += ["2  1  0.0                   #TU_21(MX)  "]
    slha_string += ["2  2  0.0                   #TU_22(MX)  "]
    slha_string += ["2  3  0.0                   #TU_23(MX)  "]
    slha_string += ["3  1  0.0                   #TU_31(MX)  "]
    slha_string += ["3  2  0.0                   #TU_32(MX)  "]
    slha_string += ["3  3  {}                    #TU_33(MX)  ".format(TU[2,2])]

    slha_string += ["Block TDIN #Trilinear coupling "]
    slha_string += ["1  1  0.0                   #TD_11(MX)  "]
    slha_string += ["1  2  0.0                   #TD_12(MX)  "]
    slha_string += ["1  3  0.0                   #TD_13(MX)  "]
    slha_string += ["2  1  0.0                   #TD_21(MX)  "]
    slha_string += ["2  2  0.0                   #TD_22(MX)  "]
    slha_string += ["2  3  0.0                   #TD_23(MX)  "]
    slha_string += ["3  1  0.0                   #TD_31(MX)  "]
    slha_string += ["3  2  0.0                   #TD_32(MX)  "]
    slha_string += ["3  3  0.0                   #TD_33(MX)  "]

    slha_string += ["Block TEIN #Trilinear coupling "]
    slha_string += ["1  1  0.0                   #TE_11(MX)  "]
    slha_string += ["1  2  0.0                   #TE_12(MX)  "]
    slha_string += ["1  3  0.0                   #TE_13(MX)  "]
    slha_string += ["2  1  0.0                   #TE_11(MX)  "]
    slha_string += ["2  2  0.0                   #TE_12(MX)  "]
    slha_string += ["2  3  0.0                   #TE_13(MX)  "]
    slha_string += ["3  1  0.0                   #TE_11(MX)  "]
    slha_string += ["3  2  0.0                   #TE_12(MX)  "]
    slha_string += ["3  3  0.0                   #TE_13(MX)  "]
  
    slha_string += ["Block MSQ2IN "]
    slha_string += ["1  1 {}    #M2Q (MX) ".format(M2Q[0,0])]
    slha_string += ["2  2 {}    #M2Q (MX) ".format(M2Q[1,1])]
    slha_string += ["3  3 {}    #M2Q (MX) ".format(M2Q[2,2])]
    
    slha_string += ["1  2  0.0          #M2Q_12(MX) Real "]
    slha_string += ["1  3  {}   #M2Q_13(MX) Real ".format(M2Q[0,2])]
    slha_string += ["2  3  {}   #M2Q_23(MX) Real ".format(M2Q[1,2])]
  
    slha_string += ["Block MSU2IN "]
    slha_string += ["1  1 {}    #M2U (MX) ".format(M2U[0,0])]
    slha_string += ["2  2 {}    #M2U (MX) ".format(M2U[1,1])]
    slha_string += ["3  3 {}    #M2U (MX) ".format(M2U[2,2])]
    
    slha_string += ["1  2  0.0          #M2U_12(MX) Real "]
    slha_string += ["1  3  {}   #M2U_13(MX) Real ".format(M2U[0,2])]
    slha_string += ["2  3  {}   #M2U_23(MX) Real ".format(M2U[1,2])]

    slha_string += ["Block MSD2IN "]
    slha_string += ["1  1 {}    #M2D (MX) ".format(M2D[0,0])]
    slha_string += ["2  2 {}    #M2D (MX) ".format(M2D[1,1])]
    slha_string += ["3  3 {}    #M2D (MX) ".format(M2D[2,2])]
    
    slha_string += ["1  2  0.0          #M2D_12(MX) Real "]
    slha_string += ["1  3  {}   #M2D_13(MX) Real ".format(M2D[0,2])]
    slha_string += ["2  3  {}   #M2D_23(MX) Real ".format(M2D[1,2])]
    
	  #To -check slepton shit
    slha_string += ["Block MSL2IN "]
    slha_string += ["1  1  1.0e8    #M2U (MX) "]
    slha_string += ["2  2  1.0e8    #M2U (MX) "]
    slha_string += ["3  3  1.0e8    #M2U (MX) "]

    slha_string += ["Block MSE2IN "]
    slha_string += ["1  1  1.0e8    #M2U (MX) "]
    slha_string += ["2  2  1.0e8    #M2U (MX) "]
    slha_string += ["3  3  1.0e8    #M2U (MX) "]
  
  
    
    return slha_string

#First try to do a pipeline for N points and 1 job
def launch_singleth(job_id):
  file = f"tuples/Genetic_scenarioA_{job_id}.root"
  op = "output_softsusy"
  os.makedirs(op, exist_ok=True)
  outfile = f"{op}/Genetic_scenarioA_{job_id}.root"
  start_time = time.time()
  #entrystop for testing
  #entrystop=-1
  entrystop = 1000
  df = uproot.open(file)["T"].pandas.df(entrystop=entrystop)
	#Here maybe add chi2 cut (?)
  of = pd.DataFrame(columns = df.columns)

  mt = 162.5
  mh_constr = 125.25
  smh_constr = np.sqrt(2**2 + 0.17**2)
  script = "/scratch47/ramon.ruiz/Spectrum-generators/softsusy-4.1.9/softpoint.x "

  N = df.shape[0]
  #Calculation of all the arrays
  #First things from df
  Q = df.eval("sqrt(mg*msq)").values
  M1 = df["M1"].values
  M2 = df["M2"].values
  M3 = df["mg"].values
  Mu = df.eval("mu*signmu").values
  MA = df["MA"].values
  Tanb = df["tb"].values
  
  #Second calculations
  mdL2 = df["msq"].values*df["msq"].values
  mur2 = mdL2*df["xuRL"].values
  mdr2 = mdL2*df["xdLR"].values
  
  ones = np.ones_like(Q)
  
  #First check module #soft does not recognize phases
  ddll_13 = np.abs(df["r_ddLL13"] + 1j*df["i_ddLL13"])
  ddll_23 = np.abs(df["r_ddLL13"] + 1j*df["i_ddLL23"])
  
  ddll_31 = np.conjugate(ddll_13)
  ddll_32 = np.conjugate(ddll_23)
  ddll_12 = np.zeros_like(ddll_13)
  ddll_21 = ddll_12
  
  ddll_11 = np.ones_like(ddll_12)
  ddll_22 = np.ones_like(ddll_12)
  ddll_33 = np.ones_like(ddll_12)
  
  zeros = np.zeros_like(ones)
  
  md_ll =  np.array([
  									 [ones, zeros, ddll_13],
  									 [zeros, ones, ddll_23],
  									 [ddll_31, ddll_32, ones]
  								 ]).transpose()
  
  
  msq = mdL2[:, np.newaxis, np.newaxis] *md_ll
  
  
  ddrr_13 = np.abs(df["r_ddRR13"] + 1j*df["i_ddRR13"])
  ddrr_23 = np.abs(df["r_ddRR23"] + 1j*df["i_ddRR23"])
  
  ddrr_31 = np.conjugate(ddll_13)
  ddrr_32 = np.conjugate(ddll_23)
  
  md_rr = np.array([
  								 [ones, zeros, ddrr_13],
  								 [zeros, ones, ddrr_23],
  								 [ddrr_31, ddrr_32, ones]
  								]).transpose()
  
  msd = mdr2[:, np.newaxis, np.newaxis] *md_rr
  
  
  durr_13 = np.abs(df["r_duRR13"] + 1j*df["i_duRR13"])
  durr_31 = np.conjugate(durr_13)
  
  mu_rr = np.array([
  								 [ones, zeros, durr_13],
  								 [zeros, ones, zeros],
  								 [durr_31, zeros, ones]
  								]).transpose()
  
  msu = mur2[:, np.newaxis, np.newaxis] *mu_rr
  
  v = 246
  
  v1 = v/np.sqrt(1+df["tb"]*df["tb"])
  v2 = v1*df["tb"]
  
  Tu_33 = - np.sqrt(2)*mt*df["At"]/v2
  
  TU =  np.array([                      
        				[zeros, zeros, zeros],  
        				[zeros, zeros, zeros],  
        				[zeros, zeros, Tu_33] 
        				]).transpose()
  
  s = 0
  h = 0
  # job = 0
  m_h, chi2_mh = [], []
  m_su1, m_sd1 = [], []
  chi2_squark = []
  for i in range(N):
    # if i%500 == 0:
    #   print("Job {} point {}".format(job, i))
    slha = get_string(Q[i], M1[i], M2[i], M3[i],
  										Mu[i], MA[i], Tanb[i],
                      msq[i, :, :], msu[i,:,:], msd[i,:,:],
                      TU[i,:,:])
  
    #To create here the folder based on the job_id
  	#1. Write input for softsusy
    ip = "slha/input_job{}".format(job_id)
    input = "{}/slha_{}".format(ip, i)
    os.makedirs(ip, exist_ok=True)
    with open(input, "w") as f:
  	  f.write("\n".join(slha))
  	  f.close()
   
    op = "slha/output_job{}".format(job_id)
    output = "{}/slha_{}".format(op, i)
    os.makedirs(op, exist_ok=True)
    command = "{} leshouches < {} > {}".format(script, input, output)
  
    result = subprocess.run(command, stdout=subprocess.PIPE,  stderr=subprocess.PIPE, shell=True, text=True)
  
    with open(output, "r") as out:
  	  content = out.read()
  
    error_message = "Declining to write spectrum"
    problem = False
    if error_message in content:
      problem = True
    if error_message in content:
      problem = True
  
    error_message = "problem"
    if error_message in content:
      problem = True
    if error_message in content:
      problem = True
    
  
    if problem:
      # print("Point {} is not valid".format(i))
      s = s + 1
      os.system("rm {}".format(output))
  
    else:
      p_mh = r"\b25\s+(\S+)"
      m_mh = re.search(p_mh, content)
      m_h.append(float(m_mh.group(1)))
      chi2_mh.append( ( float(m_mh.group(1)) - mh_constr)**2/smh_constr**2)
      p_dsq = r"\b1000001\s+(\S+)"
      m_dsq = re.search(p_dsq, content)
      p_usq = r"\b1000001\s+(\S+)"
      m_usq = re.search(p_dsq, content)
      mdsq = float(m_dsq.group(1))
      musq = float(m_usq.group(1))
      m_sd1.append(mdsq)
      m_su1.append(musq)
  
      if (mdsq > 1.0e3) and (musq > 1.0e3):
        chi2_squark.append(0.)
      else:
        chi2_squark.append(1e9)
      
      of = of.append(df.loc[i], ignore_index=True)
  
  of["mh"] = m_h
  of["m_sq_u1"] = m_su1
  of["m_sq_d1"] = m_sd1
  of["chi2_mh"] = chi2_mh
  
  with uproot.recreate(outfile) as f:
    _branches = {}
    for k, v in of.items():
      if 'int' in v.dtype.name:
        _v = np.int32
      elif 'bool' in v.dtype.name:
        _v = np.int32
      else:
        _v = np.float64
      _branches[k] = _v
    mylist = list(dict.fromkeys(_branches.values()))
    f["T"] = uproot.newtree(_branches)
    f["T"].extend(of.to_dict(orient='list'))

  # print(f"Final errors {s}")
  # print(f"Aciertos {N-s}")
  # print(f"Porcentaje {(N-s)/N}")
  # print(f"time of N events: {time.time()-start_time}")
  # print(f"Porcentaje higgs {(h)/N}")

if __name__ == "__main__":
  #To-do an arg parser
  nprocess = 2
  processes = []
  for j in range(nprocess):
    p = multiprocessing.Process(target=launch_singleth, args=(j,))
    processes.append(p)
    p.start()

  # Wait for all processes to finish
  for p in processes:
    p.join()

  print("All processes completed.")


# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
