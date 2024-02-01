# plot_edm
#
#

import numpy as np
import uproot3 as uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2
from matplotlib.colors import LogNorm
import time
import seaborn as sns
import pandas as pd


start_time = time.time()

def downsam(x, y, bins=100):
  hist, xedges, yedges = np.histogram2d(x,y, bins=bins)
  xidx, yidx = np.where(hist>0)
  return xedges[xidx], yedges[yidx]


# plt.rc("text", usetex=True)
plt.rc('font', family='serif')  # Set the default font family to serif
plt.rc("text", usetex=True)

font = {'family': 'serif',
          'color':  'black',
          'weight': 'normal',
          'size': 18,
          }




scenarios = ["A", "B", "C", "D", "E"]
# scenarios = ["A"]
# entrystop =  5e5
entrystop = -1 #around 75sec one...
color = sns.color_palette("deep", 4)


for sc in scenarios:
  print("Doing plots for scenario {}".format(sc))

  #Nsigma plotted
  nsigma = 2
  ndof = 12
  mW = 80.379

  in_pf = ["../tuples/Genetic_scenario{}_{}.root".format(sc, i) for i in range(40)]
  in_soft = ["../output_softsusy/Genetic_scenario{}_chi250_{}.root".format(sc, i) for i in range(40)]

  
  #Stat branches
  _b = ["chi2"]
  _b += ["chi2_C7"]
  _b += ["chi2_DACP"]
  _b += ["chi2_Rsmm"]
  _b += ["chi2_RTAUNU"]
  #Complex constraints
  _b += ["chi2_dphis", "chi2_dphid", "chi2_dn", "chi2_dHg"]
  #Real constraints
  _b += ["chi2_RDMd"]

  #C7-C8 related
  _b += ["C7_mb_real", "C7_mb_imag"]
  _b += ["C9_real", "C9_imag"]
  _b += ["Rtaunu"]
  _b += ["ACP"]
  _b += ["mg", "mu", "signmu", "msq", "tb"]

  _b += ["MA"]
  
  #To add chi2 of squarks for the moment done by hand
  _b_soft = ["chi2_mh", "mh", "m_sq_u1", "m_sq_d1"]
  _b_soft += _b

  #Stat
  signif = chi2.cdf((nsigma)**2, 1)
  chi2_w = chi2.ppf(signif, ndof)
  chi2_wo = chi2.ppf(signif, ndof-1)
  chi2_test = chi2.ppf(signif, 1)
  chi2_wo_wo = chi2.ppf(signif, ndof-2)
  chi2_soft = chi2.ppf(signif, ndof+1) #adding mh :)

  #Reading tuples
  dfs= []
  dfs_soft = []
  for i in range(len(in_pf)):
    dfs.append(uproot.open(in_pf[i])["T"].pandas.df(branches=_b, entrystop=entrystop))
    dfs_soft.append(uproot.open(in_soft[i])["T"].pandas.df(branches=_b_soft, entrystop=entrystop))
  
  df = pd.concat(df for df in dfs)
  df = df.query("Rtaunu < 1.7 & Rtaunu>0.8")

  #Chi2 from valid points of softsusy
  df_soft = pd.concat(df for df in dfs_soft)
  df_soft = df_soft.query("m_sq_u1 > 1e3 & m_sq_d1 > 1e3")
  df_soft = df_soft.query("chi2 + chi2_mh < {}".format(chi2_soft))
  mg_soft = df_soft ["mg"].values
  msq_soft = df_soft ["msq"].values
  tanb_soft = df_soft ["tb"].values
  ACP_soft = df_soft ["ACP"].values
  C7_soft = np.abs(df_soft["C7_mb_real"].values + 1j*df_soft["C7_mb_imag"])
  C9_soft = np.abs(df_soft["C9_real"].values + 1j*df_soft["C9_imag"])
  mHpm_soft = np.sqrt(df_soft["MA"].values**2 + mW*mW)
  Rtaunu_soft = df_soft ["Rtaunu"].values

  y_total = df["Rtaunu"].values
  x_total = df["tb"].values
  C7_total = np.abs(df["C7_mb_real"].values + 1j*df["C7_mb_imag"])
  C9_total = np.abs(df["C9_real"].values + 1j*df["C9_imag"])
  #I have 110 values with Nans
  C9_total = np.nan_to_num(C9_total)
  mg_total = df["mg"].values
  mHpm_total = np.sqrt(df["MA"].values**2 + mW*mW**2)

  df_test = df.query("chi2_RDMd < {}".format(chi2_test))
  y_test = df_test["Rtaunu"].values
  mHpm_test = np.sqrt(df_test["MA"].values**2 + mW*mW)
  x_test = df_test["tb"].values

  df_w = df.query("chi2 < {}".format(chi2_w))
  mg = df_w ["mg"].values
  msq = df_w ["msq"].values
  tanb = df_w ["tb"].values
  ACP = df_w ["ACP"].values
  C7 = np.abs(df_w["C7_mb_real"].values + 1j*df_w["C7_mb_imag"])
  C9 = np.abs(df_w["C9_real"].values + 1j*df_w["C9_imag"])
  mHpm = np.sqrt(df_w["MA"].values**2 + mW*mW)
  Rtaunu = df_w ["Rtaunu"].values

  df_woACP = df.query("chi2 - chi2_DACP < {}".format(chi2_wo))
  C7_woACP = np.abs(df_woACP["C7_mb_real"].values + 1j*df_woACP["C7_mb_imag"])
  ACP_woACP = df_woACP ["ACP"].values
  
  #Chi2 C7
  df_woC7 = df.query("chi2 - chi2_C7 < {}".format(chi2_wo))
  C7_woC7 = np.abs(df_woC7["C7_mb_real"].values + 1j*df_woC7["C7_mb_imag"])
  mg_woC7 = df_woC7 ["mg"].values

  #C7 vs Mgluino
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  # plt.scatter(*downsam(mg_total, C7_total), c="black", s=3,marker="o", label=r"Total")
  plt.scatter(*downsam(mg, C7, bins=3e2), c=color[3], s=3,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(mg_soft, C7_soft, bins=3e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.xlabel(r'$M_{\tilde{g}}$ [GeV]', fontdict=font)
  plt.ylabel(r'$ | \Delta C_7 (\mu_b) |$', fontdict=font)
  plt.axhspan(0., 0.+0.02, alpha=0.25, color="black")
  plt.legend(loc='upper left',fontsize=14, markerscale=4., frameon=True)
  # plt.ylim(0, 1.)
  plt.savefig("Figure4/mg_vs_C7_scenario{}.png".format(sc))


  #C7 vs ACP
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(ACP_woACP, C7_woACP), c="black", s=3,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{ACP}$")
  plt.scatter(*downsam(ACP, C7), c=color[3], s=3,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(ACP_soft, C7_soft), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.xlabel(r'$A_{CP}$', fontdict=font)
  plt.ylabel(r'$ | \Delta C_7 (\mu_b) |$', fontdict=font)
  plt.axhspan(0., 0.+0.02, alpha=0.25, color="black")
  plt.axvspan(0.05 - 0.04, 0.05 + 0.04, alpha=0.25, color="black")
  plt.legend(loc='upper right',fontsize=14, markerscale=2., frameon=True)
  plt.savefig("Figure4/Acp_vs_C7_scenario{}.png".format(sc))
  #

  #C7 vs C9
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  # plt.scatter(*downsam(C7_total, C9_total), c=color[1], s=3,marker="o", label=r"Total")
  plt.scatter(*downsam(C7, C9, bins=3e2), c=color[3], s=3,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(C7_soft, C9_soft, bins=3e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.xlabel(r'$ | \Delta C_7 (\mu_b) |$', fontdict=font)
  plt.ylabel(r'$ | \Delta C_9 (\mu_b) |$', fontdict=font)
  plt.axvspan(0., 0.+0.02, alpha=0.25, color="black")
  plt.legend(loc='upper right',fontsize=14, markerscale=2., frameon=True)
  plt.savefig("Figure4/C7_vs_C9_scenario{}.png".format(sc))
  

  #test -> Very interesting good DMd implies no big positive shifts in B->taunu
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(x_total, y_total, bins=3e2), c="black", s=3,marker="o", label=r"Total")
  plt.scatter(*downsam(x_test, y_test, bins=3e2), c=color[0], s=3,marker="o", label=r"$\chi^2_{DM_d}$")
  plt.scatter(*downsam(tanb, Rtaunu, bins=3e2), c=color[3], s=3,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(tanb_soft, Rtaunu_soft, bins=3e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.xlabel(r'$ \tan \beta$', fontdict=font)
  plt.ylabel(r'$ R_{B^+ \tau \nu}$', fontdict=font)
  plt.axhspan(1.26 - 0.3, 1.26 + 0.3, alpha=0.25, color="black")
  # plt.ylim(0.8, 1.8)
  plt.legend(loc='upper left',fontsize=16, markerscale=2., frameon=True)
  plt.savefig("Figure4/tanb_vs_taunu_scenario{}.png".format(sc))
 
  #R_btaunu vs mH+-
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(mHpm_total, y_total, bins=3e2), c="black", s=3,marker="o", label=r"Total")
  plt.scatter(*downsam(mHpm_test,y_test, bins=3e2), c=color[0], s=3,marker="o", label=r"$\chi^2_{DM_d}$")
  plt.scatter(*downsam(mHpm, Rtaunu, bins=3e2), c=color[3], s=3,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(mHpm_soft, Rtaunu_soft, bins=3e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.xlabel(r'$m_{H^{\pm}} [GeV]$', fontdict=font)
  plt.ylabel(r'$ R_{B^+ \tau \nu}$', fontdict=font)
  plt.axhspan(1.26 - 0.3, 1.26 + 0.3, alpha=0.25, color="black")
  plt.ylim(0.8, 1.8)
  plt.legend(loc='upper left',fontsize=16, markerscale=2., frameon=True)
  plt.savefig("Figure4/mHpm_vs_taunu_scenario{}.png".format(sc))
  print(time.time() - start_time)
