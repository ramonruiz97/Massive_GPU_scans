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
# scenarios = ["B"]
# entrystop =  5e5
entrystop = -1 #around 75sec one...
color = sns.color_palette("deep", 4)


for sc in scenarios:
  print("Doing plots for scenario {}".format(sc))

  #Nsigma plotted
  nsigma = 2
  ndof = 12

  in_pf = ["../tuples/Genetic_scenario{}_{}.root".format(sc, i) for i in range(40)]
  in_soft = ["../output_softsusy/Genetic_scenario{}_chi250_{}.root".format(sc, i) for i in range(40)]

  _b = ["chi2"]
  _b += ["chi2_dphid"]
  _b += ["chi2_RDMd"]
  _b += ["chi2_Rdmm"]
  _b += ["r_ddLL13", "i_ddLL13"]
  _b += ["r_ddRR13", "i_ddRR13"]
  _b += ["Dphid"]
  _b += ["RDMd"]
  _b += ["Bdmm"]
  _b += ["mneu1"]
  _b+= ["tb"]

  _b_soft = ["chi2_mh", "mh", "m_sq_u1", "m_sq_d1"]
  _b_soft += _b

  #Stat
  signif = chi2.cdf((nsigma)**2, 1)
  chi2_w = chi2.ppf(signif, ndof)
  chi2_wo = chi2.ppf(signif, ndof-1)
  chi2_wo_wo = chi2.ppf(signif, ndof-2)
  chi2_soft = chi2.ppf(signif, ndof+1) #adding mh :)

  dfs= []
  dfs_soft = []
  for i in range(len(in_pf)):
    dfs.append(uproot.open(in_pf[i])["T"].pandas.df(branches=_b, entrystop=entrystop))
    dfs_soft.append(uproot.open(in_soft[i])["T"].pandas.df(branches=_b_soft, entrystop=entrystop))
  
  df = pd.concat(df for df in dfs)

  df_soft = pd.concat(df for df in dfs_soft)
  df_soft = df_soft.query("m_sq_u1 > 1e3 & m_sq_d1 > 1e3")
  df_soft = df_soft.query("chi2 + chi2_mh < {}".format(chi2_soft))

  # print("Shape all chi2 {}".format(df_w.shape[0]))
  _ll = df_soft["r_ddLL13"] + 1j*df_soft["i_ddLL13"].values
  _rr = df_soft["r_ddRR13"] + 1j*df_soft["i_ddRR13"].values
  xsoft_w = np.angle(_ll*_rr)
  ysoft_w = df_soft["Dphid"].values
  y1soft_w = df_soft["RDMd"].values
  y2soft_w = df_soft["Bdmm"].values
  y3soft_w = df_soft["mneu1"].values
  y4soft_w = df_soft["tb"].values


  df_w = df.query("chi2 < {}".format(chi2_w))
  # print("Shape all chi2 {}".format(df_w.shape[0]))
  _ll = df_w["r_ddLL13"] + 1j*df_w["i_ddLL13"].values
  _rr = df_w["r_ddRR13"] + 1j*df_w["i_ddRR13"].values
  x_w = np.angle(_ll*_rr)
  y_w = df_w["Dphid"].values
  y1_w = df_w["RDMd"].values
  y2_w = df_w["Bdmm"].values
  y3_w = df_w["mneu1"].values
  y4_w = df_w["tb"].values

  df_wo = df.query("chi2 - chi2_dphid < {}".format(chi2_wo))
  # print("Shape wo dphis {}".format(df_wo.shape[0]))
  _ll = df_wo["r_ddLL13"] + 1j*df_wo["i_ddLL13"].values
  _rr = df_wo["r_ddRR13"] + 1j*df_wo["i_ddRR13"].values
  x_wo = np.angle(_ll*_rr)
  
  ######################
  ##### phis vs DMs ####
  ######################

  #\Chi2 - \Chi2_phis
  y_wo = df_wo["Dphid"].values
  y1_wo = df_wo["RDMd"].values
  y2_wo = df_wo["Bdmm"].values
  y3_wo = df_wo["mneu1"].values
  y4_wo = df_wo["tb"].values
  #Chi2 - Chi2_Dms
  x_wodms = df.query("chi2 - chi2_RDMd < {}".format(chi2_wo))["Dphid"]
  y_wodms = df.query("chi2 - chi2_RDMd < {}".format(chi2_wo))["RDMd"]
  #Chi2 - Chi2_Dms - Chi2_phis
  y_wodms_wophis = df.query("chi2 - chi2_RDMd - chi2_dphid < {}".format(chi2_wo_wo))["RDMd"]
  x_wodms_wophis = df.query("chi2 - chi2_RDMd - chi2_dphid < {}".format(chi2_wo_wo))["Dphid"]

  #######################
  ##### phis vs Bsmm ####
  ######################

  #Chi2 - Chi2_Bsmm
  x_wobsmm = df.query("chi2 - chi2_Rdmm < {}".format(chi2_wo))["Dphid"]
  y_wobsmm = df.query("chi2 - chi2_Rdmm < {}".format(chi2_wo))["Bdmm"]
  #Chi2 - Chi2_Bsmm - Chi2_phis
  x_wobsmm_wophis = df.query("chi2 - chi2_Rdmm - chi2_dphid < {}".format(chi2_wo_wo))["Dphid"]
  y_wobsmm_wophis = df.query("chi2 - chi2_Rdmm - chi2_dphid < {}".format(chi2_wo_wo))["Bdmm"]


  #phid vs ddLLRR
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(x_wo, y_wo,bins=3e2), c="black", s=3.5, marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\phi_d}$")
  plt.scatter(*downsam(x_w, y_w, bins=3e2), c=color[3], s=3.5,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(xsoft_w, ysoft_w, bins=3e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.text(+np.pi - 1.2, 0.9, r'CL: {} \%'.format(int(signif*100.)), fontdict=font)
  plt.xlabel(r'arg $((\delta^{LL}_d)_{13} (\delta^{RR}_d)_{13})$', fontdict=font)
  plt.ylabel(r'$\Delta \phi_d$ [rad]', fontdict=font)
  plt.axhspan(-0.0013-0.041, -0.013+0.041, alpha=0.25, color="black")
  plt.legend(loc='upper left',fontsize=16, markerscale=2., frameon=True)
  plt.xlim(-np.pi/2, np.pi/2)
  plt.savefig("Figure2/dphid_vs_argdeltas_scenario{}.png".format(sc))

  #phid vs DMd
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(x_wodms_wophis, y_wodms_wophis, bins=6e3), c="black", s=3.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\Delta M_d} - \chi^2_{\phi_d}$")
  plt.scatter(*downsam(x_wodms, y_wodms, bins=6e2), c=color[0], s=3.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\Delta M_d}$")
  plt.scatter(*downsam(y_wo, y1_wo, bins=6e2), c=color[1], s=3.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\phi_d}$")
  plt.scatter(*downsam(y_w, y1_w, bins=6e2), c=color[3], s=3.5,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(ysoft_w, y1soft_w, bins=6e2), c=color[2], s=3.5,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.text(+np.pi/2 - 1., 1.7, r'CL: {} \%'.format(int(signif*100.)), fontdict=font)
  plt.xlabel(r'$\Delta \phi_d$ [rad]', fontdict=font)
  plt.ylabel(r'$\frac{\Delta m_d}{\Delta m^{SM}_d}$', fontdict=font)
  plt.axvspan(-0.0013-0.041, -0.013+0.041, alpha=0.25, color="black")
  plt.axhspan(0.934-0.086, 0.934+0.086, alpha=0.2, color="black")
  plt.legend(loc='upper left',fontsize=14, markerscale=6., frameon=True)
  plt.xlim(-np.pi/2, np.pi/2)
  plt.ylim(0., 2.5)
  plt.savefig("Figure2/dphid_vs_DMd_scenario{}.png".format(sc))

  
  #phid vs Bdmm
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(x_wobsmm_wophis, y_wobsmm_wophis, bins=5e2), c="black", s=3,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{B_d\mu\mu} - \chi^2_{\phi_d}$")
  plt.scatter(*downsam(x_wobsmm, y_wobsmm, bins=5e2), c=color[0], s=3,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{B_d\mu\mu} $")
  plt.scatter(*downsam(y_wo, y2_wo, bins=5e2), c=color[1], s=3,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\phi_d}$")
  plt.scatter(*downsam(y_w, y2_w, bins=5e2), c=color[3], s=3,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(ysoft_w, y2soft_w, bins=5e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.text(+np.pi/2 - 1., 8.75, r'CL: {} \%'.format(int(signif*100.)), fontdict=font)
  plt.xlabel(r'$\Delta \phi_d$ [rad]', fontdict=font)
  plt.ylabel(r'$B_d^0 \rightarrow \mu^+\mu^-$', fontdict=font)
  plt.axhspan(3.9-4.4, 3.9+4.4, alpha=0.15, color="black")
  plt.axvspan(-0.0013-0.041, -0.013+0.041, alpha=0.25, color="black")
  plt.legend(loc='upper left',fontsize=14, markerscale=6., frameon=True)
  plt.xlim(-np.pi/2 - 1., np.pi/2 + 1)
  plt.ylim(-1, 10)
  plt.savefig("Figure2/dphid_vs_Bdmm_scenario{}.png".format(sc), dpi=10)

  #phid vs MLSP
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(y_wo, y3_wo, bins=3e2), c="black", s=2.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\phi_d}$")
  plt.scatter(*downsam(y_w, y3_w, bins=3e2), c=color[3], s=2.5,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(ysoft_w, y3soft_w, bins=3e2), c=color[2], s=2.5,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.text(+np.pi -1 ,  0.85*max(y3_wo), r'CL: {} \%'.format(int(signif*100.)), fontdict=font)
  plt.xlabel(r'$\Delta \phi_d$ [rad]', fontdict=font)
  plt.ylabel(r'$m_{\chi^0_1}$ [GeV]', fontdict=font)
  plt.axvspan(-0.0013-0.041, -0.013+0.041, alpha=0.25, color="black")
  plt.legend(loc='upper left',fontsize=15, markerscale=4., frameon=True)
  plt.xlim(-np.pi-1 , np.pi+1)
  plt.savefig("Figure2/dphid_vs_mlsp_scenario{}.png".format(sc))

  #phid vs tanb
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(y_wo, y4_wo, bins=3e2), c="black", s=2.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\phi_d}$")
  plt.scatter(*downsam(y_w, y4_w, bins=3e2), c=color[3], s=2.5,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(ysoft_w, y4soft_w, bins=3e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.text(+np.pi -1 , 0.85* max(y4_wo), r'CL: {} \%'.format(int(signif*100.)), fontdict=font)
  plt.xlabel(r'$\Delta \phi_d$ [rad]', fontdict=font)
  plt.ylabel(r'$\tan \beta$', fontdict=font)
  plt.axvspan(-0.0013-0.041, -0.013+0.041, alpha=0.25, color="black")
  plt.legend(loc='upper left',fontsize=15, markerscale=4., frameon=True)
  plt.xlim(-np.pi-1  , np.pi+1)
  plt.savefig("Figure2/dphid_vs_tanb_scenario{}.png".format(sc))


  print(time.time() - start_time)
