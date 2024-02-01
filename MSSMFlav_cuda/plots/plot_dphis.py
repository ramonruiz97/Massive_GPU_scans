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
# scenarios = ["D"]
# scenarios = ["A"]
entrystop = -1 
color = sns.color_palette("deep", 4)
# color = sns.color_palette("pastel", 6)


for sc in scenarios:
  print("Doing plots for scenario {}".format(sc))

  #Nsigma plotted
  nsigma = 2
  ndof = 12
  
  in_pf = ["../tuples/Genetic_scenario{}_{}.root".format(sc, i) for i in range(40)]
  in_soft = ["../output_softsusy/Genetic_scenario{}_chi250_{}.root".format(sc, i) for i in range(40)]

  _b = ["chi2"]
  _b += ["chi2_dphis"]
  _b += ["chi2_RDMs"]
  _b += ["chi2_Rsmm"]
  _b += ["chi2_Asl"]
  _b += ["r_ddLL23", "i_ddLL23"]
  _b += ["r_ddRR23", "i_ddRR23"]
  _b += ["Dphis"]
  _b += ["RDMs"]
  _b += ["Bsmm"]
  _b += ["mneu1"]
  _b+= ["tb"]
  _b += ["Asl"]
  _b += ["msq"]
  _b += ["xuRL"]
  _b += ["xdLR"]
   
  #To add chi2 of squarks for the moment done by hand
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

  #Chi2 from valid points of softsusy
  df_soft = pd.concat(df for df in dfs_soft)
  df_soft = df_soft.query("m_sq_u1 > 1e3 & m_sq_d1 > 1e3")
  df_soft = df_soft.query("chi2 + chi2_mh < {}".format(chi2_soft))

  #Test
  y6_test = df["msq"].values
  y7_test = y6_test*np.sqrt(df["xuRL"].values)
  y8_test = y6_test/np.sqrt(df["xdLR"].values)

  _ll = df_soft["r_ddLL23"] + 1j*df_soft["i_ddLL23"].values
  _rr = df_soft["r_ddRR23"] + 1j*df_soft["i_ddRR23"].values
  xsoft_w = np.angle(_ll*_rr)
  ysoft_w = df_soft["Dphis"].values
  y1soft_w = df_soft["RDMs"].values
  y2soft_w = df_soft["Bsmm"].values
  y3soft_w = df_soft["mneu1"].values
  y4soft_w = df_soft["tb"].values
  y5soft_w = df_soft["Asl"].values
  y6soft_w = df_soft["msq"].values
  y7soft_w = y6soft_w*np.sqrt(df_soft["xuRL"].values)
  y8soft_w = y6soft_w/np.sqrt(df_soft["xdLR"].values)

  
  #All chi2s 
  df_w = df.query("chi2 < {}".format(chi2_w))
  _ll = df_w["r_ddLL23"] + 1j*df_w["i_ddLL23"].values
  _rr = df_w["r_ddRR23"] + 1j*df_w["i_ddRR23"].values
  x_w = np.angle(_ll*_rr)
  y_w = df_w["Dphis"].values
  y1_w = df_w["RDMs"].values
  y2_w = df_w["Bsmm"].values
  y3_w = df_w["mneu1"].values
  y4_w = df_w["tb"].values
  y5_w = df_w["Asl"].values
  y6_w = df_w["msq"].values
  y7_w = y6_w*np.sqrt(df_w["xuRL"].values)
  y8_w = y6_w/np.sqrt(df_w["xdLR"].values)

  df_wo = df.query("chi2 - chi2_dphis < {}".format(chi2_wo))
  # print("Shape wo dphis {}".format(df_wo.shape[0]))
  _ll = df_wo["r_ddLL23"] + 1j*df_wo["i_ddLL23"].values
  _rr = df_wo["r_ddRR23"] + 1j*df_wo["i_ddRR23"].values
  x_wo = np.angle(_ll*_rr)
  
  ######################
  ##### phis vs DMs ####
  ######################

  #\Chi2 - \Chi2_phis
  y_wo = df_wo["Dphis"].values
  y1_wo = df_wo["RDMs"].values
  y2_wo = df_wo["Bsmm"].values
  y3_wo = df_wo["mneu1"].values
  y4_wo = df_wo["tb"].values
  y5_wo = df_wo["Asl"].values
  #Chi2 - Chi2_Dms
  x_wodms = df.query("chi2 - chi2_RDMs < {}".format(chi2_wo))["Dphis"]
  y_wodms = df.query("chi2 - chi2_RDMs < {}".format(chi2_wo))["RDMs"]
  #Chi2 - Chi2_Dms - Chi2_phis
  y_wodms_wophis = df.query("chi2 - chi2_RDMs - chi2_dphis < {}".format(chi2_wo_wo))["RDMs"]
  x_wodms_wophis = df.query("chi2 - chi2_RDMs - chi2_dphis < {}".format(chi2_wo_wo))["Dphis"]

  #######################
  ##### phis vs Bsmm ####
  ######################

  #Chi2 - Chi2_Bsmm
  x_wobsmm = df.query("chi2 - chi2_Rsmm < {}".format(chi2_wo))["Dphis"]
  y_wobsmm = df.query("chi2 - chi2_Rsmm < {}".format(chi2_wo))["Bsmm"]
  #Chi2 - Chi2_Bsmm - Chi2_phis
  x_wobsmm_wophis = df.query("chi2 - chi2_Rsmm - chi2_dphis < {}".format(chi2_wo_wo))["Dphis"]
  y_wobsmm_wophis = df.query("chi2 - chi2_Rsmm - chi2_dphis < {}".format(chi2_wo_wo))["Bsmm"]

  #######################
  ##### phis vs Asl ####
  ######################
  x_woasl = df.query("chi2 - chi2_Asl < {}".format(chi2_wo))["Dphis"]
  y_woasl = df.query("chi2 - chi2_Asl < {}".format(chi2_wo))["Asl"]

  x_woasl_wophis = df.query("chi2 - chi2_Asl - chi2_dphis < {}".format(chi2_wo))["Dphis"]
  y_woasl_wophis = df.query("chi2 - chi2_Asl - chi2_dphis < {}".format(chi2_wo))["Asl"]



  #Starting plots:

  #phis vs ddLLRR
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(x_wo, y_wo,bins=6e2), c="black", s=3, marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\phi_s}$")
  plt.scatter(*downsam(x_w, y_w, bins=6e2), c=color[3], s=3,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(xsoft_w, ysoft_w, bins=6e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.text(+np.pi/2 - 1.3, 0.6, r'CL: {} \%'.format(int(signif*100.)), fontdict=font)
  plt.xlabel(r'arg $((\delta^{LL}_d)_{23} (\delta^{RR}_d)_{23})$', fontdict=font)
  plt.ylabel(r'$\Delta \phi_s$ [rad]', fontdict=font)
  plt.axhspan(-0.0011-0.0161, -0.0011+0.0161, alpha=0.25, color="black")
  plt.legend(loc='upper left',fontsize=12, markerscale=2., frameon=True)
  plt.xlim(-np.pi/2, np.pi/2)
  plt.ylim(-0.7, 0.7)
  plt.savefig("Figure1/dphis_vs_argdeltas_scenario{}.png".format(sc))

  #phis vs DMs
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)

  plt.scatter(*downsam(x_wodms_wophis, y_wodms_wophis, bins=5e2), c="black", s=2.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\Delta M_s} - \chi^2_{\phi_s}$")
  plt.scatter(*downsam(x_wodms, y_wodms, bins=3e2), c=color[0], s=2.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\Delta M_s}$")
  plt.scatter(*downsam(y_wo, y1_wo, bins=3e2), c=color[1], s=2.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\phi_s}$")
  plt.scatter(*downsam(y_w, y1_w, bins=3e2), c=color[3], s=2.5,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(ysoft_w, y1soft_w, bins=3e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")

  plt.text(+np.pi/2 - 1., 1.7, r'CL: {} \%'.format(int(signif*100.)), fontdict=font)
  plt.xlabel(r'$\Delta \phi_s$ [rad]', fontdict=font)
  plt.ylabel(r'$\frac{\Delta m_s}{\Delta m^{SM}_s}$', fontdict=font)
  plt.axvspan(-0.0011-0.0161, -0.0011+0.0161, alpha=0.2, color="black")
  plt.axhspan(0.996-0.044, 0.996+0.044, alpha=0.2, color="black")
  plt.legend(loc='upper left',fontsize=12, markerscale=3., frameon=True)
  plt.xlim(-np.pi - 1, np.pi + 1)
  plt.ylim(0., 2.)
  # plt.savefig("Figure1/dphis_vs_DMs_scenario{}.pdf".format(sc), dpi=10)
  plt.savefig("Figure1/dphis_vs_DMs_scenario{}.png".format(sc))


  #phis vs Bsmm
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(x_wobsmm_wophis, y_wobsmm_wophis, bins=5e2), c="black", s=2.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{B_s\mu\mu} - \chi^2_{\phi_s}$")
  plt.scatter(*downsam(x_wobsmm, y_wobsmm, bins=5e2), c=color[0], s=2.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{B_s\mu\mu} $")
  plt.scatter(*downsam(y_wo, y2_wo, bins=3e2), c=color[1], s=2.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\phi_s}$")
  plt.scatter(*downsam(y_w, y2_w, bins=3e2), c=color[3], s=2.5,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(ysoft_w, y2soft_w, bins=3e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.text(+np.pi/2 , 1.75, r'CL: {} \%'.format(int(signif*100.)), fontdict=font)
  plt.xlabel(r'$\Delta \phi_s$ [rad]', fontdict=font)
  plt.ylabel(r'$\frac{BR_{B_s^0 \rightarrow \mu^+\mu^-}}{BR^{SM}_{B_s^0 \rightarrow \mu^+\mu^-}}$', fontdict=font)
  plt.axhspan(0.97-0.1, 0.97+0.1, alpha=0.15, color="black")
  plt.axvspan(-0.0011-0.0161, -0.0011+0.0161, alpha=0.15, color="black")
  plt.legend(loc='upper left',fontsize=12, markerscale=3., frameon=True)
  plt.subplots_adjust(left=0.17)
  plt.xlim(-np.pi - 1, np.pi + 1)
  plt.ylim(0, 2)
  # plt.savefig("Figure1/dphis_vs_Bsmm_scenario{}.pdf".format(sc), dpi=10)
  plt.savefig("Figure1/dphis_vs_Bsmm_scenario{}.png".format(sc))

  #phis vs MLSP
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(y_wo, y3_wo, bins=3e2), c="black", s=2.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\phi_s}$")
  plt.scatter(*downsam(y_w, y3_w, bins=3e2), c=color[3], s=2.5,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(ysoft_w, y3soft_w, bins=3e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.text(+np.pi/2 - 0.5 ,  0.85*max(y3_wo), r'CL: {} \%'.format(int(signif*100.)), fontdict=font)
  plt.xlabel(r'$\Delta \phi_s$ [rad]', fontdict=font)
  plt.ylabel(r'$m_{\chi^0_1}$ [GeV]', fontdict=font)
  plt.axvspan(-0.0011-0.0161, -0.0011+0.0161, alpha=0.25, color="black")
  plt.legend(loc='upper left',fontsize=13, markerscale=6., frameon=True)
  plt.xlim(-np.pi/2 - 1., np.pi/2 + 1)
  # plt.savefig("Figure1/dphis_vs_mlsp_scenario{}.pdf".format(sc))
  plt.savefig("Figure1/dphis_vs_mlsp_scenario{}.png".format(sc))

  #phis vs tanb
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(y_wo, y4_wo, bins=3e2), c="black", s=2.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\phi_s}$")
  plt.scatter(*downsam(y_w, y4_w, bins=3e2), c=color[3], s=2.5,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(ysoft_w, y4soft_w, bins=3e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.text(+np.pi/2 , 0.85* max(y4_wo), r'CL: {} \%'.format(int(signif*100.)), fontdict=font)
  plt.xlabel(r'$\Delta \phi_s$ [rad]', fontdict=font)
  plt.ylabel(r'$\tan \beta$', fontdict=font)
  plt.axvspan(-0.0011-0.0161, -0.0011+0.0161, alpha=0.15, color="black")
  plt.legend(loc='upper left',fontsize=13, markerscale=4., frameon=True)
  plt.xlim(-np.pi/2 -1 , np.pi/2 +1)
  # plt.savefig("Figure1/dphis_vs_tanb_scenario{}.pdf".format(sc))
  plt.savefig("Figure1/dphis_vs_tanb_scenario{}.png".format(sc))

  #phis vs ASL
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  fig = plt.gcf()
  fig.set_size_inches(8, 6)
  #Not really important to plot the two lines below as Asl is not doing anything
  # plt.scatter(*downsam(y_woasl_wophis, x_woasl_wophis,bins=1e3), c="black", s=2.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\phi_s} - \chi^2_{\Delta A_{SL}}$")
  # plt.scatter(*downsam(y_woasl, x_woasl,bins=1e3), c=color[0], s=2.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\Delta A_{SL}}$")
  plt.scatter(*downsam(y5_wo, y_wo,bins=1e3), c="black", s=2.5,marker="o", label=r"$\chi^2_{tot.} - \chi^2_{\phi_s}$")
  plt.scatter(*downsam(y5_w, y_w, bins=1e3), c=color[3], s=2.5,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(y5soft_w, ysoft_w, bins=3e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.text(-np.pi +1., -1., r'CL: {} \%'.format(int(signif*100.)), fontdict=font)
  plt.ylabel(r'$\Delta \phi_s$ [rad]', fontdict=font)
  plt.xlabel(r'$\Delta \left( \frac{a^s_{sl}*\Delta m_s}{\Delta\Gamma_s} \right)$', fontdict=font)
  plt.axhspan(-0.0011-0.0161, -0.0011+0.0161, alpha=0.15, color="black")
  plt.axvspan(-0.13 -0.59, -0.13 + 0.59, alpha=0.15, color="black")
  plt.legend(loc="best",fontsize=13, markerscale=4., frameon=True)
  plt.subplots_adjust(bottom=0.15)
  plt.ylim(-1.5, 1.5)
  plt.xlim(-3,3.)
  # plt.savefig("Figure1/dphis_vs_Als_scenario{}.pdf".format(sc))
  plt.savefig("Figure1/dphis_vs_Als_scenario{}.png".format(sc))

  # mur vs msq
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(y7_test, y6_test, bins=1e2), c="black", s=2.5,marker="o", label=r"$Total$")
  plt.scatter(*downsam(y7_w, y6_w, bins=1e2), c=color[3], s=2.5,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(y7soft_w, y6soft_w, bins=1e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.xlabel(r'$m_{\tilde{u}_R}$ [GeV]', fontdict=font)
  plt.ylabel(r'$m_{\tilde{q}}$ [GeV]', fontdict=font)
  plt.legend(loc='upper left',fontsize=15, markerscale=6., frameon=True)
  # plt.savefig("Figure1/mur_vs_msq_scenario{}.pdf".format(sc))
  plt.savefig("Figure1/mur_vs_msq_scenario{}.png".format(sc))

  # mdr vs msq
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(y8_test, y6_test, bins=3e2), c="black", s=3.,marker="o", label=r"$Total$")
  plt.scatter(*downsam(y8_w, y6_w, bins=3e2), c=color[3], s=3.,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(y8soft_w, y6soft_w, bins=3e2), c=color[2], s=3,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.xlabel(r'$m_{\tilde{d}_R}$ [GeV]', fontdict=font)
  plt.ylabel(r'$m_{\tilde{q}}$ [GeV]', fontdict=font)
  plt.legend(loc='upper left',fontsize=13, markerscale=4., frameon=True)
  # plt.savefig("Figure1/mdr_vs_msq_scenario{}.pdf".format(sc))
  plt.savefig("Figure1/mdr_vs_msq_scenario{}.png".format(sc))


  print(time.time() - start_time)
