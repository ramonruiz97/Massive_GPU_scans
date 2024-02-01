# plot_edm
#
#

import numpy as np
import uproot3 as uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2
import seaborn as sns
import pandas as pd
# plt.rc("text", usetex=True)
plt.rc('font', family='serif')  # Set the default font family to serif
plt.rc("text", usetex=True)

def get_dp(dn, dHg):
  return -5.*(dn + dHg/(2.1e-4))

def downsam(x, y, bins=100):
  hist, xedges, yedges = np.histogram2d(x,y, bins=bins)
  xidx, yidx = np.where(hist>0)
  return xedges[xidx], yedges[yidx]

dHg_constr = 7.4e-30
dn_constr = 1.8e-26
dp_constr = get_dp(dn_constr, dHg_constr)
color = sns.color_palette("deep", 4)

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
          }




# scenarios = ["A", "B", "C", "D", "E"]
scenarios = ["E"]
# scenarios = ["A"]
entrystop = -1

for sc in scenarios:
  print("Doing plots for scenario {}".format(sc))

  #Nsigma plotted
  nsigma = 2
  ndof = 12

  # in_pf = "../tuples/Scenario{}_Genetic.root".format(sc)
  _b = ["chi2"]
  _b += ["dn", "dp", "dHg"]
  _b += ["chi2_dn", "chi2_dphid", "Dphid"]
  _b += ["chi2_dHg"]
  _b += ["mg", "msq", "d_edm", "d_cedm"]
  _b_soft = ["chi2_mh", "mh", "m_sq_u1", "m_sq_d1"]
  _b_soft += _b

  #Stat
  signif = chi2.cdf((nsigma)**2, 1)
  chi2_w = chi2.ppf(signif, ndof)
  chi2_wo = chi2.ppf(signif, ndof-1)
  chi2_wo_wo = chi2.ppf(signif, ndof-2)

  chi2_soft = chi2.ppf(signif, ndof+1) #adding mh :)

  in_pf = ["../tuples/Genetic_scenario{}_{}.root".format(sc, i) for i in range(40)]
  in_soft = ["../output_softsusy/Genetic_scenario{}_chi250_{}.root".format(sc, i) for i in range(40)]

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
  logdn_soft = df_soft.eval("log10(abs(dn))").values
  logdp_soft = df_soft.eval("log10(abs(dp))").values
  logdHg_soft = df_soft.eval("log10(abs(dHg))").values
  r_m_soft = df_soft.eval("mg**2/msq**2").values
  r_edm_soft = df_soft.eval("log10(abs(d_edm/d_cedm))").values
  r_edm_soft = np.nan_to_num(r_edm_soft)

  #All chi2s
  df_w = df.query("chi2 < {}".format(chi2_w))
  logdn = df_w.eval("log10(abs(dn))").values
  logdp = df_w.eval("log10(abs(dp))").values
  logdHg = df_w.eval("log10(abs(dHg))").values
  r_m = df_w.eval("mg**2/msq**2").values
  r_edm = df_w.eval("log10(abs(d_edm/d_cedm))").values
  r_edm = np.nan_to_num(r_edm)

  #chi2s wo dn dHg
  df_wodn_wodHg = df.query("chi2 - chi2_dn - chi2_dHg < {}".format(chi2_wo_wo))
  logdn_wodn_wodHg = df_wodn_wodHg.eval("log10(abs(dn))").values
  logdp_wodn_wodHg = df_wodn_wodHg.eval("log10(abs(dp))").values
  logdHg_wodn_wodHg = df_wodn_wodHg.eval("log10(abs(dHg))").values
  rm_wodn_wodHg = df_wodn_wodHg.eval("mg**2/msq**2").values
  r_edm_wodn_wodHg = df_wodn_wodHg.eval("log10(abs(d_edm/d_cedm))").values
  r_edm_wodn_wodHg = np.nan_to_num(r_edm_wodn_wodHg)

  #dn vs dphid
  df.eval("chi2_wodn = chi2 - chi2_dn", inplace=True)
  df.eval("chi2_wophid = chi2 - chi2_dphid", inplace=True)
  df.eval("chi2_wodn_wophid = chi2 - chi2_dphid - chi2_dn", inplace=True)
  
  x = df_w.eval("log10(abs(dn))").values 
  xsoft = df_soft.eval("log10(abs(dn))").values 
  x_wodn = df.query("chi2_wodn < {}".format(chi2_wo)).eval("log10(abs(dn))").values 
  x_wodphid = df.query("chi2_wophid < {}".format(chi2_wo)).eval("log10(abs(dn))").values 
  x_woboth = df.query("chi2_wodn_wophid < {}".format(chi2_wo_wo)).eval("log10(abs(dn))").values

  y = df_w.eval("Dphid").values 
  ysoft = df_soft.eval("Dphid").values 
  y_wodn = df.query("chi2_wodn < {}".format(chi2_wo)).eval("Dphid").values 
  y_wodphid = df.query("chi2_wophid < {}".format(chi2_wo)).eval("Dphid").values 
  y_woboth = df.query("chi2_wodn_wophid < {}".format(chi2_wo_wo)).eval("Dphid").values

  #dn vs dp
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  # plt.text(-2.8, 0.27, r'CL: {} \%'.format(int(signif*100.)), fontdict=font)
  plt.scatter(*downsam(logdn_wodn_wodHg, logdp_wodn_wodHg, bins=3e2), s=3.5,color="black", marker="o", label=r"$\chi^2_{tot.} - chi^2_{dn} - chi^2_{dHg}$")
  plt.scatter(*downsam(logdn, logdp, bins=3e2), color=color[3], s=3.5,marker="o", label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(logdn_soft, logdp_soft, bins=3e2), color=color[2], s=3.5,marker="o", label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.axhspan(np.log10(abs(dp_constr)), -10, alpha=0.2, color="black")
  plt.axvspan(np.log10(abs(dn_constr)), -10, alpha=0.2, color="black")
  plt.xlabel(r'$\log{(|dn|)}$ [e cm]', fontdict=font)
  plt.ylabel(r'$\log{(|dp|)}$ [e cm]', fontdict=font)
  plt.legend(loc='upper left',fontsize=16)
  plt.xlim(-34, -18)
  plt.ylim(-34, -18)
  plt.savefig("Figure3/dn_vs_dp_scenario{}.png".format(sc))

  #dn vs dHg
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  plt.scatter(*downsam(logdn_wodn_wodHg, logdHg_wodn_wodHg, bins=3e2),s=3.5, color="black", marker="o", label=r"$\chi^2_{tot.} - \chi^2_{dn} - \chi^2_{dHg}$")
  plt.scatter(*downsam(logdn, logdHg, bins=3e2), color=color[3], marker="o", s=3.5, label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(logdn_soft, logdHg_soft, bins=3e2), color=color[2], marker="o", s=3.5, label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.axhspan(np.log10(abs(dHg_constr)), -10, alpha=0.2, color="black")
  plt.axvspan(np.log10(abs(dn_constr)), -10, alpha=0.2, color="black")
  plt.xlabel(r'$\log{(|dn|)}$ [e cm]', fontdict=font)
  plt.ylabel(r'$\log{(|dHg|)}$ [e cm]', fontdict=font)
  plt.legend(loc='upper left',fontsize=16)
  plt.xlim(-34, -22)
  plt.ylim(-34, -22)
  plt.savefig("Figure3/dn_vs_dHg_scenario{}.png".format(sc))


  #dn vs dphid
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  # plt.text(-2.8, 0.27, r'CL: {} \%'.format(int(signif*100.)), fontdict=font)
  # plt.axvspan(np.log10(abs(dn_constr)), -10, alpha=0.4, color="blue")
  # plt.axhspan(-0.0013-0.041, -0.0013+0.041, alpha=0.4, color="black")
  plt.scatter(*downsam(x_woboth, y_woboth), color="black", marker="o", s=2.5,label = r"$\chi^2_{tot} - \chi^2_{\phi_d} - \chi^2_{d_n}$")
  plt.scatter(*downsam(x_wodn, y_wodn), color=color[0], marker="o", s=2.5, label = r"$\chi^2_{tot} - \chi^2_{dn}$")
  plt.scatter(*downsam(x_wodphid, y_wodphid), color=color[1], marker="o",s=2.5, label = r"$\chi^2_{tot} - \chi^2_{\phi_d}$")
  plt.scatter(*downsam(x, y), color=color[3], marker="o", s=2.5, label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(xsoft, ysoft), color=color[2], marker="o", s=2.5, label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.xlabel(r'$\log{(|dn|)}$ [e cm]', fontdict=font)
  plt.ylabel(r'$\Delta \phi_d$ [rad]', fontdict=font)
  plt.legend(loc='upper left',fontsize=13, markerscale=4., frameon=True)
  plt.xlim(-34, -22)
  plt.ylim(-5.5, 5.5)
  plt.savefig("Figure3/dn_vs_dphid_scenario{}.png".format(sc))


  #Ratios of EDMs vs Ratios of masses
  plt.close()
  plt.title("Scenario {}".format(sc), fontdict=font)
  # plt.text(-2.8, 0.27, r'CL: {} \%'.format(int(signif*100.)), fontdict=font)
  plt.scatter(*downsam(rm_wodn_wodHg, r_edm_wodn_wodHg, bins=3e2), color="black", marker="o", s=3.5, label=r"$\chi^2_{tot.} - \chi^2_{dn} - \chi^2_{dHg}$")
  plt.scatter(*downsam(r_m, r_edm, bins=3e2), color=color[3], marker="o", s=3.5, label=r"$\chi^2_{tot.}$")
  plt.scatter(*downsam(r_m_soft, r_edm_soft, bins=3e2), color=color[2], marker="o", s=3.5, label=r"$\chi^2_{tot.} + \chi^2_{soft}$")
  plt.xlabel(r'$M_{\tilde{g}}^2/M_{\tilde{q}}^2$', fontdict=font)
  plt.ylabel(r'$\log (|{dd/dd^c}|)$ [e cm]', fontdict=font)
  plt.legend(loc='upper right',fontsize=14)
  plt.ylim(-8, 6)
  plt.savefig("Figure3/redm_vs_rm_scenario{}.png".format(sc))
  
  


