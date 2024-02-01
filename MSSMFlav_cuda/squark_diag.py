# squark_diag
#
#

__author__ = ["Ramon Angel Ruiz Fernandez"]
__email__ = ["rruizfer@cern.ch"]


import numpy as np
import uproot3 as uproot
import argparse
import time 

def build3t3(df, dou, quir):
  string = dou + quir
  rc_ll = ["r_"+string+"12", "r_"+string+"13", "r_"+string+"23"]
  ic_ll = ["i_"+string+"12", "i_"+string+"13", "i_"+string+"23"]
  ca_ll = np.vectorize(complex)(df[rc_ll].values, df[ic_ll].values)
  ca_ll = ca_ll.reshape(len(df), 3)
  conj_ll = np.conj(ca_ll)
  diag = np.ones((len(ca_ll), 3), dtype=complex)

  matrix = np.array([
                  [diag[:, 0],    ca_ll[:, 0],     ca_ll[:, 1]],
                  [conj_ll[:, 0], diag[:, 1],      ca_ll[:, 2]],
                  [conj_ll[:, 1], conj_ll[:, 2],   diag[:, 2]],
                ]).transpose()


  return matrix

def build3t3_lr(df, dou, quir):
  string = dou + quir
  rc_ll = ["r_"+string+"33"]
  ic_ll = ["i_"+string+"33"]
  ca_ll = np.vectorize(complex)(df[rc_ll].values, df[ic_ll].values)
  ca_ll = ca_ll.reshape(len(df), 1)
  zeros = np.zeros((len(df), 1))
  conj_ll = np.conj(ca_ll)

  m_lr = np.array([
                  [zeros[:, 0],    zeros[:, 0],     zeros[:, 0]],
                  [zeros[:, 0],    zeros[:, 0],     zeros[:, 0]],
                  [zeros[:, 0],    zeros[:, 0],     ca_ll[:, 0]],
                ]).transpose()
  
  m_rl = np.array([
                  [zeros[:, 0],    zeros[:, 0],     zeros[:, 0]],
                  [zeros[:, 0],    zeros[:, 0],     zeros[:, 0]],
                  [zeros[:, 0],    zeros[:, 0],     conj_ll[:, 0]],
                ]).transpose()
  return m_lr, m_rl






if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--tuple", help="Name of the tuple that contains the information")
    # parser.add_argument("--path", help="Specify path in case that is needed", default="")
    parser.add_argument("--treename", help="Specify treename", default="T")


t0 = time.time()
args = vars(parser.parse_args())

df_path = args["tuple"]
deltas = []
df = uproot.open(df_path)[args["treename"]].keys()
for i in df:
    if any(kw in i.decode() for kw in ["dd", "du"]):
        deltas.append(i.decode())
deltas += ["msq", "xdLR", "xuRL"]
df = uproot.open(df_path)[args["treename"]].pandas.df(branches=deltas)



# md_ll = df.eval("msq").values[:, np.newaxis, np.newaxis]*build3t3(df, "dd", "LL")
# md_rr = df.eval("msq/sqrt(xdLR)").values[:,np.newaxis, np.newaxis]*build3t3(df, "dd", "RR")
# mu_ll = df.eval("msq").values[:, np.newaxis, np.newaxis]*build3t3(df, "du", "LL")
# mu_rr = df.eval("msq*sqrt(xtRdL)").values[:, np.newaxis, np.newaxis]*build3t3(df, "du", "RR")
# df["xdLR"] = 10*np.ones_like(df["msq"])
print(df["msq"])
md_ll = df.eval("msq*msq").values[:, np.newaxis, np.newaxis]*build3t3(df, "dd", "LL")
md_rr = df.eval("msq*msq/xdLR").values[:,np.newaxis, np.newaxis]*build3t3(df, "dd", "RR")
mu_ll = df.eval("msq*msq").values[:, np.newaxis, np.newaxis]*build3t3(df, "du", "LL")
mu_rr = df.eval("msq*msq*xuRL").values[:, np.newaxis, np.newaxis]*build3t3(df, "du", "RR")

md_lr, md_rl = build3t3_lr(df, "dd", "LR")
md_lr = df.eval("msq*msq/sqrt(xdLR)").values[:, np.newaxis, np.newaxis]*md_lr
md_rl = df.eval("msq*msq/sqrt(xdLR)").values[:, np.newaxis, np.newaxis]*md_rl

mu_lr, mu_rl = build3t3_lr(df, "du", "LR")
mu_lr = df.eval("msq*msq*sqrt(xuRL)").values[:, np.newaxis, np.newaxis]*mu_lr
mu_rl = df.eval("msq*msq*sqrt(xuRL)").values[:, np.newaxis, np.newaxis]*mu_rl

zeros = np.zeros((md_ll.shape[0], md_ll.shape[1], md_ll.shape[2]))

# MD = np.block([
#     [np.concatenate((md_ll, zeros), axis=2)],
#     [np.concatenate((zeros, md_rr), axis=2)]])

# MU = np.block([
#     [np.concatenate((mu_ll, zeros), axis=2)],
#     [np.concatenate((zeros, mu_rr), axis=2)]])

MD = np.block([
    [np.concatenate((md_ll, md_lr), axis=2)],
    [np.concatenate((md_rl, md_rr), axis=2)]])

MU = np.block([
    [np.concatenate((mu_ll, mu_lr), axis=2)],
    [np.concatenate((mu_rl, mu_rr), axis=2)]])


eig_md = np.linalg.eigvalsh(MD)
min_md = np.sqrt(np.min(eig_md, axis=1))
# min_md = np.min(eig_md, axis=1)
eig_mu = np.linalg.eigvalsh(MU)
min_mu = np.sqrt(np.min(eig_mu, axis=1))
# min_mu = np.min(eig_mu, axis=1)

print("Constraint 1TeV:")
tick = np.count_nonzero((min_md>1e3) & (min_mu>1e3))
tot = float(len(min_md))
print tick/tot * 100


print(time.time()-t0)











# vim: fdm=marker ts=2 sw=2 sts=2 sr noet
