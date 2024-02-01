## high masses and very tiny LL and RR
## parameters treated as reals
msqmin, msqmax = 2000, 10000
Mgmin, Mgmax = 2000, 10000
tanbmin, tanbmax = 2.,50.#10.,50.
MAmin , MAmax = 2000.,10000.

r_mumin, r_mumax = 1000,10000#-10000, -1000
r_M1min, r_M1max = 1., 1. ## universality level. M1 = r_M1 *M3 * alpha1/alpha3
r_M2min, r_M2max = 1., 1. ## universality level. M2 = r_M2 *M3 * alpha2/alpha3


## NB : so far, set to 0
Atmin, Atmax = 0,0
Acmin, Acmax = 0,0
Aumin, Aumax = 0,0

xdLRmin, xdLRmax = .25, 4.
xuRLmin, xuRLmax = 1.,1.
xtRdLmin, xtRdLmax = 1.,1.
xtLdLmin, xtLdLmax = 1.,1.

BG_min, BG_max = 0. , 0.
rhv = 0.

msnumin, msnumax = 1000., 10000. #Only one contribution to C9

###########
# LL MI's #
###########

# dd_LL, real
r_dd12LLmin, r_dd12LLmax = 0.,0.#:D
r_dd13LLmin, r_dd13LLmax = -0.4,0.4#:D
r_dd23LLmin, r_dd23LLmax = -0.4,0.4#:D

# dd_LL, imag
i_dd12LLmin, i_dd12LLmax = 0.,0.#:D
i_dd13LLmin, i_dd13LLmax = -0.4,0.4#:
i_dd23LLmin, i_dd23LLmax = -0.4,0.4#:

#Not used: Calculated by SU(2) symmetry
#Needed for saving 
r_du12LLmin, r_du12LLmax = -0.,0.
r_du13LLmin, r_du13LLmax = 0.,0.
r_du23LLmin, r_du23LLmax = 0.,0.
i_du12LLmin, i_du12LLmax = -0.,0.
i_du13LLmin, i_du13LLmax = 0.,0.
i_du23LLmin, i_du23LLmax = 0.,0.

#Not used: Calculated by SU(2) symmetry
#Needed for saving 
r_dd33LRmin, r_dd33LRmax = 0., 0.
r_du33LRmin, r_du33LRmax = 0., 0.
i_dd33LRmin, i_dd33LRmax = 0., 0.
i_du33LRmin, i_du33LRmax = 0., 0.



############
## RR MI's #
############

# RR, real
r_dd12RRmin, r_dd12RRmax = -0.,0.#:
r_dd13RRmin, r_dd13RRmax = -0.4,0.4#:
r_dd23RRmin, r_dd23RRmax = -0.4,0.4#:

r_du12RRmin, r_du12RRmax = 0.,0.#:
r_du13RRmin, r_du13RRmax = 0.,0.#:
r_du23RRmin, r_du23RRmax = 0.,0.#:

i_dd12RRmin, i_dd12RRmax = -0.0,0.0#:
i_dd13RRmin, i_dd13RRmax = -0.4,0.4#:
i_dd23RRmin, i_dd23RRmax = -0.4,0.4#:

i_du12RRmin, i_du12RRmax = 0.0,0.0#:
i_du13RRmin, i_du13RRmax = 0.,0.#:
i_du23RRmin, i_du23RRmax = 0.,0.#:



#################
## LR, RL MI's #
#################
# r_dd21LRmin, r_dd21LRmax = -0., +0.
# r_dd23LRmin, r_dd23LRmax = -0., +0.

# i_dd21LRmin, i_dd21LRmax = -0., +0.
# i_dd23LRmin, i_dd23LRmax = -0., +0.

# r_dd23RLmin, r_dd23RLmax = -0., 0.
# i_dd23RLmin, i_dd23RLmax = -0., 0.

# r_du23LRmin, r_du23LRmax = -0., +0.
# i_du23LRmin, i_du23LRmax = -0., 0.

#In the future change ddRR for ddXX = ddLL * ddRR
############
## XX MI's # ### dXX = dLL*dRR
############
## Mutate XX instead of RR 
# r_dd12RRmin, r_dd12RRmax = 0.,0.#:D
# r_dd13RRmin, r_dd13RRmax = 0.,0.#:D
# r_dd23XXmin, r_dd23XXmax = -1.0e-2,1.0e-2

# r_du12RRmin, r_du12RRmax = -0.,0.#:
# r_du13RRmin, r_du13RRmax = 0.,0.#:
# r_du23RRmin, r_du23RRmax = 0.,0.#:
# LL, imag
# i_dd12RRmin, i_dd12RRmax = 0.,0.#:D
# i_dd13RRmin, i_dd13RRmax = 0.,0.#:
# i_dd23XXmin, i_dd23XXmax = -1.0e-2,1.0e-2#:

# i_du12RRmin, i_du12RRmax = -0.,0.#:
# i_du13RRmin, i_du13RRmax = 0.,0.#:
# i_du23RRmin, i_du23RRmax = 0.,0.#

### NB: limit before was in 2.0e-3
