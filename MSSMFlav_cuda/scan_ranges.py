## parameters treated as reals
msqmin, msqmax = 2000, 10000
Mgmin, Mgmax = 2000, 15000
tanbmin, tanbmax = 2.,50.
MAmin , MAmax = 2000.,10000.

# r_mumin, r_mumax = 200, 1400
r_mumin, r_mumax = 5e3, 5e4
r_M1min, r_M1max = .2, 4 ## universality level. M1 = r_M1 *M3 * alpha1/alpha3
r_M2min, r_M2max = .1, 2 ## universality level. M2 = r_M2 *M3 * alpha2/alpha3

# Atmin, Atmax = -5000,5000#-1000,1000
Atmin, Atmax = -5000., 5000.
Acmin, Acmax = 0,0#-1000,1000
Aumin, Aumax = 0,0#-1000,1000

# BG_min, BG_max = -2. , 4. #WR 
BG_min, BG_max = 0. , 0. #WR 

# xdLRmin, xdLRmax = .5, 10.
xdLRmin, xdLRmax = .25, 4.
#xudLmin, xudLmax = 1.,1.
xtRdLmin, xtRdLmax = .5,10.
xtLdLmin, xtLdLmax = .5,10.

rhv = 0.

msnumin, msnumax = 500., 10000.
###########
# LL MI's #
###########
# LL, real
r_dd12LLmin, r_dd12LLmax = -0.85, 0.85#:D
r_dd13LLmin, r_dd13LLmax = -0.85,0.85#:D
r_dd23LLmin, r_dd23LLmax = -0.85,0.85#:D

r_du12LLmin, r_du12LLmax = -0.85,0.85#:
r_du13LLmin, r_du13LLmax = 0.85,0.85#:
r_du23LLmin, r_du23LLmax = -0.85,0.85#:
# LL, imag
# i_dd12LLmin, i_dd12LLmax = -0.,0.#:D
# i_dd13LLmin, i_dd13LLmax = -0,0.#:
# i_dd23LLmin, i_dd23LLmax = -0.,0.#:
#
# i_du12LLmin, i_du12LLmax = -0.,0.#:
# i_du13LLmin, i_du13LLmax = 0.,0.#:
# i_du23LLmin, i_du23LLmax = 0.,0.#:

i_dd12LLmin, i_dd12LLmax = -0.85, 0.85#:D
i_dd13LLmin, i_dd13LLmax = -0.85,0.85#:D
i_dd23LLmin, i_dd23LLmax = -0.85,0.85#:D

i_du12LLmin, i_du12LLmax = -0.85,0.85#:
i_du13LLmin, i_du13LLmax = 0.85,0.85#:
i_du23LLmin, i_du23LLmax = -0.85,0.85#:
############
## RR MI's #
############
# RR, real
r_dd12RRmin, r_dd12RRmax = -0.85,0.85#:
r_dd13RRmin, r_dd13RRmax = 0.85,0.85#:
r_dd23RRmin, r_dd23RRmax = -0.85,0.85#:

r_du12RRmin, r_du12RRmax = 0.85,0.85#:
r_du13RRmin, r_du13RRmax = -0.85,0.85#:
r_du23RRmin, r_du23RRmax = -0.85,0.85#:
## RR, imag
# i_dd12RRmin, i_dd12RRmax = -0.,0.#:
# i_dd13RRmin, i_dd13RRmax = 0.,0.#:
# i_dd23RRmin, i_dd23RRmax = -0.,0.#:

# i_du12RRmin, i_du12RRmax = 0.,0.#:
# i_du13RRmin, i_du13RRmax = -0.,0.#:
# i_du23RRmin, i_du23RRmax = 0.,0.#:

i_dd12RRmin, i_dd12RRmax = -0.85,0.85#:
i_dd13RRmin, i_dd13RRmax = 0.85,0.85#:
i_dd23RRmin, i_dd23RRmax = -0.85,0.85#:

i_du12RRmin, i_du12RRmax = 0.85,0.85#:
i_du13RRmin, i_du13RRmax = -0.85,0.85#:
i_du23RRmin, i_du23RRmax = -0.85,0.85#:


#################
## LR, RL MI's #
#################
r_dd21LRmin, r_dd21LRmax = -0.85, +0.85
r_dd23LRmin, r_dd23LRmax = -0.85, +0.85
r_dd33LRmin, r_dd33LRmax = -0.85, +0.85

i_dd21LRmin, i_dd21LRmax = -0.85, +0.85
i_dd23LRmin, i_dd23LRmax = -0.85, +0.85
i_dd33LRmin, i_dd33LRmax = -0.85, +0.85

r_dd23RLmin, r_dd23RLmax = -0.85, 0.85
r_dd32RLmin, r_dd32RLmax = -0.85, 0.85

i_dd23RLmin, i_dd23RLmax = -0.85, 0.85
i_dd32RLmin, i_dd32RLmax = -0.85, 0.85

r_du23LRmin, r_du23LRmax = -0.85, 0.85
r_du33LRmin, r_du33LRmax = -0.85, 0.85

i_du23LRmin, i_du23LRmax = -0.85, 0.85
i_du33LRmin, i_du33LRmax = -0.85, 0.85


