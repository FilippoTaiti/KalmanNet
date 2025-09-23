"""
This file contains the parameters for the simulations with linear canonical model
* Linear State Space Models with Full Information
    # v = 0, -10, -20 dB
    # scaling model dim to 5x5, 10x10, 20x20, etc
    # scalable trajectory length T
    # random initial state
* Linear SS Models with Partial Information
    # observation model mismatch
    # evolution model mismatch
"""

import torch
import numpy as np

m = 4 # Lo stato è della forma [px, py, vx, vy]
n = 2 # observation dimension = 2 -> mi interessa osservare solo la posizione
dt = 1

##################################
### Initial state and variance ###
##################################
#m1_0 = torch.zeros(m, 1) # initial state mean




#########################################################
### state evolution matrix F and observation matrix H ###
#########################################################

F = torch.tensor([
    [1, dt, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, dt],
    [0, 0, 0, 1]
]).float()


H = torch.tensor([
    [1, 0, 0, 0],  # px
    [0, 0, 1, 0],  # py
]).float()

#MATRICE DI COVARIANZA DEL DISTURBO SULLO STATO
Q = 5*torch.tensor([[(dt**3)/3, (dt**2)/2, 0, 0],[(dt**2)/2, dt, 0, 0], [0, 0, (dt**3)/3, (dt**2)/2], [0, 0, (dt**2)/2, dt]]).float()

#matrici di covarianza del disturbo di misura
r_knet = torch.tensor([4, 0.0002]).float()
R_Knet = torch.diag(r_knet)

r_kf = torch.tensor([7**2, 7**2]).float()
R_kf = torch.diag(r_kf)


#con best-model-200try2.pt -->

#50 --> [0.05 , 25.5]

#100 --> [2.4 , 26]

#200 --> [24, 26]

#NEW

#50 --> [0.9, 12]

#100 --> [4.8 , 16.5]

#200 --> [ , ]

#50. --> [3 , 4]









'''# F in canonical form
F = torch.eye(m)
F[0] = torch.ones(1,m) 

if m == 2:
    # H = I
    H = torch.eye(2)
else:
    # H in reverse canonical form
    H = torch.zeros(n,n)
    H[0] = torch.ones(1,n)
    for i in range(n):
        H[i,n-1-i] = 1

#######################
### Rotated F and H ###
#######################
F_rotated = torch.zeros_like(F)
H_rotated = torch.zeros_like(H)
if(m==2):
    alpha_degree = 10 # rotation angle in degree
    rotate_alpha = torch.tensor([alpha_degree/180*torch.pi])
    cos_alpha = torch.cos(rotate_alpha)
    sin_alpha = torch.sin(rotate_alpha)
    rotate_matrix = torch.tensor([[cos_alpha, -sin_alpha],
                                [sin_alpha, cos_alpha]])

    F_rotated = torch.mm(F,rotate_matrix) 
    H_rotated = torch.mm(H,rotate_matrix) 

###############################################
### process noise Q and observation noise R ###
###############################################
# Noise variance takes the form of a diagonal matrix
Q_structure = torch.eye(m)
R_structure = torch.eye(n) '''

