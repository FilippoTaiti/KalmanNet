"""# **Class: System Model for Linear Cases**

1 Store system model parameters: 
    state transition matrix F, 
    observation matrix H, 
    process noise covariance matrix Q, 
    observation noise covariance matrix R, 
    train&CV dataset sequence length T,
    test dataset sequence length T_test, etc.

2 Generate dataset for linear cases
"""
import math
import random

import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class SystemModel:

    def __init__(self, F, Q, H, R, T, T_test, prior_Q=None, prior_Sigma=None, prior_S=None):

        ####################
        ### Motion Model ###
        ####################
        self.F = F
        self.m = self.F.size()[0]
        self.Q = Q

        #########################
        ### Observation Model ###
        #########################
        self.H = H
        self.n = self.H.size()[0]
        self.R = R

        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T
        self.T_test = T_test

        #########################
        ### Covariance Priors ###
        #########################
        if prior_Q is None:
            self.prior_Q = torch.eye(self.m)
        else:
            self.prior_Q = prior_Q

        if prior_Sigma is None:
            self.prior_Sigma = torch.zeros((self.m, self.m))
        else:
            self.prior_Sigma = prior_Sigma

        if prior_S is None:
            self.prior_S = torch.eye(self.n)
        else:
            self.prior_S = prior_S
        

    def f(self, x):
        batched_F = self.F.to(x.device).view(1,self.F.shape[0],self.F.shape[1]).expand(x.shape[0],-1,-1)
        #print("batched_F ", batched_F.dtype)
        #print("x ", x.dtype)
        return torch.bmm(batched_F, x)
    
    def h(self, x):
        batched_H = self.H.to(x.device).view(1,self.H.shape[0],self.H.shape[1]).expand(x.shape[0],-1,-1)
        return torch.bmm(batched_H, x)
        
    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.x_prev = m1x_0
        self.m2x_0 = m2x_0

    def Init_batched_sequence(self, m1x_0_batch, m2x_0_batch):

        self.m1x_0_batch = m1x_0_batch
        self.x_prev = m1x_0_batch
        self.m2x_0_batch = m2x_0_batch

    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R


    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T, m2x_0):

        self.InitSequence(torch.tensor([[0],  [random.uniform(5, 20)], [0], [random.uniform(5, 20)]]).float(), m2x_0)
        initConditions = self.m1x_0.view(1, self.m, 1).expand(size, -1, -1)
        self.Init_batched_sequence(initConditions, m2x_0)

        # Allocate Empty Array for Input
        self.Input = torch.empty(size, self.n, T)
        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, T)

        # Set x0 to be x previous
        self.x_prev = self.m1x_0_batch
        xt = self.x_prev


        # Generate in a batched manner
        for t in range(0, T):
            ########################
            #### State Evolution ###
            ########################

            xt = self.f(self.x_prev)
            mean = torch.zeros([size, self.m])
            distrib = MultivariateNormal(loc=mean, covariance_matrix=self.Q)
            eq = distrib.rsample().view(size, self.m, 1)
            # Additive Process Noise
            xt = torch.add(xt, eq)

            ################
            ### Emission ###
            ################
            # Observation Noise

            yt = self.H.matmul(xt).squeeze(2)
            #print("yt: ", yt)
            # print("yt size: ", yt.size())
            rho = torch.sqrt(yt[:, 0]**2 + yt[:, 1]**2)
            #print("rho: ", rho)
            theta = torch.atan2(yt[:, 1], yt[:, 0])
            #print("theta: ", theta)
            mean = torch.zeros([size, self.n])
            distrib = MultivariateNormal(loc=mean, covariance_matrix=self.R)
            er_polar = distrib.rsample()
            rho += er_polar[:, 0]
            theta += er_polar[:, 1]
            yt_n = torch.stack([rho * torch.cos(theta), rho * torch.sin(theta)], dim=1)
            yt = yt_n.unsqueeze(2)

            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.Target[:, :, t] = torch.squeeze(xt,2)

            # Save Current Observation to Trajectory Array
            self.Input[:, :, t] = torch.squeeze(yt,2)

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt


