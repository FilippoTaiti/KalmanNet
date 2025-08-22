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
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt



class SystemModel:

    def __init__(self, F, Q, H, R, T_test, T=None, prior_Q=None, prior_Sigma=None, prior_S=None):

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
    '''def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R'''

    def GenerateSequence(self, T):
        # Pre allocate an array for current state
        self.x = torch.zeros(size=[self.m, T])
        # Pre allocate an array for current observation
        self.y = torch.zeros(size=[self.n, T])
        # Set x0 to be x previous
        self.x_prev = self.m1x_0
        xt = self.x_prev

        # Generate in a batched manner
        for t in range(0, T):
            ########################
            #### State Evolution ###
            ########################

            xt = self.F.matmul(self.x_prev)
            mean = torch.zeros([self.m])
            distrib = MultivariateNormal(loc=mean, covariance_matrix=self.Q)
            eq = distrib.rsample().view(self.m, 1)
            # Additive Process Noise
            xt = torch.add(xt, eq)
            #print("xt shape: ")

            ################
            ### Emission ###
            ################
            # Observation Noise

            #print("H shape: ", self.H.shape)
            #print("xt shape: ", xt.shape)
            tmp = self.H.matmul(xt).squeeze(0)
            #print("tmp shape: ", tmp.shape)
            yt = tmp.unsqueeze(2)
            #print("yt shape: ", yt.shape)
            #print("yt: ", yt)
            # print("yt: ", yt)
            # print("yt size: ", yt.size())
            rho = torch.sqrt(torch.add(torch.pow(yt[0], 2), torch.pow(yt[1], 2)))
            #print("rho shape: ", rho.shape)
            theta = torch.atan2(yt[1], yt[0])
            #print("theta shape: ", theta.shape)
            mean = torch.zeros([self.n])
            distrib = MultivariateNormal(loc=mean, covariance_matrix=self.R)
            er_polar = distrib.rsample()
            # print("er_polar: ", er_polar)
            #print("er_polar.shape: ", er_polar.shape)
            rho += er_polar[0]
            theta += er_polar[1]
            yt = torch.stack([rho * torch.cos(theta), rho * torch.sin(theta)], dim=1)
            # print("yt_n", yt_n)
            # print("yt_n shape", yt_n.shape)
            #print("yt shape: ", yt.shape)

            ########################
            ### Squeeze to Array ###
            ########################

            # Save Current State to Trajectory Array
            self.x[:, t] = torch.squeeze(xt, 1)

            # Save Current Observation to Trajectory Array
            #print("self.y shape: ", self.y.shape)
            tmp = yt.squeeze(2)
            #print("tmp pt1 shape: ", tmp.shape)
            yt = tmp.squeeze(1)
            #print("tmp pt2 shape: ", tmp.shape)
            self.y[:, t] = tmp

            ################################
            ### Save Current to Previous ###
            ################################
            self.x_prev = xt

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size, T, T_min, randomLength=False):

        self.m1x_0_rand = torch.zeros(size, self.m, 1)
        for i in range(size):
            distrib = MultivariateNormal(loc=torch.squeeze(self.m1x_0), covariance_matrix = self.m2x_0)
            #print(distrib.rsample().shape)
            initConditions = distrib.rsample().view(self.m, 1)
            #print(initConditions.shape)
            self.m1x_0_rand[i, :, 0:1] = initConditions

        self.Init_batched_sequence(self.m1x_0_rand, self.m2x_0)
        print(self.m1x_0_rand)

        if randomLength:
            # Allocate Empty Array for Input
            self.Input = torch.empty(size, self.n, T)
            # Allocate Empty Array for Target
            self.Target = torch.empty(size, self.m, T)
            self.lengthMask = torch.zeros((size, T), dtype=torch.bool)  # init with all false
            # Init Sequence Lengths
            T_tensor = torch.round((T - T_min) * torch.rand(size)).int() + T_min  # Uniform distribution [100,1000]
            for i in range(0, size):
                # Generate Sequence
                self.GenerateSequence(T_tensor[i].item())
                # Training sequence input
                self.Input[i, :, 0:T_tensor[i].item()] = self.y
                # Training sequence output
                self.Target[i, :, 0:T_tensor[i].item()] = self.x
                # Mask for sequence length
                self.lengthMask[i, 0:T_tensor[i].item()] = True

        else:

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
                #print("yt shape: ", yt.shape)
                # print("yt: ", yt)
                # print("yt size: ", yt.size())
                rho = torch.sqrt(yt[:, 0] ** 2 + yt[:, 1] ** 2)
                # print("rho: ", rho)
                theta = torch.atan2(yt[:, 1], yt[:, 0])
                # print("theta: ", theta)
                mean = torch.zeros([size, self.n])
                distrib = MultivariateNormal(loc=mean, covariance_matrix=self.R)
                er_polar = distrib.rsample()
                # print("er_polar: ", er_polar)
                # print("er_polar.shape: ", er_polar.shape)
                rho += er_polar[:, 0]
                theta += er_polar[:, 1]
                yt_n = torch.stack([rho * torch.cos(theta), rho * torch.sin(theta)], dim=1)
                # print("yt_n", yt_n)
                # print("yt_n shape", yt_n.shape)
                yt = yt_n.unsqueeze(2)

                ########################
                ### Squeeze to Array ###
                ########################

                # Save Current State to Trajectory Array
                self.Target[:, :, t] = torch.squeeze(xt, 2)

                # Save Current Observation to Trajectory Array
                self.Input[:, :, t] = torch.squeeze(yt, 2)

                ################################
                ### Save Current to Previous ###
                ################################
                self.x_prev = xt






def plot_testset(test_input, test_target, size):
    X_batch_input = []
    Y_batch_input = []

    X_batch_target = []
    Y_batch_target = []


    for b_i in range(size):
        X_input = test_input[b_i, 0, :]
        Y_input = test_input[b_i, 1, :]

        X_batch_input.append(X_input)
        Y_batch_input.append(Y_input)

        X_target = test_target[b_i, 0, :]
        Y_target = test_target[b_i, 2, :]

        X_batch_target.append(X_target)
        Y_batch_target.append(Y_target)

    for i in range(len(X_batch_input)):
        figure, axes = plt.subplots(1)
        axes.plot(X_batch_input[i], Y_batch_input[i], 'y', label="Input Sequences")
        axes.plot(X_batch_target[i], Y_batch_target[i], 'g', label="Target Sequences")
        plt.show()
        plt.close(figure)






















