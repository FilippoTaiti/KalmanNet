"""# **Class: KalmanNet**"""

import torch
import torch.nn as nn
import torch.nn.functional as func

class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
    
    def NNBuild(self, SysModel):

        self.device = torch.device('cpu')

        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)

        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, 30, 5, 40)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, n_batch, in_mult_KNet, out_mult_KNet):

        self.seq_len_input = 1 # KNet calculates time-step by time-step
        self.batch_size = n_batch # Batch size

        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)
        


        # GRU to track Q
        self.d_input_Q = self.m * in_mult_KNet #4*5=20
        self.d_hidden_Q = self.m ** 2 #4**2=16
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        # GRU to track Sigma
        self.d_input_Sigma = self.d_hidden_Q #16
        self.d_hidden_Sigma = self.m ** 2 #4**2 = 16
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)
       
        # GRU to track S
        self.d_input_S = self.n + 2 * self.n #2+2*2=6
        self.d_hidden_S = self.n ** 2 #2**2=4
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)
        
        # Fully connected 1
        self.d_input_FC1 = self.d_hidden_Sigma #16
        self.d_output_FC1 = self.n ** 2 #2**2 = 4
        self.FC1 = nn.Sequential(
                nn.Linear(self.d_input_FC1, self.d_output_FC1),
                nn.ReLU(), nn.Dropout(p=0.1)).to(self.device)

        # Fully connected 2
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma #4+16=20
        self.d_output_FC2 = self.n * self.m #2*4=8
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult_KNet #20*40=8000
        self.FC2 = nn.Sequential(
                nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                nn.ReLU(),
                nn.Linear(self.d_hidden_FC2, self.d_output_FC2), nn.Dropout(p=0.5)).to(self.device)

        # Fully connected 3
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2 #4+8=12
        self.d_output_FC3 = self.m ** 2 #4**2=16
        self.FC3 = nn.Sequential(
                nn.Linear(self.d_input_FC3, self.d_output_FC3),
                nn.ReLU(), nn.Dropout(p=0.1)).to(self.device)

        # Fully connected 4
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3 #16+16=32
        self.d_output_FC4 = self.d_hidden_Sigma #16
        self.FC4 = nn.Sequential(
                nn.Linear(self.d_input_FC4, self.d_output_FC4),
                nn.ReLU(), nn.Dropout(p=0.1)).to(self.device)
        
        # Fully connected 5
        self.d_input_FC5 = self.m #4
        self.d_output_FC5 = self.m * in_mult_KNet #4*5=20
        self.FC5 = nn.Sequential(
                nn.Linear(self.d_input_FC5, self.d_output_FC5),
                nn.ReLU(), nn.Dropout(p=0.1)).to(self.device)

        
        # Fully connected 7
        self.d_input_FC7 = self.n #2
        self.d_output_FC7 = self.n #2
        self.FC7 = nn.Sequential(
                nn.Linear(self.d_input_FC7, self.d_output_FC7),
                nn.ReLU(), nn.Dropout(p=0.1)).to(self.device)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, f, h, m, n):
        
        # Set State Evolution Function
        self.f = f
        self.m = m

        # Set Observation Function
        self.h = h
        self.n = n

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, T):
        """
        input M1_0 (torch.tensor): 1st moment of x at time 0 [batch_size, m, 1]
        """
        self.T = T

        self.m1x_posterior = M1_0.to(self.device)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_prior_previous = self.m1x_posterior
        self.y_previous = self.h(self.m1x_posterior)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Predict the 1-st moment of x
        self.m1x_prior = self.f(self.m1x_posterior)

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):
      #[batch_size, n]
        obs_innov_diff = torch.squeeze(y,2) - torch.squeeze(self.m1y,2) #F2
        m1x_prior_previous = torch.squeeze(self.m1x_prior_previous, 2)

        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)

        m1x_prior_previous = func.normalize(m1x_prior_previous, p=2, dim=1, eps=1e-12, out=None)

        # Kalman Gain Network Step
        KG = self.KGain_step(obs_innov_diff, m1x_prior_previous)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):

        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        dy = y - self.m1y # [batch_size, n, 1]


        # Compute the 1-st posterior moment
        INOV = torch.bmm(self.KGain, dy)
        self.m1x_posterior_previous = self.m1x_posterior
        self.m1x_posterior = self.m1x_prior + INOV

        #self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior


        # update y_prev
        self.y_previous = y


        #print("dy: ", dy.abs().max().item())
        #print("self.m1x_posterior_previous: ", self.m1x_posterior_previous.abs().max().item())
        #print("self.m1x_posterior: ", self.m1x_posterior.abs().max().item())

        # return
        return self.m1x_posterior

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, obs_innov_diff, m1x_prior_previous):

        def expand_dim(x):
            #print("x shape", x.shape)
            #print("self seq len input: ", self.seq_len_input, "self  batch size: ", self.batch_size, "xshape[-1]: ", x.shape[-1])
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded


        obs_innov_diff = expand_dim(obs_innov_diff) #F2
        m1x_prior_previous = expand_dim(m1x_prior_previous)

        ####################
        ### Forward Flow ###
        ####################
        
        # FC 5
        in_FC5 = m1x_prior_previous
        out_FC5 = self.FC5(in_FC5)

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q)


        # Sigma_GRU
        in_Sigma = out_Q
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma)

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1)

        # FC 7
        in_FC7 = obs_innov_diff
        out_FC7 = self.FC7(in_FC7)


        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)


        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2)
        out_FC2 = self.FC2(in_FC2)

        #####################
        ### Backward Flow ###
        #####################

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2)
        out_FC3 = self.FC3(in_FC3)

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4 = self.FC4(in_FC4)

        # updating hidden state of the Sigma-GRU
        self.h_Sigma = out_FC4

        return out_FC2
    ###############
    ### Forward ###
    ###############
    def forward(self, y):
        y = y.to(self.device)
        return self.KNet_step(y)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden_KNet(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden.data
        self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1) # batch size expansion
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden.data
        self.h_Sigma = self.prior_Sigma.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1) # batch size expansion
        hidden = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden.data
        self.h_Q = self.prior_Q.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1) # batch size expansion



