"""
This file contains the class Pipeline_EKF, 
which is used to train and test KalmanNet.
"""

import torch
import torch.nn as nn
import random
import time
import numpy as np
from matplotlib import pyplot as plt
from torch.nn import SmoothL1Loss
from torch_optimizer import Lookahead
import torch.optim.lr_scheduler as lr_scheduler


class Pipeline_EKF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"


    def save(self):
        torch.save(self, self.PipelineName)


    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model):
        self.model = model

    def setTrainingParams(self, n_steps, n_batch, lr, wd, alpha):
        self.device = torch.device('cpu')
        self.N_steps = n_steps  # Number of Training Steps
        self.N_B = n_batch # Number of Samples in Batch
        self.learningRate = lr # Learning Rate
        self.weightDecay = wd # L2 Weight Regularization - Weight Decay
        self.alpha = alpha # Composition loss factor
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algorithms. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        base_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        self.optimizer = Lookahead(base_optimizer)

    def defineLossFunction(self, epoch):
        if epoch < 20:
            return nn.SmoothL1Loss()
        else:
            return nn.MSELoss(reduction='mean')


    def NNTrain(self, SysModel, cv_input, cv_target, train_input, train_target, path_results, CompositionLoss, loadModel):

        if loadModel:
            checkpoint = torch.load(path_results+'best-model-200try3.pt', map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        startEpoch = 0


        #self.model = torch.load(path_results + 'best-model-random200.pt', map_location=self.device, weights_only=False)

        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])

        scheduler = lr_scheduler.OneCycleLR(self.optimizer, max_lr=5e-4, epochs= self.N_steps, steps_per_epoch=30)

        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(startEpoch, self.N_steps):
            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()
            # Training Mode
            self.model.train()
            self.model.batch_size = self.N_B
            # Init Hidden State
            self.model.init_hidden_KNet()

            # Init Training Batch tensors
            y_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device)
            train_target_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
            x_out_training_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device)
            # Randomly select N_B training sequences
            assert self.N_B <= self.N_E  # N_B must be smaller than N_E
            n_e = random.sample(range(self.N_E), k=self.N_B)
            ii = 0
            for index in n_e:

                y_training_batch[ii, :, :] = train_input[index]
                train_target_batch[ii, :, :] = train_target[index]
                ii += 1

            # Init Sequence
            self.model.InitSequence(SysModel.m1x_0.reshape(1, SysModel.m, 1).repeat(self.N_B, 1, 1), SysModel.T)

            # Forward Computation
            for t in range(0, SysModel.T):
                x_out_training_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch[:, :, t], 2)))


            # Compute Training Loss
            MSE_trainbatch_linear_LOSS = 0
            if CompositionLoss:
                y_hat = torch.zeros([self.N_B, SysModel.n, SysModel.T])
                for t in range(SysModel.T):
                    y_hat[:, :, t] = torch.squeeze(SysModel.h(torch.unsqueeze(x_out_training_batch[:, :, t], dim=2)), dim=2)
                    MSE_trainbatch_linear_LOSS = self.alpha * self.loss_fn(x_out_training_batch,
                                                                               train_target_batch) + (
                                                                 1 - self.alpha) * self.loss_fn(y_hat, y_training_batch)
            else:  # no composition loss
              MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch, train_target_batch)

            # dB Loss
            self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            MSE_trainbatch_linear_LOSS.backward(retain_graph=True)
            #print("norma: ", torch.norm(10*torch.log10(MSE_trainbatch_linear_LOSS)))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()
            scheduler.step()

            #################################
            ### Validation Sequence Batch ###
            #################################

            #valid_loss = 0
            # Cross Validation Mode
            self.model.eval()
            self.model.batch_size = self.N_CV
            # Init Hidden State
            self.model.init_hidden_KNet()
            with torch.no_grad():

                SysModel.T_test = cv_input.size()[-1]  # T_test is the maximum length of the CV sequences

                x_out_cv_batch = torch.empty([self.N_CV, SysModel.m, SysModel.T_test]).to(self.device)

                # Init Sequence

                self.model.InitSequence(SysModel.m1x_0.reshape(1, SysModel.m, 1).repeat(self.N_CV, 1, 1), SysModel.T_test)

                for t in range(0, SysModel.T_test):
                    x_out_cv_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(cv_input[:, :, t], 2)))

                # Compute CV Loss
                MSE_cvbatch_linear_LOSS = 0

                MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch, cv_target)

                '''valid_loss += MSE_cvbatch_linear_LOSS.item()
                # print(f'Epoch {ti}: Validation Loss: {valid_loss:.6f}')
                early_stop(valid_loss, self.model)
                if early_stop.stop:
                    print('Early stopping')
                    break'''
                #scheduler.step(MSE_trainbatch_linear_LOSS.item())
                # dB Loss
                self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])
                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti

                    torch.save({
                        'epoch': ti, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.optimizer.state_dict()
                    }, path_results+'best-model-200try3.pt')
            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :",
                  self.MSE_cv_dB_epoch[ti],
                  "[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

        self.learningCurve(self.MSE_train_dB_epoch, self.MSE_cv_dB_epoch)


        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def NNTest(self, SysModel, test_input, test_target, path_results, load_model=False, load_model_path=None):
        # Load model
        checkpoint = torch.load(path_results + 'best-model-200try2.pt', map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.N_T = test_input.shape[0]
        SysModel.T_test = test_input.size()[-1]
        self.MSE_test_linear_arr = torch.zeros([self.N_T])
        x_out_test = torch.zeros([self.N_T, SysModel.m, SysModel.T_test]).to(self.device)

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Test mode
        self.model.eval()
        self.model.batch_size = self.N_T
        # Init Hidden State
        self.model.init_hidden_KNet()
        torch.no_grad()
        MSESingleTrajectory = torch.zeros([self.N_T, SysModel.T_test])

        start = time.time()


        self.model.InitSequence(SysModel.m1x_0.reshape(1, SysModel.m, 1).repeat(self.N_T, 1, 1), SysModel.T_test)

        for t in range(0, SysModel.T_test):
            x_out_test[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(test_input[:, :, t], 2)))

        end = time.time()
        t = end - start

        # MSE loss
        for j in range(self.N_T):  # cannot use batch due to different length and std computation
            self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j, :, :], test_target[j, :, :]).item()
            for k in range(SysModel.T_test):
                #print(type(test_target[j, :, k]), type(x_out_test[j, :, k]))
                MSESingleTrajectory[j][k] = loss_fn(x_out_test[j, :, k], test_target[j, :, k]).item()


        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # Print MSE and std
        str = self.modelName + "-" + "MSE LOSS:"
        print("--", str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD:"
        print("--", str, self.test_std_dB, "[dB]")
        # Print Run Time
        #print("Inference Time:", t)

        KalmanGainKN = self.model.KGain
        MSESingleTrajectory = 10*torch.log10(MSESingleTrajectory)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t, KalmanGainKN, MSESingleTrajectory]


    def learningCurve(self, trainLoss, validationLoss):
        print(trainLoss.shape)
        print(validationLoss.shape)
        plt.plot(range(1, len(trainLoss)+1), trainLoss, label='Train Loss', marker='o')
        plt.plot(range(1, len(validationLoss)+1), validationLoss, label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close('all')
















