import random

import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt


from Simulations.Linear_sysmdl import SystemModel, plot_testset
from Simulations.utils import DataTestGen, plot_results, plotSquaredError, plotBoxPlot, plotAverageTrajectories
from Simulations.Radar_2d.parameters import F, H, Q, R_kf, R_Knet, m

from Filters.KalmanFilter_test import KFTest

from KNet.KalmanNet_nn import KalmanNetNN

from Pipelines.Pipeline import Pipeline_EKF
def test(T_test, T_min=10):
   print("Pipeline Start")

   ################
   ### Get Time ###
   ################
   today = datetime.today()
   now = datetime.now()
   strToday = today.strftime("%m.%d.%y")
   strNow = now.strftime("%H:%M:%S")
   strTime = strToday + "_" + strNow
   print("Current Time =", strTime)
   path_results = 'KNet/'

   ### dataset parameters ##################################################
   N_T = 20000  # Numero di sequenze del Test Set
   # init condition



   randomLength = False


   #Questi parametri di training ce li ho lasciati per adesso ma qui non vengono usati, qui vengono solo fatti i test

   ### training parameters ##################################################
   N_steps = 1000  # Numero epoche
   N_batch = 100  # Dimensione del singolo batch
   lr = 1e-3
   wd = 1e-4


   device = torch.device('cpu')

   m1_0 = torch.zeros(m, 1)
   m2_0 = 1 * torch.eye(m)


   ### True model ##################################################
   sys_model = SystemModel(F, Q, H, R_kf, R_Knet, T_test)
   sys_model.InitSequence(m1_0, m2_0)


   ###################################
   ### Data Loader (Generate Data) ###
   ###################################
   dataFolderName = 'Simulations/Radar_2d/data' + '/'
   dataFileName = 'test_data.pt'
   print("Start Test Data Gen")
   DataTestGen(sys_model, dataFolderName+dataFileName, N_T, T_min, T_test, randomLength)
   print("Test Data Load")
   [test_input, test_target,_] = torch.load(dataFolderName + dataFileName, map_location=device)

   ########################################
   ### Evaluate Observation Noise Floor ###
   ########################################
   loss_obs = nn.MSELoss(reduction='mean')
   MSE_obs_linear_arr = torch.empty(N_T)# MSE [Linear]
   for i in range(N_T):
      mask = torch.tensor([True, False, True, False])
      MSE_obs_linear_arr[i] = loss_obs(test_input[i], test_target[i][mask]).item()
   MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
   MSE_obs_dB_arr = 10*torch.log10(MSE_obs_linear_arr)
   MSE_obs_dB_avg = torch.mean(MSE_obs_dB_arr)

   # Standard deviation
   MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

   # Confidence interval
   obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

   print("--Observation Noise Floor - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
   print("--Observation Noise Floor - STD:", obs_std_dB, "[dB]")

   print("Evaluate Kalman Filter True")
   [MSE_KF_linear_arr, _, _, KF_out, KalmanGainKF, squaredErrorKF] = KFTest(sys_model, N_T, T_test, test_input, test_target)


   ##########################
   ### KalmanNet Pipeline ###
   ##########################

   ### KalmanNet with full info ##########################################################################################
   # Build Neural Network
   #print("KalmanNet with full model info")
   KalmanNet_model = KalmanNetNN()
   KalmanNet_model.NNBuild(sys_model)
   #print("Number of trainable parameters for KalmanNet:",sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))
   ## Train Neural Network
   KalmanNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
   KalmanNet_Pipeline.setssModel(sys_model)
   KalmanNet_Pipeline.setModel(KalmanNet_model)
   KalmanNet_Pipeline.setTrainingParams(N_steps, N_batch, lr, wd, 0.3)

   [MSE_test_linear_arr, _, _, knet_out, _, _, squaredErrorKNet] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)


   MSE_KF_db_arr = 10*torch.log10(MSE_KF_linear_arr)
   MSE_test_db_arr = 10*torch.log10(MSE_test_linear_arr)




   #plot_testset(test_input, test_target, N_T, randomLength)
   indexes = random.sample(range(N_T), 10)
   plot_results(test_target, KF_out, knet_out, indexes)
   plotBoxPlot(MSE_obs_dB_arr, MSE_KF_db_arr, MSE_test_db_arr)
   plotSquaredError(squaredErrorKF, squaredErrorKNet, indexes)
   plotAverageTrajectories(squaredErrorKF, squaredErrorKNet)



test(200)





















