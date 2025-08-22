import random

import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt


from Simulations.Linear_sysmdl import SystemModel, plot_testset
from Simulations.utils import DataTestGen, plot_results
from Simulations.Radar_2d.parameters import F, H, Q, R, m

from Filters.KalmanFilter_test import KFTest

from KNet.KalmanNet_nn import KalmanNetNN

from Pipelines.Pipeline_EKF import Pipeline_EKF
def test(T_min, T_test):
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
   N_T = 10   # Numero di sequenze del Test Set
   # init condition



   randomLength = True
   MaskOnState = False


   #Questi parametri di training ce li ho lasciati per adesso ma qui non vengono usati, qui vengono solo fatti i test

   ### training parameters ##################################################
   N_steps = 1000  # Numero epoche
   N_batch = 100  # Dimensione del singolo batch
   lr = 1e-3
   wd = 1e-4


   device = torch.device('cpu')

   m1_0 = torch.randn(m, 1)
   m2_0 = random.uniform(1, 100) * torch.eye(m)


   ### True model ##################################################
   sys_model = SystemModel(F, Q, H, R, T_test)
   sys_model.InitSequence(m1_0, m2_0)

   ###################################
   ### Data Loader (Generate Data) ###
   ###################################
   dataFolderName = 'Simulations/Radar_2d/data' + '/'
   dataFileName = 'test_data.pt'
   print("Start Test Data Gen")
   DataTestGen(sys_model, dataFolderName+dataFileName, N_T, T_min, T_test, randomLength)
   print("Test Data Load")
   if randomLength:
      [test_input, test_target, _, test_lengthMask] = torch.load(dataFolderName + dataFileName, map_location=device)
   else:
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

   print("Observation Noise Floor - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
   print("Observation Noise Floor - STD:", obs_std_dB, "[dB]")

   print("Evaluate Kalman Filter True")
   if randomLength:
      [MSE_KF_linear_arr, _, _, KF_out, KalmanGainKF] = KFTest(sys_model, N_T, T_test, test_input, test_target, randomLength, test_lengthMask=test_lengthMask)
   else:
      [MSE_KF_linear_arr, _, _, KF_out, KalmanGainKF] = KFTest(sys_model, N_T, T_test, test_input, test_target, randomLength)


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

   if randomLength:
      [MSE_test_linear_arr, _, _,knet_out, _, _] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results, randomLength, MaskOnState, test_lengthMask)
   else:
      [MSE_test_linear_arr, _, _, knet_out, _, _] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results, randomLength, MaskOnState)


   MSE_KF_db_arr = 10*torch.log10(MSE_KF_linear_arr)
   MSE_test_db_arr = 10*torch.log10(MSE_test_linear_arr)

   plt.plot(range(N_T), MSE_obs_dB_arr, 'g', label= "Initial Loss")
   plt.plot(range(N_T), MSE_KF_db_arr, 'r', label= "KF Test Loss")
   plt.plot(range(N_T), MSE_test_db_arr, 'b', label= "KNet Test Loss")
   plt.xlabel("Sequences")
   plt.ylabel("MSE Loss")
   plt.grid(True)
   plt.show()
   plt.close()


   plot_testset(test_input, test_target, N_T)
   knet_out = knet_out.detach_().numpy()
   plot_results(test_target, KF_out, knet_out, N_T)





for i in range(100, 1000+1, 100):
   print("Test con sequenze di lunghezza nell'intervallo [", i, ", ", 1000, "]")
   test(i, 1000)
   print("\n")





















