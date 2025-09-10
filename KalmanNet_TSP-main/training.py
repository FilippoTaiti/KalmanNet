import random

import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt


from Simulations.Linear_sysmdl import SystemModel, plot_testset
from Simulations.utils import DataGen, plot_results
from Simulations.Radar_2d.parameters import F, H, Q, R_kf, R_Knet, m

from Filters.KalmanFilter_test import KFTest

from KNet.KalmanNet_nn import KalmanNetNN

from Pipelines.Pipeline import Pipeline_EKF


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

####################
### Design Model ###
####################


### dataset parameters ##################################################
N_E = 100000  # Numero di sequenze del Training Set
N_CV = 20000  # Numero di sequenze del Validation Set
# init condition


# sequence length
T = 50 #Lunghezza delle sequenze di training e di validation
T_test = 50 #lunghezza delle sequenze di test

T_min = 10
#T_max = 50

CompositionLoss = True

### training parameters ##################################################
N_steps = 1000 # Numero epoche
N_batch = 100 # Dimensione del singolo batch
lr = 1e-3
wd = 1e-4

use_cuda = False

device = torch.device('cpu')
print("Using CPU")

m1_0 = torch.zeros(m, 1)
m2_0 = 1 * torch.eye(m)


### True model ##################################################
sys_model = SystemModel(F, Q, H, R_kf, R_Knet, T_test, T)
sys_model.InitSequence(m1_0, m2_0)

###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Simulations/Radar_2d/data' + '/'
dataFileName = '2x2_rq020_T100.pt'
print("Start Data Gen")
DataGen(sys_model, dataFolderName + dataFileName, N_E, N_CV, T, T_min)
print("Data Load")
[train_input, train_target, cv_input, cv_target,_,_] = torch.load(dataFolderName + dataFileName, map_location=device)


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

[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results, CompositionLoss)
#[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, knet_out, RunTime,
#KalmanGainKN, _] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)


KalmanNet_Pipeline.save()


plt.figure()
plt.plot(range(N_steps), MSE_train_dB_epoch, 'b', label= "Training Loss")
plt.plot(range(N_steps), MSE_cv_dB_epoch, 'g', label = "Cross Validation Loss")
plt.xlabel("Training Epoch")
plt.ylabel("MSE Loss")
plt.show()
plt.close()





