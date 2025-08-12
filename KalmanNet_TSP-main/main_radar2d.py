import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt

from Simulations.Linear_sysmdl import SystemModel
from Simulations.utils import DataGen
from Simulations.Radar_2d.parameters import F, H, Q, R, m

from Filters.KalmanFilter_test import KFTest

from KNet.KalmanNet_nn import KalmanNetNN

from Pipelines.Pipeline_EKF import Pipeline_EKF 

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
N_E = 1000  # Numero di sequenze del Training Set
N_CV = 100  # Numero di sequenze del Validation Set
N_T = 100   # Numero di sequenze del Test Set
# init condition


# sequence length
T = 20
T_test = 1000

### training parameters ##################################################
N_steps = 1000 # Numero epoche
N_batch = 100 # Dimensione del singolo batch
lr = 1e-3
wd = 1e-4

use_cuda = False

device = torch.device('cpu')
print("Using CPU")

m2_0 = 0 * torch.eye(m)

### True model ##################################################
sys_model = SystemModel(F, Q, H, R, T, T_test)
#print("State Evolution Matrix:",F)
#print("Observation Matrix:",H)

###################################
### Data Loader (Generate Data) ###
###################################
dataFolderName = 'Simulations/Radar_2d/data' + '/'
dataFileName = '2x2_rq020_T100.pt'
print("Start Data Gen")
DataGen(sys_model, dataFolderName + dataFileName, N_E, N_T, T, T_test, N_CV, m2_0)
print("Data Load")

[train_input, train_target, cv_input, cv_target, test_input, test_target,_,_,_] = torch.load(dataFolderName + dataFileName, map_location=device)

print("trainset size:",train_target.size())
print("cvset size:",cv_target.size())
print("testset size:",test_target.size())

########################################
### Evaluate Observation Noise Floor ###
########################################
loss_obs = nn.MSELoss(reduction='mean')
MSE_obs_linear_arr = torch.empty(N_T)# MSE [Linear]
for i in range(N_T):
   mask = torch.tensor([True, False, True, False])
   MSE_obs_linear_arr[i] = loss_obs(test_input[i], test_target[i][mask]).item()
MSE_obs_linear_avg = torch.mean(MSE_obs_linear_arr)
MSE_obs_dB_avg = 10 * torch.log10(MSE_obs_linear_avg)

# Standard deviation
MSE_obs_linear_std = torch.std(MSE_obs_linear_arr, unbiased=True)

# Confidence interval
obs_std_dB = 10 * torch.log10(MSE_obs_linear_std + MSE_obs_linear_avg) - MSE_obs_dB_avg

print("Observation Noise Floor - MSE LOSS:", MSE_obs_dB_avg, "[dB]")
print("Observation Noise Floor - STD:", obs_std_dB, "[dB]")

##############################
### Evaluate Kalman Filter ###
##############################
print("Evaluate Kalman Filter True")
[MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out] = KFTest(sys_model, N_T, T_test, test_input, test_target)


##########################
### KalmanNet Pipeline ###
##########################

### KalmanNet with full info ##########################################################################################
# Build Neural Network
print("KalmanNet with full model info")
KalmanNet_model = KalmanNetNN()
KalmanNet_model.NNBuild(sys_model)
print("Number of trainable parameters for KalmanNet:",sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))
## Train Neural Network
KalmanNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
KalmanNet_Pipeline.setssModel(sys_model)
KalmanNet_Pipeline.setModel(KalmanNet_model)
KalmanNet_Pipeline.setTrainingParams(N_steps, N_batch, lr, wd, 0.3)

[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = KalmanNet_Pipeline.NNTrain(sys_model, cv_input, cv_target, train_input, train_target, path_results)
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg,knet_out,RunTime] = KalmanNet_Pipeline.NNTest(sys_model, test_input, test_target, path_results)
KalmanNet_Pipeline.save()


plt.figure()
plt.plot(range(N_steps), MSE_train_dB_epoch, 'b', label= "Training Loss")
plt.plot(range(N_steps), MSE_cv_dB_epoch, 'g', label = "Cross Validation Loss")
plt.xlabel("Training Epoch")
plt.ylabel("MSE Loss")
plt.show()
plt.close()


MSE_KF_db_arr = 10*torch.log10(MSE_KF_linear_arr)
MSE_test_db_arr = 10*torch.log10(MSE_test_linear_arr)

plt.figure()
plt.plot(range(N_T), MSE_KF_db_arr, 'r', label= "KF Test Loss")
plt.plot(range(N_T), MSE_test_db_arr, 'b', label= "KN Test Loss")
plt.xlabel("Sequences")
plt.ylabel("MSE Loss")
plt.show()
plt.close()