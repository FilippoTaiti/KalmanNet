import torch
import torch.nn as nn
import time
from Filters.Linear_KF import KalmanFilter

def KFTest(SysModel, N_T, T_test, test_input, test_target, randomLength, test_lengthMask=None):

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_KF_linear_arr = torch.zeros(N_T)
    # allocate memory for KF output
    start = time.time()

    KF = KalmanFilter(SysModel)
    # Init and Forward Computation 

    KF.Init_batched_sequence(SysModel.m1x_0.view(1,SysModel.m,1).expand(N_T,-1,-1), SysModel.m2x_0.view(1,SysModel.m,SysModel.m).expand(N_T,-1,-1))
    KF.GenerateBatch(test_input)
    
    end = time.time()
    t = end - start
    KF_out = KF.x
    # MSE loss
    for j in range(N_T):# cannot use batch due to different length and std computation
        if randomLength:
            MSE_KF_linear_arr[j] = loss_fn(KF.x[j, :, test_lengthMask[j]], test_target[j, :, test_lengthMask[j]]).item()
        else:
            MSE_KF_linear_arr[j] = loss_fn(KF.x[j, :, :], test_target[j, :, :]).item()

    MSE_KF_linear_avg = torch.mean(MSE_KF_linear_arr)
    MSE_KF_dB_avg = 10 * torch.log10(MSE_KF_linear_avg)

    # Standard deviation
    MSE_KF_linear_std = torch.std(MSE_KF_linear_arr, unbiased=True)

    # Confidence interval
    KF_std_dB = 10 * torch.log10(MSE_KF_linear_std + MSE_KF_linear_avg) - MSE_KF_dB_avg
    #KF_std_dB = 10 * torch.log10(MSE_KF_linear_std)

    KalmanGain = KF.KG

    print("Kalman Filter - MSE LOSS:", MSE_KF_dB_avg, "[dB]")
    print("Kalman Filter - STD:", KF_std_dB, "[dB]")
    # Print Run Time
    print("Inference Time:", t)
    return [MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg, KF_out, KalmanGain]



