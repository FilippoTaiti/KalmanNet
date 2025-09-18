"""
The file contains utility functions for the simulations.
"""
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from .Radar_2d.parameters import R_kf


def DataGen(SysModel_data, fileName, N_E, N_CV, T, T_min):

    ##################################
    ### Generate Training Sequence ###
    ##################################


    SysModel_data.GenerateBatch(N_E, T, T_min, False)
    train_input = SysModel_data.Input
    train_target = SysModel_data.Target
    ### init conditions ###
    train_init = SysModel_data.m1x_0_batch #size: N_E x m x 1
    ### length mask ###


    ####################################
    ### Generate Validation Sequence ###
    ####################################

    SysModel_data.GenerateBatch(N_CV, T, T_min, False)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target
    cv_init = SysModel_data.m1x_0_batch #size: N_CV x m x 1


    #################
    ### Save Data ###
    #################
    torch.save([train_input, train_target, cv_input, cv_target, train_init, cv_init], fileName)






def DataTestGen(SysModel_data, fileName, N_T, T_min, T_test, randomLength):
    ##############################
    ### Generate Test Sequence ###
    ##############################

    SysModel_data.GenerateBatch(N_T, T_test, T_min, True, randomLength)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target
    test_init = SysModel_data.m1x_0_batch  # size: N_T x m x 1

    #################
    ### Save Data ###
    #################
    torch.save([test_input, test_target, test_init], fileName)
    

def plot_results(trajectories, KF_trajectories, KNet_trajectories, indexes):
    KF_trajectories = KF_trajectories.detach().numpy()
    KNet_trajectories = KNet_trajectories.detach().numpy()
    for i in indexes:
        traj = trajectories[i, [0, 2], :]
        KF_traj = KF_trajectories[i, [0, 2], :]
        KNet_traj = KNet_trajectories[i, [0, 2], :]
        traj_x = traj[0, :]
        #print("Traj x: ", traj_x)
        traj_y = traj[1, :]
        #print("Traj y: ", traj_y)
        KF_traj_x = KF_traj[0, :]
        #print("KF Traj x: ", KF_traj_x)
        KF_traj_y = KF_traj[1, :]
        #print("KF Traj y: ", KF_traj_y)
        KNet_traj_x = KNet_traj[0, :]
        #print("KNet Traj x: ", KN_traj_x)
        KNet_traj_y = KNet_traj[1, :]
        #print("KNet Traj y: ", KN_traj_y)
        #plt.figure()
        figure, axis = plt.subplots(1, 1, figsize=(15, 15))
        plt.plot(traj_x, traj_y, lw=3, color = 'green')
        plt.plot(KF_traj_x, KF_traj_y, lw=3, color = 'red')
        plt.plot(KNet_traj_x, KNet_traj_y, lw=3, color = 'blue')
        plt.title('Trajectories with σ ='+ f'{torch.sqrt(R_kf[0,0]).item():.2f}')
        plt.legend(['Target Trajectory', 'KF Trajectory', 'KNet Trajectory'], loc='upper right')
        plt.show()
        plt.close('all')



def plotAverageTrajectories(squaredErrorKF, squaredErrorKNet):
    squaredErrorKF = squaredErrorKF.detach().numpy()
    squaredErrorKNet = squaredErrorKNet.detach().numpy()

    avgSquaredErrorKF = np.mean(squaredErrorKF, axis=0)
    avgSquaredErrorKNet = np.mean(squaredErrorKNet, axis=0)

    plt.plot(avgSquaredErrorKF, lw=3, color='red')
    plt.plot(avgSquaredErrorKNet, lw=3, color='blue')
    plt.title('Average Squared Error with σ ='+ f'{torch.sqrt(R_kf[0,0]).item():.2f}')
    plt.xlabel('Time step')
    plt.ylabel('Squared Error')
    plt.legend(['KF', 'KNet'], loc='best')
    plt.grid(True)
    plt.show()
    plt.close('all')



def plotSquaredError(squaredErrorKF, squaredErrorKNet, indexes):
    squaredErrorKF = squaredErrorKF.detach().numpy()
    squaredErrorKNet = squaredErrorKNet.detach().numpy()
    for i in indexes:
        plt.plot(squaredErrorKF[i], lw=3, color='red')
        plt.plot(squaredErrorKNet[i], lw=3, color='blue')
        plt.title('Squared Error of KF and KNet with σ ='+ f'{torch.sqrt(R_kf[0,0]).item():.2f}')
        plt.xlabel('Time step')
        plt.ylabel('Squared Error')
        plt.legend(['KF', 'KNet'], loc='best')
        plt.show()
        plt.close('all')

def plotBoxPlot(MSE_obs, MSE_KF, MSE_KNet):
    plt.figure(figsize=(15, 5))
    boxPlot = plt.boxplot([MSE_obs, MSE_KF, MSE_KNet], labels=["Observation", "Kalman Filter", "KalmanNet"], patch_artist=True)
    colors = ['green', 'red', 'blue']
    for i, color in zip(boxPlot['boxes'], colors):
        i.set_facecolor(color)

    plt.legend([boxPlot['boxes'][0], boxPlot['boxes'][1], boxPlot['boxes'][2]], ["Observation", "Kalman Filter", "KalmanNet"], loc='upper left')
    plt.title('Mean Squared Error with σ ='+ f'{torch.sqrt(R_kf[0,0]).item():.2f}')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.close('all')