"""
The file contains utility functions for the simulations.
"""
import random

import torch
import matplotlib.pyplot as plt


def DataGen(SysModel_data, fileName, N_E, N_CV, N_T, T, T_min, T_test, randomLength):

    ##################################
    ### Generate Training Sequence ###
    ##################################


    SysModel_data.GenerateBatch(N_E, T, T_min, randomLength)
    train_input = SysModel_data.Input
    train_target = SysModel_data.Target
    ### init conditions ###
    train_init = SysModel_data.m1x_0_batch #size: N_E x m x 1
    ### length mask ###
    if randomLength:
        train_lengthMask = SysModel_data.lengthMask

    ####################################
    ### Generate Validation Sequence ###
    ####################################

    SysModel_data.GenerateBatch(N_CV, T, T_min, randomLength)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target
    cv_init = SysModel_data.m1x_0_batch #size: N_CV x m x 1
    if randomLength:
        cv_lengthMask = SysModel_data.lengthMask

    ##############################
    ### Generate Test Sequence ###
    ##############################

    SysModel_data.GenerateBatch(N_T, T_test, T_min, randomLength)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target
    test_init = SysModel_data.m1x_0_batch #size: N_T x m x 1
    if randomLength:
        test_lengthMask = SysModel_data.lengthMask


    #################
    ### Save Data ###
    #################
    if randomLength:
        torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init, train_lengthMask, cv_lengthMask, test_lengthMask], fileName)
    else:
        torch.save(
            [train_input, train_target, cv_input, cv_target, test_input, test_target, train_init, cv_init, test_init], fileName)






def DataTestGen(SysModel_data, fileName, N_T, T_min, T_test, randomLength):
    ##############################
    ### Generate Test Sequence ###
    ##############################

    SysModel_data.GenerateBatch(N_T, T_test, T_min, True, randomLength)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target
    test_init = SysModel_data.m1x_0_batch  # size: N_T x m x 1
    if randomLength:
        test_lengthMask = SysModel_data.lengthMask

    #################
    ### Save Data ###
    #################
    if randomLength:
        torch.save([test_input, test_target, test_init, test_lengthMask], fileName)
    else:
        torch.save([test_input, test_target, test_init], fileName)
    

def plot_results(trajectories, KF_trajectories, KNet_trajectories, N_T, randomLength, lengthMask=None):

    if randomLength:
        for i in range(N_T):
            #print(lengthMask.shape)
            mask = lengthMask[i]
            #print(mask.shape)
            #print(mask, "kldeffk", len(mask))
            traj = trajectories[i, [0, 2], :][:, mask]
            KF_traj = KF_trajectories[i, [0, 2], :][:, mask]
            KNet_traj = KNet_trajectories[i, [0, 2], :][:, mask]

            traj_x = traj[0]
            # print("Traj x: ", traj_x)
            traj_y = traj[1]
            # print("Traj y: ", traj_y)

            KF_traj_x = KF_traj[0]
            # print("KF Traj x: ", KF_traj_x)
            KF_traj_y = KF_traj[1]
            # print("KF Traj y: ", KF_traj_y)

            KNet_traj_x = KNet_traj[0]
            # print("KNet Traj x: ", KN_traj_x)
            KNet_traj_y = KNet_traj[1]
            # print("KNet Traj y: ", KN_traj_y)

            # plt.figure()
            figure, axis = plt.subplots(3, 1, figsize=(15, 15))
            axis[0].plot(list(traj_x), list(traj_y), lw=3, color='green')
            axis[0].set_title('Target Trajectory', fontsize=10)

            axis[1].plot(list(KF_traj_x), list(KF_traj_y), lw=3, color='red')
            axis[1].set_title('KF Trajectory', fontsize=10)

            axis[2].plot(list(KNet_traj_x), list(KNet_traj_y), lw=3, color='blue')
            axis[2].set_title('KNet Trajectory', fontsize=10)

            plt.show()
            plt.close(figure)

    else:
        for i in range(N_T):
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
            figure, axis = plt.subplots(3, 1, figsize=(15, 15))
            axis[0].plot(list(traj_x), list(traj_y), lw=3, color = 'green')
            axis[0].set_title('Target Trajectory', fontsize=10)

            axis[1].plot(list(KF_traj_x), list(KF_traj_y), lw=3, color = 'red')
            axis[1].set_title('KF Trajectory', fontsize=10)

            axis[2].plot(list(KNet_traj_x), list(KNet_traj_y), lw=3, color = 'blue')
            axis[2].set_title('KNet Trajectory', fontsize=10)


            plt.show()
            plt.close(figure)



