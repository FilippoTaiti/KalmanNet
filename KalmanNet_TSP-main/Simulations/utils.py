"""
The file contains utility functions for the simulations.
"""

import torch

def DataGen(SysModel_data, fileName, N_E, N_T, T, T_test, N_CV, m2x_0):

    ##################################
    ### Generate Training Sequence ###
    ##################################
    SysModel_data.GenerateBatch(N_E, T, m2x_0)
    train_input = SysModel_data.Input
    train_target = SysModel_data.Target
    ### init conditions ###
    train_init = SysModel_data.m1x_0_batch #size: N_E x m x 1
    ### length mask ###

    ####################################
    ### Generate Validation Sequence ###
    ####################################
    SysModel_data.GenerateBatch(N_CV, T, m2x_0)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target
    cv_init = SysModel_data.m1x_0_batch #size: N_CV x m x 1

    ##############################
    ### Generate Test Sequence ###
    ##############################
    SysModel_data.GenerateBatch(N_T, T_test, m2x_0)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target
    test_init = SysModel_data.m1x_0_batch #size: N_T x m x 1


    #################
    ### Save Data ###
    #################
    torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init], fileName)
    


