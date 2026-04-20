import random
import numpy as np

# att_num = 3  # 攻击者数量
# dff_num = 8  # 防御者数量
# sim_time = 0.1  # 单步时间
# total_steps = 1000  # 总步数
#
# '''初始化场景双方的位置及姿态上下界'''
# target_area_x = 0.0
# target_area_y = 0.0
# target_area_rds = 2.0
# att_x_left = 25
# att_x_right = 30
# att_y_up = 6
# att_y_down = -6
# dff_x_left = -5
# dff_x_right = 5
# dff_y_up = 4
# dff_y_down = -4
# phi_random_up = 0.5
# phi_random_down = 1.5
# zeta_random_up = 1.0
# zeta_random_down = 2.0
# beta_random_up = 0.45
# beta_random_down = 0.6
# battleSize = np.array([40.0, 40.0])
# maxTime = 200.0
#
# '''动力学模型参数及策略参数'''
# # 进攻方
# max_va = 2.5
# max_ua = 20.0
# rAttswm = 2.5
# r_ad_s = 0.9 # 1.8
# r_ad_a = 10.0
# r_aa_s = 1.0
# r_aa_a = 2.0
# k_ad = 10.0
# k_ap = 6.666666
# k_aa = 4.0
# rCapture = 1.0
#
# # 防守方
# max_vd = 2.0
# max_ud = 15.0
# rString = 3.0
# k_p = 1.0
# k_av = 2.0
# k_v2u = 2.0
# r_safe = 0.2
# r_avoid = r_safe + 1.0
# damp_def = max_ud / max_vd
# k_v = 2.0
# rSafe = 0.4
#
# '''场景运行防守方String_net的位置及动作上下界'''
# alpha_bound = np.array([[-battleSize[0], battleSize[0]],
#                        [-battleSize[0], battleSize[0]],
#                        [-np.pi, np.pi],
#                        [rSafe*2, rString],
#                        [0.25*np.pi, 2*np.pi]], dtype=np.float32)
#
# action_bound = np.array([[-1/2, 1/2],
#                          [-1/2, 1/2],
#                          [-1/9, 1/9],
#                          [-1/5, 1/5],
#                          [-1/3, 1/3]], dtype=np.float32) * max_vd
#
# # action_bound_local = np.array([[-0.6, 0.6],
# #                          [-0.6, 0.6],
# #                          # [-1/10, 1/10],
# # [-1/1000, 1/1000],
# #                          # [-1/3, 1/3],
# #                          # [-1/2, 1/2]], dtype=np.float32) * max_vd
# #                          [0, 0.001],
# #                          [0, 0.001]], dtype=np.float32) * max_vd
#
# action_bound_local = np.array([[-0.6, 0.6],
#                          [-0.6, 0.6]], dtype=np.float32) * max_vd
#
# # cos_phi_da, sin_phi_da, cos_phi_ap, sin_phi_ap,
# # cos_phi_a_vel, sin_phi_a_vel, cos_phi_beta, sin_phi_beta,
# # r_da, r_ap, r_asw, r_str, r_a_vel
# # state_bound = np.array([[-1, 1],  # cos_phi_da
# #                         [-1, 1],  # sin_phi_da
# #                         [-1, 1],  # cos_phi_ap
# #                         [-1, 1],  # sin_phi_ap
# #                         [-1, 1],  # cos_phi_a_vel
# #                         [-1, 1],  # sin_phi_a_vel
# #                         [-1, 1],  # cos_phi_beta
# #                         [-1, 1],  # sin_phi_beta
# #                         [0, battleSize[0]],  # r_da
# #                         [0, battleSize[0]],  # r_ap
# #                         [0, att_num*2],  # r_asw
# #                         [0, rString],  # r_str
# #                         [0, max_va],   # r_a_vel
# #                         [0, maxTime]  # running_time
# #                         ])
# '''(cos_phi, sin_phi, cos_beta, sin_beta, zeta,
# r_da, cos_da, sin_da, r_ap, cos_ap, sin_ap,
# v_att[0], v_att[1], att_rds, simulation_time)'''
# state_bound = np.array([[-1, 1],  # cos_phi
#                         [-1, 1],  # sin_phi
#                         [-1, 1],  # cos_beta
#                         [-1, 1],  # sin_beta
#                         [0, rCapture*2],  # zeta
#                         [0, battleSize[0]],  # r_da
#                         [-1, 1],  # cos_da
#                         [-1, 1],  # sin_da
#                         [0, 10],  # r_ap
#                         [-1, 1],  # cos_ap
#                         [-1, 1],  # sin_ap
#                         [-max_va, max_va],  # v_att[0]
#                         [-max_va, max_va],  # v_att[0]
#                         [0, rCapture * att_num * 2],  # att_rds
#                         [0, maxTime]])  # running_time
#
# '''(cos_beta, sin_beta, zeta,
# r_da, cos_da, sin_da, r_ap, cos_ap, sin_ap,
# r_v_att, cos_v_att, sin_v_att, att_rds, simulation_time)'''
# # state_bound_local = np.array([[-1, 1],  # cos_beta
# #                         [-1, 1],  # sin_beta
# #                         [0, rCapture*2],  # zeta
# #                         [0, 7.5],  # r_da
# #                         [-1, 1],  # cos_da
# #                         [-1, 1],  # sin_da
# #                         [0, 10],  # r_ap
# #                         [-1, 1],  # cos_ap
# #                         [-1, 1],  # sin_ap
# #                         [0, max_va+max_vd],  # r_v_att
# #                         [-1, 1],  # cos_v_att
# #                         [-1, 1],  # sin_v_att
# #                         [0, rCapture * att_num * 2],  # att_rds
# #                         ])  # running_time
# state_bound_local = np.array([
#                         # [-1, 1],  # cos_beta
#                         # [-1, 1],  # sin_beta
#                         # [0, rCapture*2],  # zeta
#                         [0, 15.0],  # r_ad
#                         [-1, 1],  # cos_ad
#                         [-1, 1],  # sin_ad
#                         [0, 20.0],  # r_dp
#                         [-1, 1],  # cos_dp
#                         [-1, 1],  # sin_dp
#                         # [0, 20.0],  # r_ap
#                         # [-1, 1],  # cos_ap
#                         # [-1, 1],  # sin_ap
#                         # [0, max_va+max_vd],  # r_v_att
#                         # [-1, 1],  # cos_v_att
#                         # [-1, 1],  # sin_v_att
#                         # [0.5, rCapture * att_num * 2],  # att_rds
#                         ])  # running_time
#
# '''算法超参数'''
# POP = 5  # population size, default 20, 5
# MAX_GEN = 8  # maximum number of generations, default 10, 5
# W = 0.9  # inertia weight
# C1 = 0.1  # cognitive constant, the higher the value, the more the particle will look for the personal best
# C2 = 0.1  # social constant, the higher the value, the more the particle will look for the global best
# PREDICTION_HORIZON = 5  # prediction horizon, default 10
# DECISION_HORIZON = 2  # decision horizon, default 2
# INIT_RAND_RATE = 0.5  # random rate used in action sequence initialization
# INIT_RAND_SIGMA = 0.1  # random sigma used in action sequence initialization
# TRAIN_PER_STEPS = 10  # training frequency
# Q_MEMORY_CAPACITY = int(25000)  # maximum size of critic replay buffer
# Q_BATCH_SIZE = 128  # update batch size of critic 1024
# Q_START_TRAIN = 5*Q_BATCH_SIZE  # start training after Q_START_TRAIN steps
# A_MEMORY_CAPACITY = int(1E5)  # maximum size of actor replay buffer
# A_BATCH_SIZE = 64  # update batch size of actor 1024
# A_START_TRAIN = 3*A_BATCH_SIZE  # start training after A_START_TRAIN steps
# M_MEMORY_CAPACITY = int(1E5)  # maximum size of model replay buffer
#
# TARGET_UPDATE_INTERVAL = 10  # target network update interval
#
# MAX_EPISODE = int(1E4)  # maximum episode
# MAX_UPDATE_PER_STEP = 10  # maximum update times per step 4
#
# GAMMA = 0.95  # reward discount
# TAU = 0.01  # soft replacement
#
# s_dim = 7# 17 - 7
# a_dim = 2# 5
# # hidden_nodes = [32, 64, 128, 256, 128, 64, 32]  # 18, 18
#
# hidden_nodes = [128, 128, 128]  # 18, 18
#
# # k_dis = 5.0E-2
# # k_sur = 1.0E-2 * 180 / np.pi
# # k_ali = 1.0E-2 * 180 / np.pi
# # ctr_matrix = np.eye(5) * 1.0E-3
# # k_act = 5.0E-3
#
# k_dis = 5.0E-1
# k_sur = 4.0E-1 * 180 / np.pi
# k_ali = 2.0E-1 * 180 / np.pi
# ctr_matrix = np.eye(5) * 1.0 * 1.0E-2
# k_act = 5.0E-2
#
# import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode = '30'
sim_time = 0.1  # 单步时间
total_steps = 1000  # 总步数

if mode == 'ablation':
    att_num = 3  # 攻击者数量
    dff_num = 8  # 防御者数量
    goal_area_x = 35.0
    goal_area_y = 7.5
    obstacles = []
    att_x_left = 25
    att_x_right = 30
    att_y_up = -5
    att_y_down = -8
    xlim_left = -5
    xlim_right = 40
    ylim_up = 10
    ylim_down = -10
    dff_pos = np.array([[4.5484476, 1.620812],
                        [3.5000215, 1.796988],
                        [1.9243426, 2.634684],
                        [1.3889976, 3.0474734],
                        [0.11116849, 0.66243345],
                        [-1.360835, - 0.46587336],
                        [0.85418916, - 1.6181339],
                        [-0.77942044, - 4.1287184]], dtype=np.float32)
    att_pos = np.array([[25.363508, - 7.8416505],
                        [26.066116, - 5.9335065],
                        [25.80009, - 6.6538234]], dtype=np.float32)

if mode == '50vs20':
    att_num = 20
    dff_num = 50
    goal_area_x = 35.0 * 2.0
    goal_area_y = 7.5 * 2.0
    obstacles = [[18 * 2.0, 8.0 * 2.0, 1.5 * 2.0], [24 * 2.0, 2.0 * 2.0, 2.0 * 2.0], [70, -10, 1.5 * 2.0]]
    att_x_left = 25 * 2.0
    att_x_right = 30 * 2.0
    att_y_up = -5 * 2.0
    att_y_down = -8 * 2.0
    xlim_left = -5 * 2.0
    xlim_right = 40 * 2.0
    ylim_up = 10 * 2.0
    ylim_down = -10 * 2.0
elif mode == '40vs15':
    att_num = 15
    dff_num = 40
    goal_area_x = -35.0 * 2.0
    goal_area_y = -7.5 * 2.0
    obstacles = [[-22 * 2.0, -3 * 2.0, 2.0 * 2.0], [-32 * 2.0, 4 * 2.0, 2.0 * 2.0], [-14 * 2.0, -6.0 * 2.0, 1.0 * 2.0]]
    att_x_left = -25 * 2.0
    att_x_right = -30 * 2.0
    att_y_up = 8 * 2.0
    att_y_down = 5 * 2.0
    xlim_left = -40 * 2.0
    xlim_right = 5 * 2.0
    ylim_up = 10 * 2.0
    ylim_down = -10 * 2.0
elif mode == '30vs15':
    att_num = 15
    dff_num = 30
    goal_area_x = -35.0 * 2.0
    goal_area_y = 0 * 2.0
    obstacles = [[-25 * 2.0, 2.0 * 2.0, 1.8 * 2.0], [-22 * 2.0, -7 * 2.0, 2.0 * 2.0]]
    att_x_left = -15 * 2.0
    att_x_right = -20 * 2.0
    att_y_up = 0 * 2.0
    att_y_down = -2 * 2.0
    xlim_left = -40 * 2.0
    xlim_right = 5 * 2.0
    ylim_up = 10 * 2.0
    ylim_down = -10 * 2.0
elif mode == '15vs10':
    att_num = 10
    dff_num = 15
    goal_area_x = -35 * 2.0
    goal_area_y = 0 * 2.0
    obstacles = [[-18 * 2.0, 0 * 2.0, 1.5 * 2.0], [-29 * 2.0, 3 * 2.0, 1.0 * 2.0], [-14 * 2.0, 11 * 2.0, 1.0 * 2.0]]
    att_x_left = -32 * 2.0
    att_x_right = -35 * 2.0
    att_y_up = 10 * 2.0
    att_y_down = 7 * 2.0
    xlim_left = -40 * 2.0
    xlim_right = 5 * 2.0
    ylim_up = 15 * 2.0
    ylim_down = -5 * 2.0
elif mode == 'scalability_multiturn':
    att_num = 10  # 攻击者数量
    dff_num = 15  # 防御者数量
    obstacles = []

    theta = random.uniform(0, np.pi * 2.0)
    distance_1 = random.uniform(0.7, 1.0) * 35
    distance_2 = distance_1 + 5.0
    att_x_left = distance_1 * np.cos(theta)
    att_x_right = distance_2 * np.cos(theta)
    att_y_up = distance_1 * np.sin(theta)
    att_y_down = distance_2 * np.sin(theta)
    goal_area_x = att_x_left * 2.0
    goal_area_y = att_y_up*2

    xlim_left = -40*2
    xlim_right = 5*2
    ylim_up = 10*2
    ylim_down = -10*2

if mode == 'inferior_attacker':
    att_num = 3
    dff_num = 9
    goal_area_x = 35.0
    goal_area_y = 7.5
    obstacles = [[29, 2, 2.0]]
    att_x_left = 25
    att_x_right = 30
    att_y_up = -5
    att_y_down = -8
    xlim_left = -5
    xlim_right = 40
    ylim_up = 10
    ylim_down = -10
    max_va = 1.5
    max_ua = 10.0
    dff_pos = np.array([[ 1.01420500, -4.6814036],
                        [-2.05101200, -1.8025767],
                        [ 1.08125020,  5.2556180],
                        [-4.17848870, -3.2104880],
                        [ 2.48013020,  1.7321734],
                        [-0.40064263,  0.9800772],
                        [-4.16695260,  2.2714858],
                        [-1.16029050, -2.3743868],
                        [-0.16029050, -3.3743868],
                        ], dtype=np.float32)
    att_pos = np.array([[26.62792200, -8.1200330],
                        [28.63077500, -8.5453420],
                        [26.74894100, -7.0210620]
                        ], dtype=np.float32)
if mode == 'superior_attacker':
    att_num = 3
    dff_num = 9
    goal_area_x = 35.0
    goal_area_y = 7.5
    obstacles = [[29, 2, 2.0]]
    att_x_left = 25
    att_x_right = 30
    att_y_up = -5
    att_y_down = -8
    xlim_left = -5
    xlim_right = 40
    ylim_up = 10
    ylim_down = -10
    max_va = 2.5
    max_ua = 20.0
    dff_pos = np.array([[ 1.01420500, -4.6814036],
                        [-2.05101200, -1.8025767],
                        [ 1.08125020,  5.2556180],
                        [-4.17848870, -3.2104880],
                        [ 2.48013020,  1.7321734],
                        [-0.40064263,  0.9800772],
                        [-4.16695260,  2.2714858],
                        [-1.16029050, -2.3743868],
                        [-0.16029050, -3.3743868],
                        ], dtype=np.float32)
    att_pos = np.array([[26.62792200, -8.1200330],
                        [28.63077500, -8.5453420],
                        [26.74894100, -7.0210620]
                        ], dtype=np.float32)

if mode == '15':
    att_num = 3  # 攻击者数量
    dff_num = 9  # 防御者数量
    obstacles = []

    theta = random.uniform(0, np.pi * 2.0)
    distance_1 = random.uniform(0.7, 1.0) * 35 / 2
    distance_2 = (distance_1 + 5.0) / 2
    att_x_left = distance_1 * np.cos(theta)
    att_x_right = distance_2 * np.cos(theta)
    att_y_up = distance_1 * np.sin(theta)
    att_y_down = distance_2 * np.sin(theta)
    goal_area_x = att_x_left * 2.0
    goal_area_y = att_y_up * 2.0

    xlim_left = -40
    xlim_right = 5
    ylim_up = 10
    ylim_down = -10
    max_va = 2.5
    max_ua = 20.0
elif mode == '30':
    att_num = 3  # 攻击者数量
    dff_num = 9  # 防御者数量
    obstacles = []

    theta = random.uniform(0, np.pi * 2.0)
    distance_1 = random.uniform(0.7, 1.0) * 35 / 2
    distance_2 = (distance_1 + 5.0) / 2
    att_x_left = distance_1 * np.cos(theta)
    att_x_right = distance_2 * np.cos(theta)
    att_y_up = distance_1 * np.sin(theta)
    att_y_down = distance_2 * np.sin(theta)
    goal_area_x = att_x_left * 2.0
    goal_area_y = att_y_up * 2.0

    xlim_left = -40
    xlim_right = 5
    ylim_up = 10
    ylim_down = -10
    max_va = 1.5
    max_ua = 10.0

#####################################
target_area_x = 0.0
target_area_y = 0.0
target_area_rds = 2.0
target_area = np.array([target_area_x, target_area_y])
dff_x_left = -5
dff_x_right = 5
dff_y_up = 4
dff_y_down = -4

phi_random_up = -10 / 180 * np.pi
phi_random_down = 10 / 180 * np.pi
zeta_random_up = 1.0
zeta_random_down = 2.0
beta_random_up = 0.45
beta_random_down = 0.6
battleSize = np.array([80.0, 80.0])
maxTime = 200.0

'''动力学模型参数及策略参数'''
# 进攻方
# max_va = 2.5
# max_ua = 20.0
rAttswm = 2.5
r_ad_s_down = 0.9
r_ad_s_up = 1.8
r_ad_a_down = 7.0
r_ad_a_up = 14.0
r_aa_s_down = 1.0
r_aa_s_up = 2.0
r_aa_a = 2.0
k_ad = 10.0
k_ap = 6.666666
k_aa = 4.0
rCapture = 1.0

# 防守方
max_vd = 2.0
max_ud = 15.0
rString = 3.0
k_p = 1.0
k_av = 2.0
k_v2u = 2.0
r_safe = 0.2
r_avoid = 2 * r_safe #  + 1.0
damp_def = max_ud / max_vd
k_v = 2.0
rSafe = 0.4

'''场景运行防守方String_net的位置及动作上下界'''
state_bound = np.array([[0, 15],  # r_ad
                        [-1, 1],  # cos_ad
                        [-1, 1],  # sin_ad
                        # [0, 20],  # r_ap
                        # [-1, 1],  # cos_ad
                        # [-1, 1],  # sin_ad
                        [0, 20],  # r_dp
                        [-1, 1],  # cos_dp
                        [-1, 1],  # sin_dp
                        [rSafe*2, rString],   # zeta
                        [-1, 1],  # cos_beta
                        [-1, 1],  # sin_beta
                        [0.5, 10],  # att_rds
                        [0, max_vd+max_va],   # r_v_att
                        [-1, 1],  # cos_v_att
                        [-1, 1],  # sin_v_att
                        ])

alpha_bound = np.array([[-battleSize[0], battleSize[0]],
                       [-battleSize[0], battleSize[0]],
                       [-np.pi, np.pi],
                       [rSafe*2, rString],
                       [0.25*np.pi, 2*np.pi]], dtype=np.float32)

action_bound = np.array([[-1/2, 1/2],
                         [-1/2, 1/2],
                         [-1/40, 1/40],
                         [-1/5, 1/5],
                         [-1/3, 1/3]], dtype=np.float32) * max_vd

'''算法超参数'''
POP = 5  # population size, default 20, 5
MAX_GEN = 8  # maximum number of generations, default 10, 5
W = 0.9  # inertia weight
C1 = 0.1  # cognitive constant, the higher the value, the more the particle will look for the personal best
C2 = 0.1  # social constant, the higher the value, the more the particle will look for the global best
PREDICTION_HORIZON = 5  # prediction horizon, default 10
DECISION_HORIZON = 2  # decision horizon, default 2
INIT_RAND_RATE = 0.5  # random rate used in action sequence initialization
INIT_RAND_SIGMA = 0.1  # random sigma used in action sequence initialization
TRAIN_PER_STEPS = 10  # training frequency
Q_MEMORY_CAPACITY = int(25000)  # maximum size of critic replay buffer
Q_BATCH_SIZE = 128  # update batch size of critic 1024
Q_START_TRAIN = 5*Q_BATCH_SIZE  # start training after Q_START_TRAIN steps
A_MEMORY_CAPACITY = int(1E5)  # maximum size of actor replay buffer
A_BATCH_SIZE = 64  # update batch size of actor 1024
A_START_TRAIN = 3*A_BATCH_SIZE  # start training after A_START_TRAIN steps
M_MEMORY_CAPACITY = int(1E5)  # maximum size of model replay buffer

TARGET_UPDATE_INTERVAL = 10  # target network update interval

MAX_EPISODE = int(1E4)  # maximum episode
MAX_UPDATE_PER_STEP = 10  # maximum update times per step 4

GAMMA = 0.95  # reward discount
TAU = 0.01  # soft replacement

s_dim = 7# 17 - 7
a_dim = 2# 5
# hidden_nodes = [32, 64, 128, 256, 128, 64, 32]  # 18, 18

hidden_nodes = [128, 128, 128]  # 18, 18

# k_dis = 5.0E-2
# k_sur = 1.0E-2 * 180 / np.pi
# k_ali = 1.0E-2 * 180 / np.pi
# ctr_matrix = np.eye(5) * 1.0E-3
# k_act = 5.0E-3

k_dis = 5.0E-1
k_sur = 4.0E-1 * 180 / np.pi
k_ali = 2.0E-1 * 180 / np.pi
ctr_matrix = np.eye(5) * 1.0 * 1.0E-2
k_act = 5.0E-2

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")