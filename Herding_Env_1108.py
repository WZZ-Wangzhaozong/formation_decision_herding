import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import Scene_para as sp
import math
from scipy.optimize import linear_sum_assignment

class second_order_agent():
    def __init__(self, max_v, max_u, r=1.0):
        self.p = np.zeros(2, dtype=np.float32)
        self.v = np.zeros(2, dtype=np.float32)
        self.u = np.zeros(2, dtype=np.float32)
        self.max_v = float(max_v)
        self.max_u = float(max_u)
        self.r = r

    def reset(self, cen):
        self.p[:] = cen + (np.random.rand(2) - 0.5) * self.r
        self.v.fill(0.0)
        self.u.fill(0.0)

    def update(self, action, time):
        self.u = self.clip_fun(action, self.max_u)
        v_ini = self.v.copy()
        self.v = self.clip_fun(self.v + self.u * time, self.max_v)

        self.p += (v_ini + self.v) / 2 * time

    def clip_fun(self, vector, max_value):
        norm = np.linalg.norm(vector)
        if norm > max_value:
            vector = vector / norm * max_value
        return vector

def g_func(r, r1, r2):
    if r >= r2:
        return 0
    elif r <= 1.01 * r1:
        return (1 + np.cos(np.pi * 0.01 * r1 / (r2 - r1))) / (0.01 * r1)
    else:
        return (1 + np.cos(np.pi * (r - r1) / (r2 - r1))) / (r - r1)

class Attacking_swarm():
    def __init__(self, max_v, max_u, att_num, ini_cen, parameters):
        self.att_num = att_num
        self.rAttswm, self.r_ad_s, self.r_ad_a, self.r_aa_s, self.r_aa_a, self.k_ad, self.k_ap, self.k_aa = parameters

        self.attackers = [second_order_agent(max_v, max_u, self.rAttswm) for _ in range(self.att_num)]
        self.att_pos = np.empty((self.att_num, 2), dtype=np.float32)
        self.att_vel = np.empty((self.att_num, 2), dtype=np.float32)
        for i, attacker in enumerate(self.attackers):
            attacker.reset(ini_cen)
            self.att_pos[i] = attacker.p
            self.att_vel[i] = attacker.v

        self.cen = np.mean(self.att_pos, axis=0)
        self.rds = np.max(np.linalg.norm(self.att_pos - self.cen, axis=1))

    def update(self, tar, strings, dffs, time):
        att_pos = self.att_pos.copy()
        for i, attacker in enumerate(self.attackers):
            action = self.attacker_strategy(tar, strings, dffs, att_pos, attacker.p)
            attacker.update(action, time)
            self.att_pos[i] = attacker.p
            self.att_vel[i] = attacker.v

        self.cen = np.mean(self.att_pos, axis=0)
        self.rds = np.max(np.linalg.norm(self.att_pos - self.cen, axis=1))

    def attacker_strategy(self, tar, strings, dffs, atts, pos):

        def point_to_segment_distance(P, A, B):
            AB = B - A
            AP = P - A

            if np.allclose(A, B):
                return np.linalg.norm(P - A)

            t = np.dot(AP, AB) / np.dot(AB, AB)
            t = np.clip(t, 0.0, 1.0)  # 限制在[0,1]范围内

            closest = A + t * AB

            return closest - P

        # 1.Calculate the nearest attacker
        att_dirs = atts - pos
        att_dists = np.linalg.norm(att_dirs, axis=1)
        idx = np.argsort(att_dists)[1]  # 排除自身
        att_dir = att_dirs[idx]

        # 2.Calculate the target area
        tar_dir = tar - pos

        # 3.Calculate the nearest defender
        dff_dirs = dffs - pos
        dff_dists = np.linalg.norm(dff_dirs, axis=1)
        dff_dir = dff_dirs[np.argmin(dff_dists)]

        # Calculate the distance
        dff_dis = np.linalg.norm(dff_dir)
        att_dis = np.linalg.norm(att_dir)
        tar_dis = np.linalg.norm(tar_dir)

        # 4.Calculate the nearest string
        for string in strings:
            str_dir = point_to_segment_distance(pos, string[0], string[1])
            str_dis = np.linalg.norm(str_dir)
            if str_dis < dff_dis:
                dff_dir = str_dir.copy()
                dff_dis = str_dis

        # 5.Calculate the control law
        u1 = g_func(dff_dis, self.r_ad_s, self.r_ad_a) * (-dff_dir) / dff_dis
        u2 = tar_dir / tar_dis
        u3 = g_func(att_dis, self.r_aa_s, self.r_aa_a) * (-att_dir) / att_dis
        return self.k_ad * u1 + self.k_ap * u2 + self.k_aa * u3

    def reset(self, ini_cen):
        for i, attacker in enumerate(self.attackers):
            attacker.reset(ini_cen)
            self.att_pos[i] = attacker.p
        self.att_vel[:] = 0.0

        self.cen = np.mean(self.att_pos, axis=0)
        self.rds = np.max(np.linalg.norm(self.att_pos - self.cen, axis=1))

class String_net():
    def __init__(self, max_v, max_u, alpha_bound, dff_num, pos, phi, zeta, beta, parameters):
        self.rString, self.r_safe, self.r_avoid, self.k_p, self.k_av, self.k_v2u = parameters
        self.alpha_bound = alpha_bound

        # Formation parameters
        self.actual_cen = np.array(pos, dtype=np.float32)
        self.phi = phi
        self.zeta = zeta
        self.beta = beta

        # Parameters for defenders
        self.dff_num = dff_num
        self.dff_pos_desired = np.empty((self.dff_num, 2), dtype=np.float32)
        self.dff_vel_desired = np.zeros([2], dtype=np.float32)
        self.dff_pos = np.empty((self.dff_num, 2), dtype=np.float32)
        self.dff_vel = np.empty((self.dff_num, 2), dtype=np.float32)
        self.defenders = [second_order_agent(max_v, max_u, self.rString * 2.5) for _ in range(self.dff_num)]
        self.formation_round_up()
        self.shape_cen = self.arc_center(self.dff_pos_desired[0, :], self.dff_pos_desired[1, :], self.dff_pos_desired[2, :])
        for i, defender in enumerate(self.defenders):
            defender.reset(self.dff_pos_desired[i])
            self.dff_pos[i] = defender.p
            self.dff_vel[i] = defender.v
        self.rds = self.zeta / (2 * np.sin(self.beta / (2 * (self.dff_num - 1))))
        self.bottom = self.shape_cen - self.rds * np.array([np.cos(self.phi), np.sin(self.phi)])
        self.rds_last = self.rds

        strings = []
        for i in range(-1, self.dff_num-1):
            if np.linalg.norm(self.dff_pos[i] - self.dff_pos[i + 1]) <= self.rString * 1.05:
                strings.append([self.dff_pos[i], self.dff_pos[i+1]])
        self.strings = np.array(strings, dtype=np.float32)
        self.actual_vel = np.zeros([2], dtype=np.float32)

        self.reordering()
        self.update(np.zeros([5]), 0.0)

    def get_params(self):
        return np.array([
            self.actual_cen[0],
            self.actual_cen[1],
            self.phi,
            self.zeta,
            self.beta
        ], dtype=np.float32)

    def negotiation_update(self, rate):
        rate = np.asarray(rate, dtype=np.float32)
        if rate.shape[0] != 5:
            raise ValueError(f"rate should have 5 elements, got {len(rate)}")

        vx_local, vy_local = rate[0], rate[1]
        self.dff_vel_desired = np.array([
            vx_local * np.cos(self.phi) - vy_local * np.sin(self.phi),
            vx_local * np.sin(self.phi) + vy_local * np.cos(self.phi)
        ])

        # Update Defensive Formation
        consensus_gain = 0.1
        self.actual_cen += self.dff_vel_desired * consensus_gain
        self.phi += rate[2] * consensus_gain
        self.zeta += rate[3] * consensus_gain
        self.beta += rate[4] * consensus_gain

        self.actual_cen[0] = np.clip(self.actual_cen[0], self.alpha_bound[0, 0], self.alpha_bound[0, 1])
        self.actual_cen[1] = np.clip(self.actual_cen[1], self.alpha_bound[1, 0], self.alpha_bound[1, 1])
        self.phi = (self.phi + np.pi) % (2 * np.pi) - np.pi
        self.zeta = np.clip(self.zeta, self.alpha_bound[3, 0], self.alpha_bound[3, 1])
        self.beta = np.clip(self.beta, self.alpha_bound[4, 0], self.alpha_bound[4, 1])

    def update(self, rate, time, type='central'):
        if type == 'central':
            # self.reordering()
            None
        elif type == 'distributed':
            None

        rate = np.asarray(rate, dtype=np.float32)
        if rate.shape[0] != 5:
            raise ValueError(f"rate should have 5 elements, got {len(rate)}")

        vx_local, vy_local = rate[0], rate[1]
        self.dff_vel_desired = np.array([
            vx_local * np.cos(self.phi) - vy_local * np.sin(self.phi),
            vx_local * np.sin(self.phi) + vy_local * np.cos(self.phi)
        ])

        # Update Defensive Formation
        self.actual_cen += self.dff_vel_desired * time
        self.phi += rate[2] * time
        self.zeta += rate[3] * time
        self.beta += rate[4] * time

        self.actual_cen[0] = np.clip(self.actual_cen[0], self.alpha_bound[0, 0], self.alpha_bound[0, 1])
        self.actual_cen[1] = np.clip(self.actual_cen[1], self.alpha_bound[1, 0], self.alpha_bound[1, 1])
        self.phi = (self.phi + np.pi) % (2 * np.pi) - np.pi
        self.zeta = np.clip(self.zeta, self.alpha_bound[3, 0], self.alpha_bound[3, 1])
        self.beta = np.clip(self.beta, self.alpha_bound[4, 0], self.alpha_bound[4, 1] / sp.dff_num * (sp.dff_num - 0.5))

        # Update Defenders
        self.formation_round_up()
        self.shape_cen = self.arc_center(self.dff_pos_desired[0, :], self.dff_pos_desired[1, :], self.dff_pos_desired[2, :])
        for i, (defender, dff_pos_desired) in enumerate(zip(self.defenders, self.dff_pos_desired)):
            neighbor_pos = np.delete(self.dff_pos, i, axis=0)
            action = self.defender_strategy(dff_pos_desired, self.dff_vel_desired, neighbor_pos,
                                            defender.p, defender.v)
            defender.update(action, time)

        # Preserve the defender's actual position and speed
        self.dff_pos[:] = [defender.p for defender in self.defenders]
        self.dff_vel[:] = [defender.v for defender in self.defenders]
        self.rds = self.zeta / (2 * np.sin(self.beta / (2 * (self.dff_num - 1))))
        self.rds_last = self.rds
        self.bottom = self.shape_cen - self.rds * np.array([np.cos(self.phi), np.sin(self.phi)])
        self.actual_vel = np.mean(self.dff_vel, axis=0)

        # Save the paired defender positions to form a string
        strings = []
        for i in range(-1, self.dff_num - 1):
            if np.linalg.norm(self.dff_pos[i] - self.dff_pos[i + 1]) <= self.rString * 1.05:
                strings.append([self.dff_pos[i], self.dff_pos[i + 1]])
        self.strings = np.array(strings, dtype=np.float32)

    def update_distributed(self, dff_pos_desired, dff_vel_desired, time):
        for i, (defender, dff_pos_desired_i, dff_vel_desired_i) in enumerate(zip(self.defenders, dff_pos_desired, dff_vel_desired)):
            neighbor_pos = np.delete(self.dff_pos, i, axis=0)
            action = self.defender_strategy(dff_pos_desired_i, dff_vel_desired_i, neighbor_pos,
                                            defender.p, defender.v)
            defender.update(action, time)

        self.dff_pos[:] = [defender.p for defender in self.defenders]
        self.dff_pos_desired = dff_pos_desired
        strings = []
        for i in range(-1, self.dff_num - 1):
            if np.linalg.norm(self.dff_pos[i] - self.dff_pos[i + 1]) <= self.rString * 1.05:
                strings.append([self.dff_pos[i], self.dff_pos[i + 1]])
        self.strings = np.array(strings, dtype=np.float32)

    def reset(self, pos, phi, zeta, beta):
        self.actual_cen = np.array(pos, dtype=np.float32)
        self.phi = phi
        self.zeta = zeta
        self.beta = beta

        self.formation_round_up()  # 用于更新 self.dff_pos_desired
        self.shape_cen = self.arc_center(self.dff_pos_desired[0, :], self.dff_pos_desired[1, :], self.dff_pos_desired[2, :])  # 计算形状中心
        self.dff_vel_desired[:] = 0.0
        for i, defender in enumerate(self.defenders):
            defender.reset(self.dff_pos_desired[i])
            # defender.p = self.dff_pos_desired[i].copy()
            self.dff_pos[i] = defender.p
        self.dff_vel[:] = 0.0
        self.rds = self.zeta / (2 * np.sin(self.beta / (2 * (self.dff_num - 1))))
        self.bottom = self.shape_cen - self.rds * np.array([np.cos(self.phi), np.sin(self.phi)])

        strings = []
        for i in range(-1, self.dff_num-1):
            if np.linalg.norm(self.dff_pos[i] - self.dff_pos[i + 1]) <= self.rString * 1.05:
                strings.append([self.dff_pos[i], self.dff_pos[i+1]])
        self.strings = np.array(strings, dtype=np.float32)
        self.actual_vel = np.zeros([2], dtype=np.float32)

        self.reordering()
        self.update(np.zeros([5]), 0.0)

    def formation_round_up(self):
        self.dff_pos_desired[:] = 0
        zeta = self.zeta * 8 / self.dff_num * np.sqrt(sp.att_num / 3)

        d_safe = 1.0
        offset_dist = 0.0

        if zeta < d_safe:
            offset_dist = np.sqrt(max(0, d_safe ** 2 - zeta ** 2)) / 2

        for i in range(self.dff_num)[1:]:
            angle = (2 * i - self.dff_num) / (self.dff_num - 1) * self.beta / 2 + np.pi / 2 + self.phi
            self.dff_pos_desired[i, 0] = self.dff_pos_desired[i - 1, 0] - zeta * np.cos(angle)
            self.dff_pos_desired[i, 1] = self.dff_pos_desired[i - 1, 1] - zeta * np.sin(angle)

        for i in range(self.dff_num):
            angle = (2 * i - self.dff_num) / (self.dff_num - 1) * self.beta / 2 + np.pi / 2 + self.phi
            normal_vec = np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)])

            if i % 2 == 0:
                self.dff_pos_desired[i] += offset_dist * normal_vec
            else:
                self.dff_pos_desired[i] -= offset_dist * normal_vec

        origin = np.mean(self.dff_pos_desired, axis=0)
        self.dff_pos_desired -= origin
        self.dff_pos_desired += self.actual_cen

    def arc_center(self, p1, p2, p3):
        # p1, p2, p3: np.array([x, y])
        A = np.array([[p2[0] - p1[0], p2[1] - p1[1]],
                      [p3[0] - p2[0], p3[1] - p2[1]]])
        B = np.array([[(p2[0] ** 2 - p1[0] ** 2 + p2[1] ** 2 - p1[1] ** 2) / 2],
                      [(p3[0] ** 2 - p2[0] ** 2 + p3[1] ** 2 - p2[1] ** 2) / 2]])
        center = np.linalg.lstsq(A, B, rcond=None)[0].flatten()
        return center

    def defender_strategy(self, desired_pos, desired_vel, neighbor_pos, pos, vel):
        diff_pos = self.k_p * (desired_pos - pos)
        diff_vel = desired_vel - vel
        v_form = diff_pos + diff_vel

        if len(neighbor_pos):
            rela_pos = pos - neighbor_pos
            distances = np.linalg.norm(rela_pos, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_rela_pos = rela_pos[nearest_idx]
            nearest_distance = distances[nearest_idx]
            v_col = self.k_av * g_func(nearest_distance, self.r_safe, self.r_avoid) * nearest_rela_pos / nearest_distance

        u_ctrl = self.k_v2u * (v_form + v_col)

        return u_ctrl

    def hungarian_matching(self, coords1, coords2):
        n = len(coords1)

        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i, j] = np.linalg.norm(coords1[i] - coords2[j])

        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        matches = []
        total_distance = 0
        for i, j in zip(row_ind, col_ind):
            dist = distance_matrix[i, j]
            matches.append((i, j, dist))
            total_distance += dist

        return matches, total_distance

    def reordering(self):
        self.shape_cen = self.arc_center(self.dff_pos_desired[0, :], self.dff_pos_desired[1, :],
                                         self.dff_pos_desired[2, :])

        matches, total_distance = self.hungarian_matching(self.dff_pos, self.dff_pos_desired)

        mapping = sorted(matches, key=lambda x: x[1])
        reorder_index = [i for i, j, _ in mapping]

        self.defenders = [self.defenders[i] for i in reorder_index]
        self.dff_pos[:] = [defender.p for defender in self.defenders]
        self.dff_vel[:] = [defender.v for defender in self.defenders]

class herding_env():
    def __init__(self, str_net_para, att_swm_para, save_data=False):
        # fixed parameters settings
        self.max_va, self.max_ua, self.att_num, self.attacker_para = att_swm_para
        self.max_vd, self.max_ud, self.alpha_bound, self.dff_num, self.defender_para = str_net_para
        self.target_area = np.array([sp.target_area_x, sp.target_area_y]); self.target_area_rds = sp.target_area_rds
        self.time = sp.sim_time
        self.simulation_time = 0.0

        # random parameters settings
        self.att_cen = np.array([random.uniform(sp.att_x_left, sp.att_x_right), random.uniform(sp.att_y_down, sp.att_y_up)])
        self.dff_cen = np.array([random.uniform(sp.dff_x_left, sp.dff_x_right), random.uniform(sp.dff_y_down, sp.dff_y_up)])
        self.phi = np.arctan2(self.att_cen[1]-self.dff_cen[1], self.att_cen[0]-self.dff_cen[0]) + random.uniform(sp.phi_random_down, sp.phi_random_up)

        if self.phi < -np.pi:
            self.phi += 2 * np.pi
        elif self.phi > np.pi:
            self.phi -= 2 * np.pi
        self.zeta = random.uniform(sp.zeta_random_down, sp.zeta_random_up)
        self.beta = random.uniform(sp.beta_random_down, sp.beta_random_up) * np.pi

        # instantiation of both sides
        self.attack_swarm = Attacking_swarm(self.max_va, self.max_ua, self.att_num, self.att_cen, self.attacker_para)
        self.string_net = String_net(self.max_vd, self.max_ud, self.alpha_bound, self.dff_num, self.dff_cen, self.phi, self.zeta, self.beta, self.defender_para)

        self.done = False
        self.game_finish = 0  # 1为被入侵；2为包围成功

        self.alpha = np.zeros([5])
        self.action = np.zeros([5])
        self.action_agent = np.zeros([self.dff_num, 2])
        self.defenders_pos = np.zeros([self.dff_num, 2])
        self.attackers_pos = np.zeros([self.att_num, 2])
        self.defenders_vel = np.zeros([self.dff_num, 2])
        self.attackers_vel = np.zeros([self.att_num, 2])

    def deception_observe(self, dist):
        base_error_p = 0.0
        k_p = 2.0
        base_error_v = 0.0
        k_v = 0.4

        dist -= 5.0
        dist[dist < 0] = 0.0
        dist[dist > 5] = 5.0

        sigmas_p = base_error_p + k_p * dist
        sigmas_v = base_error_v + k_v * dist

        noise_p = np.array([np.random.normal(0, s, 2) for s in sigmas_p])
        noise_v = np.array([np.random.normal(0, s, 2) for s in sigmas_v])
        return noise_p, noise_v

    def global_trans_to_local(self, string_net=None, attack_swarm=None, target_area=None):
        string_net = string_net or self.string_net
        attack_swarm = attack_swarm or self.attack_swarm
        target_area = target_area if target_area is not None else self.target_area

        phi = string_net.phi
        trans_matrix = np.array([[np.cos(phi), np.sin(phi)],
                                 [-np.sin(phi), np.cos(phi)]])

        ######################### 增加观测误差 ########################
        attack_swarm_att_pos = attack_swarm.att_pos.copy()
        attack_swarm_att_dis = np.linalg.norm(attack_swarm_att_pos, axis=1)
        attack_swarm_att_random = self.deception_observe(attack_swarm_att_dis)
        attack_swarm_att_pos += attack_swarm_att_random[0]
        attack_swarm_cen_obs = np.mean(attack_swarm_att_pos, axis=0)
        ##############################################################

        attack_swarm_cen = trans_matrix @ (attack_swarm_cen_obs - string_net.actual_cen)  # @ trans_matrix
        target_area_cen = trans_matrix @ (target_area - string_net.actual_cen)  # @ trans_matrix

        cos_beta = math.cos(string_net.beta)
        sin_beta = math.sin(string_net.beta)

        zeta = string_net.zeta

        r_ad = float(min(np.linalg.norm(attack_swarm_cen).astype(np.float32), 15.0))
        angle_ad = np.arctan2(attack_swarm_cen[1], attack_swarm_cen[0])
        cos_ad = math.cos(angle_ad)
        sin_ad = math.sin(angle_ad)

        r_dp = float(min(np.linalg.norm(target_area_cen).astype(np.float32), 20.0))
        angle_dp = np.arctan2(target_area_cen[1], target_area_cen[0])
        cos_dp = math.cos(angle_dp)
        sin_dp = math.sin(angle_dp)

        rela_pos_ap = attack_swarm_cen - target_area_cen
        r_ap = float(min(np.linalg.norm(rela_pos_ap).astype(np.float32) - 2.0, 20.0))
        angle_ap = np.arctan2(rela_pos_ap[1], rela_pos_ap[0])
        cos_ap = math.cos(angle_ap)
        sin_ap = math.sin(angle_ap)

        att_rds = max(attack_swarm.rds, 0.5)

        ######################### 增加观测误差 ########################
        attack_swarm_att_vel = attack_swarm.att_vel.copy()
        attack_swarm_att_vel += attack_swarm_att_random[1]
        attack_swarm_vel_obs = np.mean(attack_swarm_att_vel, axis=0)
        ##############################################################

        v_att = trans_matrix @ (attack_swarm_vel_obs - string_net.actual_vel)
        r_v_att = np.linalg.norm(v_att).astype(np.float32)
        angle_v_att = np.arctan2(v_att[1], v_att[0])
        cos_v_att = np.cos(angle_v_att).astype(np.float32)
        sin_v_att = np.sin(angle_v_att).astype(np.float32)

        return torch.tensor([
                r_ad, cos_ad, sin_ad,
                # r_ap, cos_ap, sin_ap,
                r_dp, cos_dp, sin_dp,
                zeta,
                cos_beta, sin_beta,
                att_rds * 3 / attack_swarm.att_num,
                r_v_att, cos_v_att, sin_v_att,
        ]).to(sp.device)

    def update(self, action):
        self.action = action
        self.string_net.update(self.action, self.time)
        self.attack_swarm.update(self.target_area, self.string_net.strings, self.string_net.dff_pos, self.time)
        self.simulation_time += self.time

        att_dis = np.linalg.norm(self.attack_swarm.att_pos - self.target_area, axis=1)
        if np.any(att_dis <= self.target_area_rds):
            self.game_finish = 1
            self.done = True
        elif self.get_herding_info():
            self.game_finish = 2
            self.done = True
        elif self.simulation_time >= sp.maxTime:  # 超出时长，平
            self.game_finish = 3
            self.done = True

    def update_distributed(self, dff_pos_desired, dff_vel_desired):
        self.string_net.update_distributed(dff_pos_desired, dff_vel_desired, self.time)
        self.attack_swarm.update(self.target_area, self.string_net.strings, self.string_net.dff_pos, self.time)
        self.simulation_time += self.time

        att_dis = np.linalg.norm(self.attack_swarm.att_pos - self.target_area, axis=1)
        if np.any(att_dis <= self.target_area_rds):
            self.game_finish = 1
            self.done = True
        elif self.get_herding_info():
            self.game_finish = 2
            self.done = True
        elif self.simulation_time >= sp.maxTime:  # 超出时长，平
            self.game_finish = 3
            self.done = True

    def reset(self):
        # random parameters settings
        self.att_cen = np.array([random.uniform(sp.att_x_left, sp.att_x_right), random.uniform(sp.att_y_down, sp.att_y_up)])
        self.dff_cen = np.array([random.uniform(sp.dff_x_left, sp.dff_x_right), random.uniform(sp.dff_y_down, sp.dff_y_up)])
        self.phi = np.arctan2(self.att_cen[1] - self.dff_cen[1], self.att_cen[0] - self.dff_cen[0]) + random.uniform(sp.phi_random_down, sp.phi_random_up)

        if self.phi < -np.pi:
            self.phi += 2 * np.pi
        elif self.phi > np.pi:
            self.phi -= 2 * np.pi
        self.zeta = random.uniform(sp.zeta_random_down, sp.zeta_random_up)
        self.beta = random.uniform(sp.beta_random_down, sp.beta_random_up) * np.pi

        # resetting of both sides
        self.attack_swarm.reset(self.att_cen)
        self.string_net.reset(self.dff_cen, self.phi, self.zeta, self.beta)

        self.simulation_time = 0.0
        self.done = False  # True为成功包围
        self.game_finish = 0  # 1为被入侵；2为包围成功

    def get_herding_info(self, string_net=None, attack_swarm=None):
        """
        calculate done and info based on the states of defenders and attackers, for herding game
        Args:
            defs: defenders' states, [x, y, vx, vy]* numDefender
            atts: attackers' states, [x, y, vx, vy]* numAttacker
            r_string: radius of the string
        """
        string_net = self.string_net if string_net is None else string_net
        attack_swarm = self.attack_swarm if attack_swarm is None else attack_swarm

        if string_net.strings.shape[0] < self.dff_num:
            return False
        else:
            cen_def = np.mean(string_net.dff_pos, axis=0)
            for att_pos in attack_swarm.att_pos:
                if np.linalg.norm(att_pos - cen_def) > string_net.zeta * self.dff_num / np.pi / 2:
                    return False
            return True

    def func_da_dis(self, string_net=None, attack_swarm=None):
        string_net = self.string_net if string_net is None else string_net
        attack_swarm = self.attack_swarm if attack_swarm is None else attack_swarm

        rela_pos = string_net.actual_cen - attack_swarm.cen
        r_da = np.linalg.norm(rela_pos)
        r_da = min(r_da, 15.0)

        return r_da

    def func_ap_dis(self, target_area=None, attack_swarm=None, target_radius=None):
        target_area = self.target_area if target_area is None else target_area
        target_radius = self.target_area_rds if target_radius is None else target_radius
        attack_swarm = self.attack_swarm if attack_swarm is None else attack_swarm

        rela_pos = target_area - attack_swarm.cen
        r_ap = np.linalg.norm(rela_pos) - target_radius
        r_ap = min(r_ap, 20.0)

        return r_ap

    def func_sur(self, attack_swarm=None, string_net=None):
        string_net = self.string_net if string_net is None else string_net
        attack_swarm = self.attack_swarm if attack_swarm is None else attack_swarm

        def get_surround_angle(points, point):
            """
            calculate the surrounding angle
            按照围捕距离，计算每一个防御者对进攻者的捕获角度区间，之后按照几何叠加
            """
            rCapture = 1.6
            dis_vec = np.linalg.norm(points - point, axis=1)  # 计算防御者到进攻者的距离
            angle_delta = np.arctan2(rCapture, dis_vec)  # 计算防御者到进攻者的捕获角度
            order = np.argsort(angle_delta)[::-1].astype(int)  # 按照捕获角度排序,使用切片操作获得降序排列的索引
            angle_center = np.arctan2(points[:, 1] - point[1], points[:, 0] - point[0])  # 初始化捕获角度
            angle_min = saturation_angle(angle_center - angle_delta, angle_min=0.0, angle_max=2 * np.pi, dim=2)
            angle_max = saturation_angle(angle_center + angle_delta, angle_min=0.0, angle_max=2 * np.pi, dim=2)

            angle = 0.0  # 初始化捕获角度
            for i in range(points.shape[0]):
                # 计算被前面已考虑过的防御者的捕获角度区间遮挡的区域
                contain_flag = False  # 是否被前面已考虑过的的防御者完全遮挡
                if i > 0:
                    # edge_side = np.zeros((i, 2))  # 其他区域的边界在自己区域的哪一侧，0表示内侧，1表示外侧。 左，右，自己中心线
                    blocking_angle = np.zeros((i, 2))  # 被遮挡的角度区间，左，右
                    for j in range(0, i):
                        left = saturation_angle(angle_max[order[j]] - angle_min[order[i]], 0.0, 2 * np.pi)
                        right = saturation_angle(angle_max[order[i]] - angle_min[order[j]], 0.0, 2 * np.pi)
                        if left < 2 * angle_delta[order[i]]:
                            blocking_angle[j, 0] = left
                        if right < 2 * angle_delta[order[i]]:
                            blocking_angle[j, 1] = right
                        if blocking_angle[j, 0] ** 2 + blocking_angle[j, 1] ** 2 == 0.0:  # 都在外侧的情况判断是否完全包含
                            center = saturation_angle(angle_center[order[i]] - angle_min[order[j]], 0.0, 2 * np.pi)
                            if center < 2 * angle_delta[order[j]]:
                                contain_flag = True
                                break
                    block_angle_l = np.max(blocking_angle[:, 0])
                    block_angle_r = np.max(blocking_angle[:, 1])
                    if not contain_flag:
                        angle += np.max([0.0, 2 * angle_delta[order[i]] - block_angle_l - block_angle_r])
                else:
                    angle += 2 * angle_delta[order[i]]
            return angle

        def saturation_angle(angle, angle_min, angle_max, dim=1):
            """
            saturation
            ----------------------------------
                - angle: the angle to be saturated
                - angle_min: the minimum value of the angle, default: -pi
                - angle_max: the maximum value of the angle, default: pi
            """
            if dim == 1:
                while angle < angle_min:
                    angle += 2 * np.pi
                while angle > angle_max:
                    angle -= 2 * np.pi
            elif dim == 2:
                angle[angle > angle_max] -= 2 * np.pi
                angle[angle < angle_min] += 2 * np.pi
            else:
                raise ValueError("dim must be 1 or 2")
            return angle

        # 进攻者中心已在Stringnet形状内部
        if np.linalg.norm(string_net.shape_cen - attack_swarm.cen) <= string_net.rds:
            # print(1)
            pass
        else:
            def is_between(p1, p2, p3):
                # 计算三个方向角
                phi1 = np.arctan2(p1[1], p1[0])
                phi2 = np.arctan2(p2[1], p2[0])
                phi3 = np.arctan2(p3[1], p3[0])

                # 计算最短弧的角度差
                def angle_diff(a, b):
                    diff = a - b
                    return (diff + np.pi) % (2 * np.pi) - np.pi

                diff12 = angle_diff(phi2, phi1)  # φ1→φ2
                diff13 = angle_diff(phi3, phi1)  # φ1→φ3

                # 如果 φ1→φ2 的角度 < π
                if abs(diff12) <= np.pi:
                    return 0 <= diff13 <= diff12 if diff12 >= 0 else diff12 <= diff13 <= 0
                else:
                    # 如果 φ1→φ2 的弧度 > π，取补角
                    return not (0 <= diff13 <= diff12) if diff12 >= 0 else not (diff12 <= diff13 <= 0)

            end1 = string_net.dff_pos[0]
            end2 = string_net.dff_pos[-1]
            bottom = string_net.bottom

            rela_pos_1 = attack_swarm.cen - end1
            rela_pos_2 = attack_swarm.cen - end2
            rela_pos_3 = attack_swarm.cen - bottom

            # 进攻者中心在Stringnet底部正前方和开口内部
            if is_between(rela_pos_1, rela_pos_2, -rela_pos_3):
                # print(2)
                pass
            elif is_between(rela_pos_1, rela_pos_2, rela_pos_3):
                # 进攻者中心在开口前方
                if np.dot(np.array([np.cos(string_net.phi), np.sin(string_net.phi)]), rela_pos_3) > 0:
                    # print(3)
                    pass
                # 进攻者中心在底端后方
                else:
                    # print(4)
                    return -1, -1
            elif not is_between(rela_pos_1, rela_pos_2, rela_pos_3):
                # 进攻者中心在两端口连线前方
                if np.dot(np.array([np.cos(string_net.phi), np.sin(string_net.phi)]), (rela_pos_1 + rela_pos_2) / 2) > 0:
                    # print(5)
                    return -1, -1
                # 进攻者中心在两端口连线后方
                else:
                    # print(6)
                    return -1, -1

        rela_pos = string_net.actual_cen - attack_swarm.cen
        r_da = np.linalg.norm(rela_pos)
        r_da = min(r_da, 15.0)

        strings_len = string_net.beta * string_net.rds
        if strings_len >= np.pi * (r_da + attack_swarm.rds) and string_net.beta >= np.pi:
            sur_cen = attack_swarm.cen
        else:
            sur_cen = string_net.actual_cen + (attack_swarm.cen - string_net.actual_cen) * (1 + attack_swarm.rds / r_da)

        phi_nea = get_surround_angle(string_net.dff_pos, sur_cen).item()
        if phi_nea >= np.pi * 1.99:
            phi_nea = np.pi * 2
        return phi_nea

    @staticmethod
    def angle_diff(a, b):
        """Compute minimal difference between two angles in [-pi, pi]."""
        diff = a - b
        return (diff + np.pi) % (2 * np.pi) - np.pi  # 将差映射到 [-π, π]

    @staticmethod
    def point_to_segment_distance(P, A, B):
        """
        计算点 P 到线段 AB 的最短距离。
        P, A, B: np.array([x, y]) 或 [x, y]
        """
        AB = B - A
        AP = P - A

        # 若A与B重合
        if np.allclose(A, B):
            return np.linalg.norm(P - A)

        # 投影比例t
        t = np.dot(AP, AB) / np.dot(AB, AB)
        t = np.clip(t, 0.0, 1.0)  # 限制在[0,1]范围内

        # 最近点
        closest = A + t * AB

        return closest - P

    @staticmethod
    def f(x, k=1.0, b=0.5, alpha=20):
        """
        平滑分段函数:
        - 在 [0, b] 区间近似为 1
        - 在 (b, +∞) 区间逐渐接近 -k*x
        - alpha 控制平滑度，越大过渡越硬
        """
        s = 0.5 * (1 + np.tanh(alpha * (x - b)))  # 平滑开关
        return (1 - s) * 1 + s * (-k * x)

    def func_ali(self, target_area=None, attack_swarm=None, string_net=None, target_radius=None):
        string_net = self.string_net if string_net is None else string_net
        attack_swarm = self.attack_swarm if attack_swarm is None else attack_swarm
        target_area = self.target_area if target_area is None else target_area
        target_radius = self.target_area_rds if target_radius is None else target_radius

        # 1️⃣ 计算 min_point：在保护区→攻击者方向上延伸 target_radius + 1.0 的点
        direction = attack_swarm.cen - target_area
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-6:
            direction_unit = direction / direction_norm
        else:
            direction_unit = np.zeros_like(direction)
        min_point = target_area + direction_unit * (target_radius + 1.0)

        # 2️⃣ 计算防御者到 min_point–attacker 线段的最近点
        line_vec = attack_swarm.cen - min_point
        point_vec = string_net.actual_cen - min_point
        line_len = np.linalg.norm(line_vec)
        if line_len > 1e-6:
            proj = np.dot(point_vec, line_vec) / line_len
            proj = np.clip(proj, 0, line_len)  # 限制在线段内
            closest_point = min_point + (proj / line_len) * line_vec
        else:
            closest_point = min_point

        # 3️⃣ 计算距离与奖励
        r_dl = np.linalg.norm(string_net.actual_cen - closest_point)

        rela_pos_ad = attack_swarm.cen - string_net.actual_cen
        phi_ad = np.arctan2(rela_pos_ad[1], rela_pos_ad[0])
        ali_ad = abs(self.angle_diff(string_net.phi, phi_ad))
        return r_dl, ali_ad

    def func_opening(self, phi_nea, string_net=None):
        string_net = self.string_net if string_net is None else string_net

        opening = 0.0
        if phi_nea < np.pi / 3 and string_net.beta >= np.pi:
            opening = string_net.beta - np.pi  # + np.pi / 180 * 2

        return opening

    def situation_calculation(self):
        r_dl, ali_ad = self.func_ali()
        reward_r_dl = self.f(r_dl, k=1.0, b=0.5)
        reward_ali_ad = self.f(ali_ad / np.pi * 180, k=0.1, b=3.0)

        r_ap = self.func_ap_dis()
        if r_ap >= 10.0:
            reward_r_ap = 1.0
        else:
            reward_r_ap = r_ap - 9.0

        phi_nea = self.func_sur()
        r_da = self.func_da_dis()
        if type(phi_nea) == float:
            x = phi_nea / (np.pi * 2)
            base_reward = (x - 2.0) * 2.0
            # base_reward = ((phi_nea / (np.pi * 2) + 1) ** 3 - 1) / 7.0 * 3.0  # + 0.5

            if r_da <= 5.0:
                reward_phi_nea = base_reward
                if phi_nea >= np.pi:
                    reward_r_dl = 1.0
                    reward_ali_ad = 1.0
                    reward_r_ap = 1.0
            elif r_da < 10.0:
                reward_phi_nea = base_reward * (0.1 * r_da + 0.5)  # * (10.0 - r_da) / (10.0 - 5.0)
            else:
                reward_phi_nea = base_reward * 1.5

            # if phi_nea >= np.pi * 1.6 and self.string_net.beta < 2.0 * np.pi:
            #     penalty = -8.0 * (2.0 - self.string_net.beta / np.pi)
            #     reward_phi_nea += penalty
        else:
            phi_nea = 0.0
            reward_phi_nea = -6.0
        reward_phi_nea -= 3.0

        return reward_r_dl + reward_ali_ad + reward_r_ap + reward_phi_nea


if __name__ == "__main__":
    attacker_para = sp.rAttswm, sp.r_ad_s, sp.r_ad_a, sp.r_aa_s, sp.r_aa_a, sp.k_ad, sp.k_ap, sp.k_aa
    defender_para = sp.rString, sp.r_safe, sp.r_avoid, sp.k_p, sp.k_av, sp.k_v2u
    att_swm_para = sp.max_va, sp.max_ua, sp.att_num, attacker_para
    str_net_para = sp.max_vd, sp.max_ud, sp.alpha_bound, sp.dff_num, defender_para

    env = herding_env(str_net_para, att_swm_para, save_data=True)
    for i in range(100):
        aaa = env.global_trans_to_local()
        env.reset()