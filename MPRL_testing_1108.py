import sys
import copy
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import Herding_Env_1108 as Herding_Env
import Scene_para as sp
from matplotlib.patches import Circle
from RVO import RVO_update, reach, compute_V_des, reach
import os

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.action_low = torch.tensor(action_low, dtype=torch.float32).to(sp.device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32).to(sp.device)

    def forward(self, x):
        out = self.net(x)  # 输出 [-1,1]
        return self.action_low + (out + 1.0) / 2.0 * (self.action_high - self.action_low)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, state_low, state_high, action_low, action_high):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.state_low = torch.tensor(state_low, dtype=torch.float32).to(sp.device)
        self.state_high = torch.tensor(state_high, dtype=torch.float32).to(sp.device)
        self.action_low = torch.tensor(action_low, dtype=torch.float32).to(sp.device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32).to(sp.device)

    def forward(self, x, u):
        x = (x - self.state_low) / (self.state_high - self.state_low) * 2.0 - 1.0
        u = (u - self.action_low) / (self.action_high - self.action_low) * 2.0 - 1.0
        return self.net(torch.cat([x, u], dim=1))

class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_low, action_high, state_low, state_high):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, action_low, action_high).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, action_low, action_high).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, state_low, state_high, action_low, action_high).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, state_low, state_high, action_low, action_high).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.buffer = deque(maxlen=250000)
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64

    def store(self, s, a, r, s_, d):
        # 确保不保存计算图
        self.buffer.append((
            s.detach().clone(),
            a.detach().clone(),
            float(r),
            s_.detach().clone(),
            float(d)
        ))
        # self.buffer.append((s, a, r, s_, d))

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s_, d = zip(*batch)

        s = torch.stack(s)
        a = torch.stack(a)
        r = torch.FloatTensor(r).to(self.device).unsqueeze(1)
        s_ = torch.stack(s_)
        d = torch.FloatTensor(d).to(self.device).unsqueeze(1)

        with torch.no_grad():
            a_next = self.actor_target(s_)
            q_next = self.critic_target(s_, a_next)
            q_target = r + self.gamma * (1 - d) * q_next

        q_value = self.critic(s, a)
        critic_loss = nn.MSELoss()(q_value, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # 更新 actor
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # 软更新
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def angle_diff(phi_target, phi_source):
    """
    计算 phi_target - phi_source 的最短弧度差
    返回值范围在 [-pi, pi]
    """
    diff = phi_target - phi_source
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff

def global_to_local_velocity(vx_global, vy_global, phi):
    """
    将全局速度 (vx_global, vy_global) 转换为局部速度 (vx_local, vy_local)
    相当于旋转 -phi
    """
    vx_local = vx_global * np.cos(phi) + vy_global * np.sin(phi)
    vy_local = -vx_global * np.sin(phi) + vy_global * np.cos(phi)
    return np.array([vx_local, vy_local])


if __name__ == "__main__":
    plt.ion()
    plt.pause(0.01)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(sp.xlim_left, sp.xlim_right)
    ax.set_ylim(sp.ylim_down, sp.ylim_up)
    ax.set_aspect('equal')
    ax.tick_params(
        axis='both',
        which='both',
        labelleft=False,
        labelbottom=False,
        left=True,
        bottom=True,
        length=4,
        width=1
    )
    ax.set_xticks(np.arange(sp.xlim_left, sp.xlim_right + 1, 20))  # x轴每10个单位一个刻度
    ax.set_yticks(np.arange(sp.ylim_down, sp.ylim_up + 1, 10))  # y轴每10个单位一个刻度
    # ax.grid(True)
    dff_cen, = ax.plot([], [], 'bo', markersize=4.0, label='Dff_Cen')
    dff_bot, = ax.plot([], [], '*', markersize=3.0, label='Dff_Cen')
    att_cen, = ax.plot([], [], 'ro', label='Dff_Cen')
    att_pts, = ax.plot([], [], 'ro', markersize=5.0, label='Attackers')
    dff_des_pts, = ax.plot([], [], 'ko', markersize=4.0)
    dff_pts, = ax.plot([], [], 'bo', markersize=5.0, label='Defenders')
    strings, = ax.plot([], [], 'b', linewidth=1.5)
    att_traj_lines = [
        ax.plot([], [], '-', color='lightcoral', linewidth=1.0)[0]
        for _ in range(sp.att_num)
    ]
    dff_traj_lines = [
        ax.plot([], [], '-.', color='cornflowerblue', linewidth=1.0)[0]
        for _ in range(sp.dff_num)
    ]

    target_circle = Circle((sp.target_area_x, sp.target_area_y), sp.target_area_rds, color='purple', fill=True)
    goal_circle = Circle((sp.goal_area_x, sp.goal_area_y), sp.target_area_rds, color='green', fill=True)
    ax.add_patch(target_circle)
    ax.add_patch(goal_circle)
    for i in range(len(sp.obstacles)):
        obs = sp.obstacles[i]
        obs_circle = Circle((obs[0], obs[1]), obs[2], color='grey', fill=True)
        ax.add_patch(obs_circle)

    r_ad_s = random.uniform(sp.r_ad_s_down, sp.r_ad_s_up)
    r_ad_a = random.uniform(sp.r_ad_a_down, sp.r_ad_a_up)
    r_aa_s = random.uniform(sp.r_aa_s_down, sp.r_aa_s_up)
    r_aa_a = 2 * r_aa_s
    attacker_para = sp.rAttswm, r_ad_s, r_ad_a, r_aa_s, r_aa_a, sp.k_ad, sp.k_ap, sp.k_aa
    defender_para = sp.rString, sp.r_safe, sp.r_avoid, sp.k_p, sp.k_av, sp.k_v2u
    att_swm_para = sp.max_va, sp.max_ua, sp.att_num, attacker_para
    str_net_para = sp.max_vd, sp.max_ud, sp.alpha_bound, sp.dff_num, defender_para
    env = Herding_Env.herding_env(str_net_para, att_swm_para)

    state_low = sp.state_bound[:, 0]
    state_high = sp.state_bound[:, 1]
    action_low = sp.action_bound[:, 0]
    action_high = sp.action_bound[:, 1]
    state_dim = state_low.shape[0]
    action_dim = action_low.shape[0]
    agent = DDPGAgent(state_dim, action_dim, action_low, action_high, state_low, state_high)
    agent.actor.load_state_dict(torch.load("Episode/actor_8000.pth"))
    agent.actor.eval()

    import pickle
    log_data = []
    for ep in range(1):
        if not plt.fignum_exists(fig.number):
            break

        r_ad_s = random.uniform(sp.r_ad_s_down, sp.r_ad_s_up)
        r_ad_a = random.uniform(sp.r_ad_a_down, sp.r_ad_a_up)
        r_aa_s = random.uniform(sp.r_aa_s_down, sp.r_aa_s_up)
        r_aa_a = 2 * r_aa_s
        attacker_para = sp.rAttswm, r_ad_s, r_ad_a, r_aa_s, r_aa_a, sp.k_ad, sp.k_ap, sp.k_aa
        defender_para = sp.rString, sp.r_safe, sp.r_avoid, sp.k_p, sp.k_av, sp.k_v2u
        att_swm_para = sp.max_va, sp.max_ua, sp.att_num, attacker_para
        str_net_para = sp.max_vd, sp.max_ud, sp.alpha_bound, sp.dff_num, defender_para
        env = Herding_Env.herding_env(str_net_para, att_swm_para)
        env.reset()

        # for i, defender in enumerate(env.string_net.defenders):
        #     defender.p = sp.dff_pos[i].copy()
        # env.string_net.dff_pos = sp.dff_pos.copy()
        # for i, attacker in enumerate(env.attack_swarm.attackers):
        #     attacker.p = sp.att_pos[i].copy()
        # env.attack_swarm.att_pos = sp.att_pos.copy()

        success = False
        ep_log = []
        ep_reward = 0
        traj_att = []
        traj_dff = []
        str_strings = []
        for i in range(sp.dff_num):
            ax.scatter(env.string_net.defenders[i].p[0], env.string_net.defenders[i].p[1], c='darkgrey', s=20.0)
        for i in range(sp.att_num):
            ax.scatter(env.attack_swarm.attackers[i].p[0], env.attack_swarm.attackers[i].p[1], c='darkgrey', s=20.0)
        for t in range(2010):
            if env.game_finish == 0:
                s = env.global_trans_to_local()
                a = agent.actor(s)
                a_np = a.detach().cpu().numpy()   # + np.random.normal(0, 0.1, size=action_dim)
                a_np[-2] = a_np[-2] / 8 * sp.dff_num
                if sp.max_vd > sp.max_va:
                    a_np[:2] = a_np[:2] / sp.max_vd * sp.max_va
            elif env.game_finish == 2 and not success:
                a_np = np.zeros([5])
                actual_cen = env.string_net.actual_cen
                phi = env.string_net.phi
                beta = env.string_net.beta
                zeta = env.string_net.zeta
                a_np[:2] = np.array([sp.goal_area_x, sp.goal_area_y]) - actual_cen
                a_np[:2] = a_np[:2] / np.linalg.norm(a_np[:2]) * min(sp.max_vd, sp.max_va) * 0.75
                # a_np[2] = np.arctan2(a_np[1], a_np[0]) - phi
                a_np[3] = 1.2 - zeta
                a_np[4] = (np.pi * 2.0) / sp.dff_num * (sp.dff_num - 1) - beta
                a_np[2:] = np.clip(a_np[2:], action_low[2:], action_high[2:])

                ws_model = dict()
                # robot radius
                ws_model['robot_radius'] = env.string_net.rds * 0.8  # * 2 + sp.rSafe
                # circular obstacles, format [x,y,rad]
                # with obstacles
                ws_model['circular_obstacles'] = sp.obstacles
                # rectangular boundary, format [x,y,width/2,heigth/2]
                ws_model['boundary'] = []
                a_np[:2] = RVO_update(actual_cen.reshape(1, -1), a_np[:2].reshape(1, -1), env.string_net.actual_vel.reshape(1, -1), ws_model)[0]

                # === 转成 local（逆旋转）===
                vx_local = a_np[0] * np.cos(phi) + a_np[1] * np.sin(phi)
                vy_local = -a_np[0] * np.sin(phi) + a_np[1] * np.cos(phi)
                a_np[:2] = np.array([vx_local, vy_local])

                if np.linalg.norm(actual_cen - np.array([sp.goal_area_x, sp.goal_area_y])) < 0.1:
                    success = True

            env.update(a_np.flatten())
            str_strings.append(env.string_net.strings)
            traj_att.append(env.attack_swarm.att_pos.copy())
            traj_dff.append(env.string_net.dff_pos.copy())
            for i in range(sp.att_num):
                att_traj_lines[i].set_data(
                    [step[i, 0] for step in traj_att],
                    [step[i, 1] for step in traj_att],
                )
            for i in range(sp.dff_num):
                dff_traj_lines[i].set_data(
                    [step[i, 0] for step in traj_dff],
                    [step[i, 1] for step in traj_dff],
                )

            att_pts.set_data(env.attack_swarm.att_pos[:, 0], env.attack_swarm.att_pos[:, 1])
            dff_des_pts.set_data(env.string_net.dff_pos_desired[:, 0], env.string_net.dff_pos_desired[:, 1])
            dff_pts.set_data(env.string_net.dff_pos[:, 0], env.string_net.dff_pos[:, 1])
            # shape_cen = env.string_net.actual_cen.reshape(1, -1)
            # dff_cen.set_data(shape_cen[:, 0], shape_cen[:, 1])

            x_data = []
            y_data = []
            for seg in env.string_net.strings:
                x_data += [seg[0, 0], seg[1, 0], np.nan]
                y_data += [seg[0, 1], seg[1, 1], np.nan]
            strings.set_data(x_data, y_data)

            plt.pause(0.001)
            if env.game_finish == 1:
                file = 'data/our/' + str(sp.mode) + '/'
                os.makedirs(os.path.dirname(file), exist_ok=True)
                np.save(file + 'result.npy', np.array(True))
                np.save(file + 'traj_att.npy', np.array(traj_att))
                np.save(file + 'traj_dff.npy', np.array(traj_dff))
                np.save(file + 'obstacle.npy', np.array(sp.obstacles))
                np.save(file + 'goal_area.npy', np.array([sp.goal_area_x, sp.goal_area_y]))
                np.save(file + 'strings.npy', np.array(str_strings, dtype=object))

                break
            elif env.game_finish == 2:
                if success:
                    file = 'data/our/'+str(sp.mode)+'/'
                    os.makedirs(os.path.dirname(file), exist_ok=True)
                    np.save(file + 'result.npy', np.array(True))
                    np.save(file + 'traj_att.npy', np.array(traj_att))
                    np.save(file + 'traj_dff.npy', np.array(traj_dff))
                    np.save(file + 'obstacle.npy', np.array(sp.obstacles))
                    np.save(file + 'goal_area.npy', np.array([sp.goal_area_x, sp.goal_area_y]))
                    np.save(file + 'strings.npy', np.array(str_strings, dtype=object))

                    break
            elif env.game_finish == 3:
                break

    plt.ioff()
    plt.tight_layout()
    plt.show()