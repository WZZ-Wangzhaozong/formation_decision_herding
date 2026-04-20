import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import Herding_Env_1108 as Herding_Env
import Scene_para as sp
from matplotlib.patches import Circle
import copy
import pickle

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
        out = self.net(x)
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

        # Update Critic
        q_value = self.critic(s, a)
        critic_loss = nn.MSELoss()(q_value, q_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Update Actor
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Soft Update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def visualize_dynamic(env, agent, steps=300, delay=0.05):
    env.reset()
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(-7, 20)
    ax.set_ylim(-7, 20)
    ax.set_aspect('equal')

    for t in range(steps):
        state = env.global_trans_to_local()
        action = agent.actor(state)
        env.string_net.update(action.detach().cpu().numpy().flatten(), env.time)
        env.attack_swarm.update(env.target_area, env.string_net.strings, env.string_net.dff_pos, env.time)

        ax.clear()
        ax.set_title(f"Step {t}")
        ax.scatter(*env.target_area, c='g', s=100, label="Protect Area")
        positions = np.array(env.string_net.dff_pos)
        ax.scatter(positions[:, 0], positions[:, 1], c='b', s=50)
        ax.scatter(*env.attack_swarm.cen, c='r', s=100, label="Attacker")
        positions = np.array(env.attack_swarm.att_pos)
        ax.scatter(positions[:, 0], positions[:, 1], c='b', s=50)
        ax.scatter(*env.string_net.actual_cen, c='b', s=100, label="Defender")
        ax.legend(loc='upper right')
        plt.pause(delay)

    plt.ioff()
    plt.show()

if __name__ == "__main__":

    plt.ion()
    plt.pause(0.01)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(-5, 40)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    ax.grid(True)
    dff_cen, = ax.plot([], [], 'bo', markersize=4.0, label='Dff_Cen')
    dff_bot, = ax.plot([], [], '*', markersize=3.0, label='Dff_Cen')
    att_cen, = ax.plot([], [], 'ro', label='Dff_Cen')
    att_pts, = ax.plot([], [], 'ro', markersize=4.0, label='Attackers')
    dff_des_pts, = ax.plot([], [], 'ko', markersize=4.0)
    dff_pts, = ax.plot([], [], 'bo', markersize=3.0, label='Defenders')
    strings, = ax.plot([], [], 'b', linewidth=0.8)
    target_circle = Circle((0, 0), 2.0, color='g', fill=False, linestyle='--', linewidth=1.5)
    ax.add_patch(target_circle)

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

    log_data = []
    his_reward = []
    his_result = []
    for ep in range(10000):
        if ep >= 3000:
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
        ep_log = []
        ep_reward = 0
        for t in range(2010):
            s = env.global_trans_to_local()
            a = agent.actor(s)
            a_np = a.detach().cpu().numpy() + np.random.normal(0, 0.1, size=action_dim)
            env.update(a_np.flatten())
            if ep % 20 == 0 and ep >= 300:
                att_pts.set_data(env.attack_swarm.att_pos[:, 0], env.attack_swarm.att_pos[:, 1])
                dff_des_pts.set_data(env.string_net.dff_pos_desired[:, 0], env.string_net.dff_pos_desired[:, 1])
                dff_pts.set_data(env.string_net.dff_pos[:, 0], env.string_net.dff_pos[:, 1])
                shape_cen = env.string_net.actual_cen.reshape(1, -1)
                dff_cen.set_data(shape_cen[:, 0], shape_cen[:, 1])
                plt.pause(0.0001)

            r = env.situation_calculation()
            s_ = env.global_trans_to_local()
            d = env.done

            agent.store(s, a, r, s_, d)
            agent.train()
            s = s_
            ep_reward += r

            if ep % 1 == 0:
                ep_log.append({
                    'attcking_swarm': copy.deepcopy(env.attack_swarm),
                    'string_net': copy.deepcopy(env.string_net),
                    # 'att_pos': env.attack_swarm.att_pos.copy(),
                    # 'dff_pos': env.string_net.dff_pos.copy(),
                    # 'strings': env.string_net.strings.copy(),
                    # 'att_vel': env.attack_swarm.att_vel.copy(),
                    # 'dff_vel': env.string_net.dff_vel.copy(),
                    # 'action': a.detach().cpu().numpy().copy(),
                    # 'state': np.array([env.string_net.actual_cen[0],
                    #                    env.string_net.actual_cen[1],
                    #                    env.string_net.phi,
                    #                    env.string_net.zeta,
                    #                    env.string_net.beta])
                })

            if env.game_finish == 1:
                print(1)
                r += -2000
                ep_reward += -2000
                break
            elif env.game_finish == 2:
                print(2)
                r += 2000
                ep_reward += 2000
                break
            elif env.game_finish == 3:
                print(3)
                break

        his_reward.append(ep_reward)
        his_result.append(env.game_finish)
        # log_data.append(ep_log)
        print(f"Episode {ep}, Reward = {ep_reward:.2f}")
        if ep % 100 == 0 and ep != 0:
            torch.save(agent.actor.state_dict(), "Episode/actor_"+str(ep)+".pth")
            torch.save(agent.actor_target.state_dict(), "Episode/actor_target_"+str(ep)+".pth")
            torch.save(agent.critic.state_dict(), "Episode/critic_"+str(ep)+".pth")
            torch.save(agent.critic_target.state_dict(), "Episode/critic_target_"+str(ep)+".pth")
            np.save('Episode/his_reward_' + str(ep) + '.npy', np.array(his_reward))
            np.save('Episode/his_result_' + str(ep) + '.npy', np.array(his_result))
        with open('Episode/train_log_'+str(ep)+'.pkl', 'wb') as f:
            pickle.dump(ep_log, f)

    plt.ioff()
    plt.show()