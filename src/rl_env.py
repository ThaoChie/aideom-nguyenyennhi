"""Reinforcement Learning Environment for Vietnam Economy."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class VietnamEconomyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 5 actions:
        # 0: Truyền thống (K 70%)
        # 1: Cân bằng (K 40%, D 25%, AI 15%, H 20%)
        # 2: Số hóa nhanh (K 25%, D 45%, AI 15%, H 15%)
        # 3: AI dẫn dắt (K 20%, D 20%, AI 45%, H 15%)
        # 4: Bao trùm (K 30%, D 20%, AI 10%, H 40%)
        self.action_space = spaces.Discrete(5)
        
        # 4 observations: GDP_growth (0-2), Digital_index (0-2), AI_capacity (0-2), Unemp_risk (0-2)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
        self.T = 10
        
        self.allocation = {
            0: np.array([0.70, 0.10, 0.10, 0.10]),
            1: np.array([0.40, 0.25, 0.15, 0.20]),
            2: np.array([0.25, 0.45, 0.15, 0.15]),
            3: np.array([0.20, 0.20, 0.45, 0.15]),
            4: np.array([0.30, 0.20, 0.10, 0.40])
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Default state: Vietnam 2026 actual (medium, medium, low, medium)
        self.state = np.array([1, 1, 0, 1])
        if options and 'initial_state' in options:
            self.state = np.array(options['initial_state'])
            
        self.t = 0
        self.K = 27500
        self.D = 20.3
        self.AI = 86
        self.H = 30
        return self.state, {}

    def step(self, action):
        a = self.allocation[action]
        budget = 1000
        
        self.K += a[0]*budget
        self.D += a[1]*budget/100
        self.AI += a[2]*budget/20
        self.H += a[3]*budget/200
        
        # Cobb-Douglas output Y
        Y = (self.K**0.33) * (54.0**0.42) * (self.D**0.10) * (self.AI**0.08) * (self.H**0.07)
        reward = Y / 10000.0
        
        # Adjust reward based on unemployment risk state
        unemp_risk = self.state[3]
        if unemp_risk == 2:  # high risk
            reward -= 0.5  # penalty
            
        # Update states pseudo-randomly but influenced by action
        new_state = self.state.copy()
        # AI capacity improves if we pick AI-focused action (3)
        if action == 3 and np.random.rand() < 0.6:
            new_state[2] = min(2, new_state[2] + 1)
        # Digital improves if action 2
        if action == 2 and np.random.rand() < 0.6:
            new_state[1] = min(2, new_state[1] + 1)
        # Unemp risk increases if AI is high and H is low
        if action == 3 and new_state[3] < 2 and np.random.rand() < 0.5:
            new_state[3] += 1
        # Unemp risk decreases if H action (4)
        if action == 4 and new_state[3] > 0 and np.random.rand() < 0.7:
            new_state[3] -= 1
            
        self.state = new_state
        self.t += 1
        done = self.t >= self.T
        
        return self.state, float(reward), done, False, {}

def solve_bai11(learning_rate=0.1, discount_factor=0.95, episodes=10000, use_dqn=True):
    env = VietnamEconomyEnv()
    
    # 1. Tabular Q-Learning
    Q = np.zeros((3, 3, 3, 3, 5))
    q_rewards = []
    
    eps_start = 1.0
    eps_end = 0.05
    
    for ep in range(episodes):
        s, _ = env.reset()
        total_reward = 0
        eps = eps_start - (eps_start - eps_end) * (ep / episodes)
        
        while True:
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[tuple(s)]))
                
            s_next, r, done, _, _ = env.step(action)
            total_reward += r
            
            best_next_a = np.argmax(Q[tuple(s_next)])
            td_target = r + discount_factor * Q[tuple(s_next)][best_next_a]
            td_error = td_target - Q[tuple(s)][action]
            Q[tuple(s)][action] += learning_rate * td_error
            
            s = s_next
            if done:
                break
        q_rewards.append(total_reward)
        
    # Smoothing Q-learning learning curve
    window = max(1, episodes // 100)
    q_smoothed = [np.mean(q_rewards[i:i+window]) for i in range(0, len(q_rewards), window)]

    # 2. Extract policy for 5 different states
    policy_desc = ["Truyền thống", "Cân bằng", "Số hóa nhanh", "AI dẫn dắt", "Bao trùm"]
    test_states = {
        "VN 2026 (1,1,0,1)": [1, 1, 0, 1],
        "Tụt hậu (0,0,0,2)": [0, 0, 0, 2],
        "Khởi sắc (2,2,1,0)": [2, 2, 1, 0],
        "Khủng hoảng LĐ (1,1,2,2)": [1, 1, 2, 2],
        "Phát triển AI (2,2,2,0)": [2, 2, 2, 0]
    }
    extracted_policies = {}
    for name, st in test_states.items():
        best_a = int(np.argmax(Q[tuple(st)]))
        extracted_policies[name] = policy_desc[best_a]

    # 3. Rule-based evaluation (always a1, always a3, random)
    def eval_policy(policy_type, evals=100):
        tot = 0
        for _ in range(evals):
            env.reset()
            d = False
            while not d:
                if policy_type == 'a1': a = 1
                elif policy_type == 'a3': a = 3
                elif policy_type == 'random': a = env.action_space.sample()
                else: a = int(np.argmax(Q[tuple(env.state)]))
                _, r, d, _, _ = env.step(a)
                tot += r
        return tot / evals
        
    rules_perf = {
        'Q-Learning': eval_policy('q'),
        'Luôn chọn A1': eval_policy('a1'),
        'Luôn chọn A3': eval_policy('a3'),
        'Random': eval_policy('random')
    }
    
    # 4. Deep Q-Network (DQN) with stable-baselines3
    dqn_rewards = []
    dqn_perf = 0
    if use_dqn:
        try:
            from stable_baselines3 import DQN
            from stable_baselines3.common.callbacks import BaseCallback

            class RewardLoggerCallback(BaseCallback):
                def __init__(self):
                    super().__init__(0)
                    self.rewards = []
                    self.ep_reward = 0
                def _on_step(self):
                    self.ep_reward += self.locals["rewards"][0]
                    if self.locals["dones"][0]:
                        self.rewards.append(self.ep_reward)
                        self.ep_reward = 0
                    return True

            model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=50000, 
                        learning_starts=1000, target_update_interval=500,
                        policy_kwargs=dict(net_arch=[64, 64]), verbose=0)
            cb = RewardLoggerCallback()
            model.learn(total_timesteps=20000, callback=cb)
            
            dqn_smoothed = [np.mean(cb.rewards[i:i+max(1, len(cb.rewards)//100)]) for i in range(0, len(cb.rewards), max(1, len(cb.rewards)//100))]
            dqn_rewards = dqn_smoothed
            
            tot = 0
            for _ in range(100):
                obs, _ = env.reset()
                d = False
                while not d:
                    a, _ = model.predict(obs, deterministic=True)
                    obs, r, d, _, _ = env.step(a)
                    tot += r
            dqn_perf = tot / 100
            rules_perf['DQN'] = dqn_perf
            
        except ImportError:
            pass

    return {
        'episodes': episodes,
        'q_smoothed': q_smoothed,
        'dqn_smoothed': dqn_rewards,
        'extracted_policies': extracted_policies,
        'rules_perf': rules_perf
    }
