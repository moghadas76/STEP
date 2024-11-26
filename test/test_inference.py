import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../.."))
from argparse import ArgumentParser

from easytorch import launch_runner, Runner


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        
        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head - outputs mean and log_std for continuous actions
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value head for critic
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.feature_net(state)
        
        # Get action distribution parameters
        action_mean = self.policy_mean(features)
        action_log_std = self.policy_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        # Get value estimate
        value = self.value_head(features)
        
        return action_mean, action_std, value

    def evaluate_actions(self, state, action):
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().mean()
        
        return action_log_probs, value, entropy

    def act(self, state):
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        return action, action_log_prob, value

class PPOTrader:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.2,
        c1=1.0,
        c2=0.01
    ):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # clipping parameter
        self.c1 = c1  # value loss coefficient
        self.c2 = c2  # entropy coefficient
        
    def update(self, trajectories):
        """
        Update policy using the PPO algorithm
        
        Args:
            trajectories: List of (state, action, reward, next_state, done, old_log_prob)
        """
        states = torch.FloatTensor([t[0] for t in trajectories])
        actions = torch.FloatTensor([t[1] for t in trajectories])
        rewards = torch.FloatTensor([t[2] for t in trajectories])
        next_states = torch.FloatTensor([t[3] for t in trajectories])
        dones = torch.FloatTensor([t[4] for t in trajectories])
        old_log_probs = torch.FloatTensor([t[5] for t in trajectories])
        
        # Calculate returns and advantages
        with torch.no_grad():
            _, _, next_values = self.policy(next_states)
            _, _, values = self.policy(states)
            
            returns = []
            advantages = []
            gae = 0
            for r, d, val, next_val in zip(
                reversed(rewards),
                reversed(dones),
                reversed(values),
                reversed(next_values)
            ):
                delta = r + self.gamma * next_val * (1 - d) - val
                gae = delta + self.gamma * 0.95 * (1 - d) * gae
                returns.insert(0, gae + val)
                advantages.insert(0, gae)
                
            returns = torch.FloatTensor(returns)
            advantages = torch.FloatTensor(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(10):  # number of policy update iterations
            # Get current policy distributions
            new_log_probs, value_pred, entropy = self.policy.evaluate_actions(states, actions)
            
            # Calculate policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = 0.5 * (returns - value_pred).pow(2).mean()
            
            # Total loss
            loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

    def train_episode(self, env):
        """
        Train for one episode
        
        Args:
            env: Trading environment that implements gym-like interface
        """
        state = env.reset()
        done = False
        trajectories = []
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, _ = self.policy.act(state_tensor)
            
            next_state, reward, done, _ = env.step(action.numpy()[0])
            
            trajectories.append((state, action.numpy()[0], reward, 
                               next_state, done, log_prob.item()))
            
            state = next_state
            
            if done:
                self.update(trajectories)
                break
        
        return sum(t[2] for t in trajectories)  # Return total episode reward

# Example usage:
"""
# Initialize environment and trader
env = TradingEnvironment()  # Your custom trading environment
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

trader = PPOTrader(state_dim=state_dim, action_dim=action_dim)

# Training loop
n_episodes = 1000
for episode in range(n_episodes):
    episode_reward = trader.train_episode(env)
    print(f"Episode {episode + 1}, Total Reward: {episode_reward}")
"""

def parse_args():
    parser = ArgumentParser(description='Welcome to EasyTorch!')
    parser.add_argument('-c', '--cfg', default="step/STEP_METR-LA.py", help='training config')
    parser.add_argument('--ckpt', default="checkpoints/STEP_100/4831df1c147dd7dbb643ef143092743d/STEP_best_val_MAE.pt", help='ckpt path. if it is None, load default ckpt in ckpt save dir', type=str)
    parser.add_argument("--gpus", default="0", help="visible gpus")
    return parser.parse_args()


def main(cfg: dict, runner: Runner, ckpt: str = None):
    # init logger
    runner.init_logger(logger_name='easytorch-inference', log_file_name='validate_result')

    runner.load_model(ckpt_path=ckpt)

    runner.test_process(cfg)


if __name__ == '__main__':
    args = parse_args()
    try:
        launch_runner(args.cfg, main, (args.ckpt,), devices=args.gpus)
    except TypeError as e:
        if "launch_runner() got an unexpected keyword argument" in repr(e):
            # NOTE: for earlier easytorch version
            launch_runner(args.cfg, main, (args.ckpt,), gpus=args.gpus)
        else:
            raise e
