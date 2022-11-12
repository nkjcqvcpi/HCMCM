import os

import gym
import tianshou as ts
from tianshou.data import Batch
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import robot

task = 'robot-v0'
lr = 1e-3  # 网络参数的学习率
gamma = 0.9
n_step = 4
eps_train, eps_test = 0.1, 0.05
num_epoch = 1000
step_per_epoch = 100  # 每一个epoch执行动作的次数
step_per_collect = 10  # 每10个动作做为一组，这一组不一定是一个完整的运动轨迹
target_freq = 320
batch_size = 64
buffer_size = 20000


class Net(nn.Module):  # 定义一个全连接神经网络
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(*[
            nn.Linear(int(np.prod(state_shape)), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, int(np.prod(action_shape)))
        ])

    def forward(self, s, state=None, info={}):
        o = s.observation
        # s.achieved_goal, s.desired_goal are also available
        if not isinstance(o, torch.Tensor):
            o = torch.tensor(o, dtype=torch.float)
        batch = o.shape[0]
        logits = self.model(o.view(batch, -1))
        return logits, state


env = gym.make(task)
env.spec.reward_threshold = 1e10
net = Net(env.observation_space.spaces['observation'].shape, env.action_space.shape)
optim = torch.optim.Adam(net.parameters(), lr=lr)  # 优化器Adam

policy = ts.policy.DQNPolicy(  # 使用DQN
    net, optim, gamma, n_step,
    target_update_freq=target_freq)
train_collector = ts.data.Collector(
    policy, env, ts.data.VectorReplayBuffer(buffer_size, 1),
    exploration_noise=True)
test_collector = ts.data.Collector(policy, env, exploration_noise=True)

if os.path.exists('dqn_best.pth') and True:
    policy.load_state_dict(torch.load('dqn_best.pth'))

writer = SummaryWriter()
env.tensorboard = writer

result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector, max_epoch=num_epoch,
    step_per_epoch=step_per_epoch, step_per_collect=step_per_collect,
    update_per_step=1 / step_per_collect, episode_per_test=1,
    batch_size=batch_size, logger=ts.utils.BasicLogger(writer),
    train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
    save_fn=torch.save(policy.state_dict(), 'dqn_best.pth'))

writer.add_hparams(
    {'Learning Rate': lr, 'num_epoch': num_epoch, 'Size/Batch': batch_size, 'Size/Buffer': buffer_size, 'Gamma': gamma,
     'eps/train': eps_train, 'eps/test': eps_test, 'step/epoch': step_per_epoch, 'Step/n': n_step,
     'step/collect': step_per_collect, 'target_freq': target_freq}, {})

print(f'Finished training! Use {result["duration"]}')

policy.load_state_dict(torch.load('dqn.pth'))
