import gym
import b0RemoteApi
import math
import numpy as np
from gym import spaces
from gym.utils import seeding
from torch.utils.tensorboard import SummaryWriter
import os

lr = 1


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class Robot(gym.GoalEnv):
    """
    创建一个环境（environment），用于与仿真程序交互，判断是否完成目标
    """
    metadata = {'render.modes': 'ansi'}

    def __init__(self):
        self.tensorboard: SummaryWriter
        self.seed()  # 定义随机数生成的种子
        self.distance_threshold = 0.5  # 投掷后落入的范围，即设(1，1)为目标点，半径为0.5范围内都算投中
        self.default_state = (-90, 0, 0, 0, -1.9189, 0, 1.3255, 0, 0)  # 初始状态
        self.low = np.array([-3.819, -3.051, -2.985, 0, -2.5000, -2.5000, +0.8000, -3.140, -3.140], dtype=np.float16)
        self.high = np.array([+2.281, +3.052, +2.984, 1, +2.5000, +2.5000, +0.0000, +3.139, +3.139],
                             dtype=np.float16)  # 设定观测参数的上下界

        self.goal = self._sample_goal()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(self.low, self.high, dtype=np.float16),
            achieved_goal=spaces.Box(self.low, self.high, dtype=np.float16),
            observation=spaces.Box(self.low, self.high, dtype=np.float16),
        ))
        self.action_space = spaces.Box(np.array([-3.819, -3.051, -2.985, 0], dtype=np.float16),
                                       np.array([+2.281, +3.052, +2.984, 1], dtype=np.float16), dtype='float32')

        self.client = b0RemoteApi.RemoteApiClient('b0RemoteApi_V-REP', 'b0RemoteApi', 60, timeout=60)
        _, self.baseHandle = self.client.simxGetObjectHandle('Sawyer', self.client.simxServiceCall())
        self.sawyerHandle = {}
        for j in range(2, 7, 2):
            _, self.sawyerHandle[j] = self.client.simxGetObjectHandle('Sawyer_joint' + str(j),
                                                                      self.client.simxServiceCall())
        _, self.gripperHandle = self.client.simxGetObjectHandle('BaxterGripper_closeJoint',
                                                                self.client.simxServiceCall())
        _, self.bottleHandle = self.client.simxGetObjectHandle('Cylinder', self.client.simxServiceCall())
        _, self.floorHandle = self.client.simxGetObjectHandle('Floor', self.client.simxServiceCall())
        _, self.sensorHandle = self.client.simxGetObjectHandle('BaxterGripper_attachPoint',
                                                               self.client.simxServiceCall())
        _, self.deskHandle = self.client.simxGetObjectHandle('customizableTable', self.client.simxServiceCall())
        self.client.simxSetJointTargetVelocity(self.gripperHandle, -5, self.client.simxDefaultPublisher())
        self.start_time = self.client.simxGetServerTimeInMs(self.client.simxServiceCall())[1]
        self.time = self.client.simxGetServerTimeInMs(self.client.simxServiceCall())[1]
        self.obs = self._get_obs()  # 沟通仿真软件的部分

    def step(self, action):
        """
        执行每一步动作
        动作指令包含：
        【机械臂2角度，机械臂4角度，机械臂6角度，投掷指令】
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        try:
            time = self.time - self.start_time
            self.tensorboard.add_scalars('Action/Sawyer',
                                         {'joint2': action[0], 'joint4': action[1], 'joint6': action[2]}, time)
            self.tensorboard.add_scalar('Action/Gripper', action[3], time)
        except Exception:
            pass
        self.client.simxStartSimulation(self.client.simxDefaultPublisher())
        self._set_action(action)
        # [jpos_sawyer2, jpos_sawyer4, jpos_sawyer6, jpos_gripper]

        self.obs = self._get_obs()
        info = {'is_success': self._is_success(self.obs['achieved_goal'], self.goal), 'gripper': action[3] >= 0.5,
                'action': action}
        reward = self.compute_reward(self.obs['achieved_goal'], self.goal, info)
        if action[3] >= 0.5 or \
                self.client.simxCheckCollision(self.floorHandle, self.bottleHandle, self.client.simxServiceCall())[1]:
            done = True
            self.time = self.client.simxGetServerTimeInMs(self.client.simxServiceCall())[1]
            self.client.simxSleep(2)
        else:
            done = False
        # 判断本次仿真是否结束，条件一：水杯抛掷出去了，条件二：水杯落地（即与地面发生碰撞）

        self.client.simxPauseSimulation(self.client.simxDefaultPublisher())
        return self.obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        计算奖赏，影响因素有：
        每次仿真的时间（-）
        与目标点的距离（--）
        抛出水瓶（+）
        水瓶直立（++）
        与目标点重合（++）
        机械臂运动幅度（+）
        水瓶落地（--）
        """
        reward = -(self.client.simxGetServerTimeInMs(self.client.simxServiceCall())[1] - self.time) / 1000
        distance = goal_distance(self.goal[:3], np.array([-1.9189, 0, 1.3255]))
        reward += distance - goal_distance(self.goal[:3], achieved_goal[:3])

        if info['gripper']:
            reward += 25
            if achieved_goal[3] == 0 and achieved_goal[4] == 0:
                reward += 50
                if distance == 0:
                    reward += 50
        reward += math.sqrt(info['action'][0] ** 2 + info['action'][1] ** 2 + info['action'][2] ** 2)
        if self.client.simxCheckCollision(self.floorHandle, self.bottleHandle, self.client.simxServiceCall())[1]:
            reward -= 50
        elif self.client.simxCheckCollision(self.deskHandle, self.bottleHandle, self.client.simxServiceCall())[1]:
            reward += 50
        try:
            self.tensorboard.add_scalar('Reward', reward, self.time - self.start_time)
        except Exception:
            pass
        return reward

    def reset(self):  # 重置环境
        self.client.simxStopSimulation(self.client.simxDefaultPublisher())
        self.obs = self._get_obs()
        return self.obs

    def render(self, mode='ansi'):
        for j in self.sawyerHandle:
            _, pos = self.client.simxGetJointPosition(j, self.client.simxServiceCall())
            print('s' + str(j) + ' = ' + str(round(math.degrees(pos), 2)), end=' ')
        print('g = ' + str(self.client.simxGetJointPosition(self.gripperHandle, self.client.simxServiceCall())[1]))
        print('b = ' + str(self.client.simxGetJointPosition(self.bottleHandle, self.client.simxServiceCall())[1]))
        print()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        self.client.simxStopSimulation(self.client.simxDefaultPublisher())

    def _get_obs(self):
        """
        获得环境信息
        可观测信息：
        【机械臂2角度，机械臂4角度，机械臂6角度，是否投掷，水瓶x,y,z坐标, 水瓶A,B倾角】
        """
        # positions
        sawyer_pos = np.array([self.client.simxGetJointPosition(self.sawyerHandle[2], self.client.simxServiceCall())[1],
                               self.client.simxGetJointPosition(self.sawyerHandle[4], self.client.simxServiceCall())[1],
                               self.client.simxGetJointPosition(self.sawyerHandle[6], self.client.simxServiceCall())[
                                   1]])
        gripper_state = np.array(
            [self.client.simxGetJointPosition(self.gripperHandle, self.client.simxServiceCall())[1]])
        bottle_pos = np.array(
            self.client.simxGetObjectPosition(self.bottleHandle, -1, self.client.simxServiceCall())[1])
        bottle_rot = np.array(
            self.client.simxGetObjectOrientation(self.bottleHandle, -1, self.client.simxServiceCall())[1][:2])

        achieved_goal = np.concatenate([bottle_pos.copy(), np.squeeze(bottle_rot[:2].copy())])
        # (pos_x, pos_y, pos_z, rot_a, rot_b)
        obs = np.concatenate([sawyer_pos, gripper_state, bottle_pos.ravel(), bottle_rot.ravel()])
        # (jpos_2, jpos_4, jpos_6, jvel_g, pos_x, pos_y, pos_z, rot_a, rot_b)
        try:
            time = self.time - self.start_time
            self.tensorboard.add_scalars('Observation/Sawyer',
                                         {'joint2': sawyer_pos[0], 'joint4': sawyer_pos[1], 'joint6': sawyer_pos[2]},
                                         time)
            self.tensorboard.add_scalar('Observation/Gripper', gripper_state, time)
            self.tensorboard.add_scalars('Achieved Goal/Bottle Position',
                                         {'x': bottle_pos[0], 'y': bottle_pos[1], 'z': bottle_pos[2]}, time)
            self.tensorboard.add_scalars('Achieved Goal/Bottle Rotation', {'A': bottle_rot[0], 'B': bottle_rot[1]}, time)
        except Exception:
            pass
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    @staticmethod
    def _sample_goal():  # 理想结果
        goal = np.array([0, 0, 0.655, 0, 0])
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):  # 判断水杯落点是否在制定范围内
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _set_action(self, action):  # 执行动作
        assert action.shape == (4,)
        action = action.tolist()
        sawyer, gripper_ctrl = action[:3], action[3]
        for i, j in enumerate(self.sawyerHandle.values()):
            self.client.simxSetJointTargetPosition(j, sawyer[i], self.client.simxDefaultPublisher())
        if gripper_ctrl >= 0.5:
            self.client.simxSetObjectParent(self.bottleHandle, -1, True, False, self.client.simxDefaultPublisher())


if __name__ == '__main__':
    r = Robot()
    joint = np.array([math.radians(-90), 0, math.radians(45), 1])
    observation, rew, don, inf = r.step(joint)
    # r.close()
    os.system("pause")
