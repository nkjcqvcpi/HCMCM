import gym
from old_api import sim
import math
import numpy as np
from gym import spaces
from gym.utils import seeding

lr = 0.005
base_name = 'Sawyer'
joint_name = 'Sawyer_joint'


class Robot(gym.Env):
    metadata = {'render.modes': 'ansi'}

    def __init__(self):
        self.seed()
        self.target = [1.5, 0, 0.655]
        # (jpos_2, jpos_4, jpos_6, jvel_g, pos_x, pos_y, pos_z, rot_a, rot_b, rot_g)
        self.default_state = (-90, 0, 0, 5, -1.9189, 0, 1.3255, 0, 0, 0)
        self.low = np.array(
            [-218.8, -174.8, -171.0, -5, -2.5000, -2.5000, +0.8000, -179.9, -179.9, -179.9],
            dtype=np.float16
        )
        self.high = np.array(
            [+130.7, +174.9, +171.0, +5, +2.5000, +2.5000, +0.0000, +179.9, +179.9, +179.9],
            dtype=np.float16
        )
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float16)
        self.action_space = spaces.Discrete(4)
        self.bottle_stand = -1

        sim.simxFinish(-1)
        self.clientID = sim.simxStart('127.0.0.1', 20001, True, True, 5000, 5)
        if self.clientID != -1:
            print('Connected to remote API server :' + str(self.clientID))
        else:
            print('Failed connecting to remote API server')
            exit(1)
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)
        sim.simxSetFloatingParameter(self.clientID, sim.sim_floatparam_simulation_time_step, lr,
                                     sim.simx_opmode_oneshot)
        sim.simxSynchronous(self.clientID, True)
        # sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)

        _, self.baseHandle = sim.simxGetObjectHandle(self.clientID, base_name, sim.simx_opmode_blocking)
        self.sawyerHandle = {}
        for j in range(2, 7, 2):
            _, self.sawyerHandle[j] = sim.simxGetObjectHandle(self.clientID, joint_name + str(j),
                                                              sim.simx_opmode_blocking)
        _, self.gripperHandle = sim.simxGetObjectHandle(self.clientID, 'BaxterGripper_closeJoint',
                                                        sim.simx_opmode_blocking)
        _, self.bottleHandle = sim.simxGetObjectHandle(self.clientID, 'Cylinder', sim.simx_opmode_blocking)

        _, bottle_pos = sim.simxGetObjectPosition(self.clientID, self.bottleHandle, -1, sim.simx_opmode_streaming)
        _, bottle_rot = sim.simxGetObjectOrientation(self.clientID, self.bottleHandle, -1, sim.simx_opmode_streaming)
        self.state = [
            sim.simxGetJointPosition(self.clientID, self.sawyerHandle[2], sim.simx_opmode_streaming)[1],
            sim.simxGetJointPosition(self.clientID, self.sawyerHandle[4], sim.simx_opmode_streaming)[1],
            sim.simxGetJointPosition(self.clientID, self.sawyerHandle[6], sim.simx_opmode_streaming)[1],
            sim.simxGetJointPosition(self.clientID, self.gripperHandle, sim.simx_opmode_streaming)[1],
            bottle_pos[0], bottle_pos[1], bottle_pos[2],
            bottle_rot[0], bottle_rot[1], bottle_rot[2]
        ]

        _, self.sensorHandle = sim.simxGetObjectHandle(self.clientID, 'BaxterGripper_attachPoint', sim.simx_opmode_blocking)

        sim.simxSetJointTargetVelocity(self.clientID, self.gripperHandle, 5, sim.simx_opmode_oneshot)

    def step(self, action: list):
        sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)
        time_start = sim.simxGetLastCmdTime(self.clientID)
        sawyer = {2: action[0], 4: action[1], 6: action[2]}
        # [jpos_sawyer2, jpos_sawyer4, jpos_sawyer6, jpos_gripper]
        gripper = action[3]

        sim.simxSynchronousTrigger(self.clientID)
        for j in self.sawyerHandle.items():
            sim.simxSetJointTargetPosition(self.clientID, j[1], math.radians(sawyer[j[0]]), sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(self.clientID, self.gripperHandle, gripper, sim.simx_opmode_oneshot)
        if gripper==-5:
            sim.simxSetObjectParent(self.clientID, self.bottleHandle, -1, True, sim.simx_opmode_oneshot)
        sim.simxGetPingTime(self.clientID)

        for i, j in enumerate(self.sawyerHandle.values()):
            _, self.state[i] = sim.simxGetJointPosition(self.clientID, j, sim.simx_opmode_buffer)
        _, self.state[3] = sim.simxGetJointPosition(self.clientID, self.gripperHandle, sim.simx_opmode_buffer)
        _, self.state[4:7] = sim.simxGetObjectPosition(self.clientID, self.bottleHandle, -1, sim.simx_opmode_buffer)
        _, self.state[7:10] = sim.simxGetObjectOrientation(self.clientID, self.bottleHandle, -1, sim.simx_opmode_buffer)

        reward = -(sim.simxGetLastCmdTime(self.clientID) - time_start) / 1000

        distance = np.linalg.norm(np.array(self.target) - np.array([self.state[4:7]]))
        reward += 100 - distance

        if gripper != 0:
            if self.state[7] == 0 and self.state[8] == 0:
                self.bottle_stand = 1
                reward += 100
            else:
                self.bottle_stand = 0
                reward -= 100
            done = True
        else:
            done = False
        info = {}
        return np.array(self.state), reward, done, info

    def reset(self):
        # sim.simxStopSimulation(self.clientID, sim.simx_opmode_oneshot)
        self.bottle_stand = -1
        sim.simxSynchronousTrigger(self.clientID)
        for i, j in enumerate(self.sawyerHandle.values()):
            sim.simxSetJointTargetPosition(self.clientID, j, math.radians(self.default_state[i]), sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(self.clientID, self.gripperHandle, self.default_state[3], sim.simx_opmode_oneshot)

        return np.array(self.state)

    def render(self, mode='ansi'):
        for j in self.sawyerHandle:
            _, pos = sim.simxGetJointPosition(self.clientID, j, sim.simx_opmode_buffer)
            print('s' + str(j) + ' = ' + str(round(math.degrees(pos), 2)), end=' ')
        print('g = ' + str(sim.simxGetJointPosition(self.clientID, self.gripperHandle, sim.simx_opmode_buffer)[1]))
        print('b = ' + str(sim.simxGetJointPosition(self.clientID, self.bottleHandle, sim.simx_opmode_buffer)[1]))
        print()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        sim.simxStopSimulation(self.clientID, sim.simx_opmode_oneshot)
        sim.simxFinish(self.clientID)


if __name__ == '__main__':
    r = Robot()
    joint = [-90, 0, -30, 5]
    obs, rew, d, inf = r.step(joint)
    joint = [-90, 0, 0, 5]
    obs, rew, d, inf = r.step(joint)
    joint = [-90, 0, 30, 5]
    obs, rew, d, inf = r.step(joint)
    joint = [-90, 0, 30, -5]
    obs, rew, d, inf = r.step(joint)
    r.close()
    i = 0
