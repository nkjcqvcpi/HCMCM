from gym import register

register(
    id='robot-v0',
    entry_point='robot.robot:Robot',
    reward_threshold=100.0,
)
