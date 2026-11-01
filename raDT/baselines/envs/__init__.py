from gymnasium.envs.registration import register


def register_envs():
    register(id='PointMazeObstacle-v0',
             entry_point='raDT.baselines.envs.point_maze_obstacle:PointMazeObstacleEnv',
             max_episode_steps=1000)

    register(id='FetchReachObstacle-v0',
             entry_point='raDT.baselines.envs.reach_obstacle:FetchReachObstacleEnv',
             max_episode_steps=50)
