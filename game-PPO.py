
#for wsl1 visualization
import os
os.environ['DISPLAY'] = 'localhost:0.0'

import gym
from stable_baselines3 import PPO

models_dir = "models/PPO"

env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2
env.reset()

model_path = f"{models_dir}/270000.zip"
model = PPO.load(model_path, env=env)
"""
The Proximal Policy Optimization algorithm combines ideas from A2C (having multiple workers)
and TRPO (it uses a trust region to improve the actor).
"""

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)