
#for wsl1 visualization
import os
os.environ['DISPLAY'] = 'localhost:0.0'

import gym
from stable_baselines3 import A2C

# Create the environmentC
env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2

# required before you can step the environment
env.reset()

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

"""
A synchronous, deterministic variant of Asynchronous Advantage Actor Critic (A3C).
It uses multiple workers to avoid the use of a replay buffer.
"""
episodes = 10
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()

        # print(reward)
# # sample action:
# print("sample action:", env.action_space.sample())

# # observation space shape:
# print("observation space shape:", env.observation_space.shape)

# # sample observation:
# print("sample observation:", env.observation_space.sample())

env.close()
