from agent import Agent
from monitor import interact
import gym
import numpy as np
import time,datetime

env = gym.make('Taxi-v2')
agent = Agent()
start_time = time.time()
# avg_rewards, best_avg_reward = interact(env, agent)
avg_rewards, best_avg_reward = interact(env, agent, num_episodes=1000000)
elapsed_time = time.time() - start_time
print(str(datetime.timedelta(seconds=elapsed_time)))
'''
self.epsilon = 0.005
self.gamma=1.0
self.alpha=0.01

9/22/18 20000 score = 9.21
100,000 score = 9.32 at 
Episode 21000/100000 || Best average reward 9.06
Episode 22000/100000 || Best average reward 9.06
Episode 23000/100000 || Best average reward 9.08
Episode 24000/100000 || Best average reward 9.08
Episode 25000/100000 || Best average reward 9.13
Episode 26000/100000 || Best average reward 9.13
Episode 27000/100000 || Best average reward 9.32
Episode 28000/100000 || Best average reward 9.32
agent.epsilon = 1.0/i_episode
Episode 1000/100000  || Best average reward -210.66
Episode 14000/100000 || Best average reward 9.02
Episode 19000/100000 || Best average reward 9.15
Episode 22000/100000 || Best average reward 9.34
Episode 38000/100000 || Best average reward 9.38
Episode 100000/100000 || Best average reward 9.38
0:02:18.542739

self.epsilon = 0.005
self.gamma = 0.8 # 1.0
self.alpha = 0.07 # 0.01
Episode 1000/100000 || Best average reward -16.61
Episode 2000/100000 || Best average reward 7.31
Episode 3000/100000 || Best average reward 8.42
Episode 4000/100000 || Best average reward 8.52
Episode 5000/100000 || Best average reward 8.97
Episode 8000/100000 || Best average reward 9.17
Episode 26000/100000 || Best average reward 9.27
Episode 97000/100000 || Best average reward 9.31
0:01:48.531079

agent.epsilon = 1.0/i_episode
self.gamma = 0.8 # 1.0
self.alpha = 0.07 # 0.01
Episode 1000/1000000 || Best average reward -22.01
Episode 2000/1000000 || Best average reward 7.2
Episode 3000/1000000 || Best average reward 8.66
Episode 4000/1000000 || Best average reward 8.81
Episode 5000/1000000 || Best average reward 9.11
Episode 8000/1000000 || Best average reward 9.25
Episode 33000/1000000 || Best average reward 9.31
Episode 87000/1000000 || Best average reward 9.32
Episode 270000/1000000 || Best average reward 9.36
Episode 306000/1000000 || Best average reward 9.52
Episode 307000/1000000 || Best average reward 9.52
Episode 308000/1000000 || Best average reward 9.52
Episode 309000/1000000 || Best average reward 9.52
Episode 310000/1000000 || Best average reward 9.53
Episode 1000000/1000000 || Best average reward 9.53
0:17:11.886009
'''