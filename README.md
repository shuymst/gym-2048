# gym-2048
Gymnasium environment for the game 2048.

## Installation
```bash
git clone git@github.com:shuymst/gym-2048.git
poetry add ./gym-2048 --without test
```

## API
```python
import gymnasium as gym
import gym_2048
import random

env = gym.make("TwentyFortyeight-v0")
observation, info = env.reset(seed=42)
while True:
    action = random.choice(info['legal actions'])
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        env.render()
        break
env.close()

"""
-----------------------------
|     4|     2|    16|     2|
-----------------------------
|     8|    32|     8|     4|
-----------------------------
|    16|    64|   128|     8|
-----------------------------
|     2|     4|    32|     4|
-----------------------------
score: 1440
"""
```
