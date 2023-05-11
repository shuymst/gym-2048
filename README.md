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

random.seed(42)

env = gym.make("TwentyFortyEight-v0")
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
|     2|    32|    16|     2|
-----------------------------
|    16|     2|   128|     4|
-----------------------------
|     8|    32|     4|     2|
-----------------------------
|     2|     8|    16|     4|
-----------------------------
score: 1168
"""
```
