# gym-2048
Gymnasium environment for the game 2048.

## Installation
```bash
git clone git@github.com:shuymst/gym-2048.git
poetry add ./gym-2048 --without test
```

## Usage
```python
import gymnasium as gym
import gym_2048
env = gym.make("TwentyFortyeight-v0")
env.reset()
```