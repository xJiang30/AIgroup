## A custom navigation environment built on top of Gymnasium's Ant-v5

### You can change:
- goal: [x, y]
- obstacles: [] or [(x ,y, radius),...]
- waypoints: [(x, y),...]
- reward cfg including{
        "progress": 4.0,
        "collision": -2.0,
        "waypoint": 3.0,
        "control": -0.05,
        "goal": 6.0,
        "alive": 0.05,
    } 
- Logic of updating rewards for different tasks (if you think is better)

### How to create a instance for the class **AntNavigationEnv**:

#### 1. If you want to run multiple environments in parallel for efficient training

```python

def make_env_fn(rank: int = 0, seed: int = 42):
    def _init():
        env = AntNavigationEnv(
            goal=[5, 5],     
            obstacles=[],    
            waypoints=[],    
            render_mode=None
        )
        env.reset(seed=seed + rank)
        return env
    return _init

envs = SubprocVecEnv([make_env_fn(rank=i) for i in range(n_envs)])

```

#### 2. If you want to run single environment
````python
'''keep same function above'''

envs = env = make_env_fn(rank=0)() #change here
````

