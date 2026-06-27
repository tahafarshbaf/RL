
import numpy as np

class Env2D:
    def __init__(self,world_min=-1.0, world_max=1.0, max_steps=50, reach_threshold=0.1):
        self.world_min = world_min
        self.world_max = world_max
        self.max_steps = max_steps
        self.reach_threshold = reach_threshold
        self.position = [0.0, 0.0]
        self.target   = [0.0, 0.0]
        self.current_step = 0
        self.reset()

    def reset(self, target=None):
    
        self.position = [
            np.random.uniform(self.world_min, self.world_max),
            np.random.uniform(self.world_min, self.world_max),
        ]
        if target is None:
            self.target = [
                np.random.uniform(self.world_min, self.world_max),
                np.random.uniform(self.world_min, self.world_max),
            ]
        else:
            self.target = list(target)
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        if action is None:
            action = [0.0, 0.0]
        else:
            arr = np.asarray(action, dtype=float).ravel()
            if arr.size >= 2:
                action = [float(arr[0]), float(arr[1])]
            elif arr.size == 1:
                action = [float(arr[0]), 0.0]
            else:
                action = [0.0, 0.0]

        prev_dist = self._distance_to_target()

        self.position[0] = max(self.world_min, min(self.position[0] + action[0], self.world_max))
        self.position[1] = max(self.world_min, min(self.position[1] + action[1], self.world_max))

        new_dist = self._distance_to_target()
        reward = self._compute_reward(prev_dist, new_dist)
        self.current_step += 1

        terminated = new_dist < self.reach_threshold
        truncated  = self.current_step >= self.max_steps

        return self._get_state(), reward, terminated, truncated

    def _distance_to_target(self):
        dx = self.position[0] - self.target[0]
        dy = self.position[1] - self.target[1]
        return (dx ** 2 + dy ** 2) ** 0.5

    def _compute_reward(self, prev_dist, new_dist):
        shaping_reward = (prev_dist - new_dist)
        reach_bonus    = 10.0 if new_dist < self.reach_threshold else 0.0
        step_penalty   = -0.01
        return shaping_reward + reach_bonus + step_penalty

    def _get_state(self):
        return np.array([
            self.position[0], self.position[1],
            self.target[0],   self.target[1],
        ])

    def current_position(self):
        return self.position
