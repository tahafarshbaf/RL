import numpy as np

class GripperEnv:
    
    ANGLE_MIN  =   0.0   
    ANGLE_MAX  = 180.0   
    FORCE_MAX  =  10.0   
    MAX_DELTA  =   5.0   

    def __init__(self, max_steps=100, force_threshold=0.5):
        self.max_steps       = max_steps
        self.force_threshold = force_threshold  # بازه‌ی قبول برای نیرو

        self.angle        = 0.0
        self.target_force = 0.0
        self.current_step = 0
        self.reset()

    def reset(self, target_force=None):
        
        self.angle = np.random.uniform(self.ANGLE_MIN, self.ANGLE_MAX)

        
        if target_force is None:
            self.target_force = np.random.uniform(1.0, 8.0)
        else:
            self.target_force = float(target_force)

        self.current_step = 0
        return self._get_state()

    def step(self, action):
        
        delta = float(np.clip(action, -self.MAX_DELTA, self.MAX_DELTA))

        self.angle = np.clip(
            self.angle + delta,
            self.ANGLE_MIN, self.ANGLE_MAX
        )

        force1, force2 = self._simulate_forces()

        reward, terminated = self._compute_reward(force1, force2)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        return self._get_state(), reward, terminated, truncated

    def _simulate_forces(self):
        
        base_force = (self.angle / self.ANGLE_MAX) * self.FORCE_MAX

        force1 = np.clip(base_force * 0.95 + np.random.normal(0, 0.1), 0, self.FORCE_MAX)
        force2 = np.clip(base_force * 1.05 + np.random.normal(0, 0.1), 0, self.FORCE_MAX)

        return force1, force2

    def _compute_reward(self, force1, force2):
        
        avg_force = (force1 + force2) / 2.0
        error     = abs(avg_force - self.target_force)

        shaping_reward = -error

        terminated    = error < self.force_threshold
        reach_bonus   = 10.0 if terminated else 0.0

        step_penalty  = -0.01

        reward = shaping_reward + reach_bonus + step_penalty
        return reward, terminated

    def _get_state(self):
        
        force1, force2 = self._simulate_forces()

        angle_norm  = (self.angle / self.ANGLE_MAX) * 2 - 1
        force1_norm = (force1 / self.FORCE_MAX) * 2 - 1
        force2_norm = (force2 / self.FORCE_MAX) * 2 - 1
        target_norm = (self.target_force / self.FORCE_MAX) * 2 - 1

        return np.array([angle_norm, force1_norm, force2_norm, target_norm],
                        dtype=np.float32)



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
