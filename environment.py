import numpy as np

class GripperEnv:
    """
    Gripper simulation with object deformation and unknown stiffness.

    Physics model:
    - Phase 1 (free motion): angle < contact_angle → no force
    - Phase 2 (contact):     force = stiffness * (angle - contact_angle) * deformation_factor
    - Deformation:           object softens over time → force drops even at fixed angle

    Each episode has a different object (random stiffness + contact angle),
    so the agent must learn to generalize across objects.
    """

    ANGLE_MIN   =   0.0
    ANGLE_MAX   = 180.0
    FORCE_MAX   =  10.0
    MAX_DELTA   =   5.0    # larger steps to reach target faster
    MAX_STEPS   = 200      # more steps
    FORCE_THRESHOLD = 0.3

    def __init__(self):
        # Object properties — unknown to agent, randomized each episode
        self.stiffness = 1.0  # N/degree
        self.contact_angle = 0.0  # angle at which contact begins
        self.deform_rate = 0.0  # how fast object softens (lambda)
        self.deform_factor = 1.0  # current deformation state (starts at 1, decays)

        self.angle = 0.0
        self.target_force = 0.0
        self.current_step = 0

        self.reset()

    def reset(self, target_force=None):
        self.stiffness     = np.random.uniform(0.05, 0.15)
        self.contact_angle = np.random.uniform(10.0, 60.0)
        self.deform_rate   = np.random.uniform(0.002, 0.01)
        self.deform_factor = 1.0
    
        if target_force is None:
            self.target_force = np.random.uniform(0.5, 4.0)
        else:
            self.target_force = float(target_force)
    
        # Calculate the angle needed to reach target force
        # F = stiffness * (angle - contact_angle)
        # angle = contact_angle + F/stiffness
        angle_for_target = self.contact_angle + (self.target_force / self.stiffness)
        angle_for_target = np.clip(angle_for_target, self.ANGLE_MIN, self.ANGLE_MAX)
    
        # Start angle: randomly either before or close to target angle
        # so agent sometimes needs to go up, sometimes fine-tune
        start_offset = np.random.uniform(-10.0, 10.0)
        self.angle = float(np.clip(
            angle_for_target + start_offset,
            self.contact_angle,       # never start before contact
            self.ANGLE_MAX
        ))
    
        self.current_step = 0
        return self._get_state()

    def step(self, action):
        # Clip action to allowed range and convert to scalar
        delta = float(np.clip(action, -self.MAX_DELTA, self.MAX_DELTA))

        # Update angle
        self.angle = np.clip(self.angle + delta, self.ANGLE_MIN, self.ANGLE_MAX)

        # Update deformation: object softens over time when in contact
        force1, force2 = self._simulate_forces()
        avg_force = (force1 + force2) / 2.0
        if avg_force > 0.1:  # only deforms when in contact
            self.deform_factor *= 1.0 - self.deform_rate
            self.deform_factor = max(
                self.deform_factor, 0.3
            )  # floor: object doesn't vanish

        reward, terminated = self._compute_reward(force1, force2)

        self.current_step += 1
        truncated = self.current_step >= self.MAX_STEPS

        return self._get_state(force1, force2), reward, terminated, truncated

    def _simulate_forces(self):
        if self.angle <= self.contact_angle:
            base_force = 0.0
        else:
            compression = self.angle - self.contact_angle
            base_force  = self.stiffness * compression * self.deform_factor

        base_force = np.clip(base_force, 0.0, self.FORCE_MAX)

        # Reduce noise significantly
        noise_std = 0.02   # was 0.05
        force1 = np.clip(base_force * 0.95 + np.random.normal(0, noise_std), 0, self.FORCE_MAX)
        force2 = np.clip(base_force * 1.05 + np.random.normal(0, noise_std), 0, self.FORCE_MAX)

        return force1, force2

    def _compute_reward(self, force1, force2):
        avg_force = (force1 + force2) / 2.0
        error     = abs(avg_force - self.target_force)

    # Normalize error to [0, 1] range relative to FORCE_MAX
        error_normalized = error / self.FORCE_MAX

    # Shaping: small and normalized
        shaping = -error_normalized

    # Big enough bonus to dominate the shaping signal
        terminated  = error < self.FORCE_THRESHOLD
        reach_bonus = 50.0 if terminated else 0.0

    # Overshoot penalty (normalized)
        overshoot = avg_force - (self.target_force + 1.5)
        overshoot_penalty = -min(overshoot / self.FORCE_MAX, 0.0) * 5.0 if avg_force > self.target_force + 1.5 else 0.0

        step_penalty = -0.005

        return shaping + reach_bonus + overshoot_penalty + step_penalty, terminated

    def _get_state(self, force1=None, force2=None):
        """
        State: 5 values, all normalized to [-1, 1]

        Added force_error compared to previous version:
        agent needs to know how far it is from target,
        especially when object is deforming under constant angle.
        """
        if force1 is None or force2 is None:
            force1, force2 = self._simulate_forces()
        avg_force = (force1 + force2) / 2.0
        force_error = avg_force - self.target_force  # signed error

        angle_norm = (self.angle / self.ANGLE_MAX) * 2 - 1
        force1_norm = (force1 / self.FORCE_MAX) * 2 - 1
        force2_norm = (force2 / self.FORCE_MAX) * 2 - 1
        error_norm = np.clip(force_error / self.FORCE_MAX, -1, 1)
        target_norm = (self.target_force / self.FORCE_MAX) * 2 - 1

        return np.array(
            [
                angle_norm,
                force1_norm,
                force2_norm,
                error_norm,
                target_norm,
            ],
            dtype=np.float32,
        )





class Env2D:
    def __init__(
        self, world_min=-1.0, world_max=1.0, max_steps=50, reach_threshold=0.1
    ):
        self.world_min = world_min
        self.world_max = world_max
        self.max_steps = max_steps
        self.reach_threshold = reach_threshold
        self.position = [0.0, 0.0]
        self.target = [0.0, 0.0]
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

        self.position[0] = max(
            self.world_min, min(self.position[0] + action[0], self.world_max)
        )
        self.position[1] = max(
            self.world_min, min(self.position[1] + action[1], self.world_max)
        )

        new_dist = self._distance_to_target()
        reward = self._compute_reward(prev_dist, new_dist)
        self.current_step += 1

        terminated = new_dist < self.reach_threshold
        truncated = self.current_step >= self.max_steps

        return self._get_state(), reward, terminated, truncated

    def _distance_to_target(self):
        dx = self.position[0] - self.target[0]
        dy = self.position[1] - self.target[1]
        return (dx**2 + dy**2) ** 0.5

    def _compute_reward(self, prev_dist, new_dist):
        shaping_reward = prev_dist - new_dist
        reach_bonus = 10.0 if new_dist < self.reach_threshold else 0.0
        step_penalty = -0.01
        return shaping_reward + reach_bonus + step_penalty

    def _get_state(self):
        return np.array(
            [
                self.position[0],
                self.position[1],
                self.target[0],
                self.target[1],
            ]
        )

    def current_position(self):
        return self.position
