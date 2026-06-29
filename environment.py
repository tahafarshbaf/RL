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

    ANGLE_MIN = 0.0  # fully open (degrees)
    ANGLE_MAX = 180.0  # fully closed (degrees)
    FORCE_MAX = 10.0  # max measurable force (N)
    MAX_DELTA = 3.0  # max angle change per step (degrees)
    MAX_STEPS = 150
    FORCE_THRESHOLD = 0.3  # N — tighter than before

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
        # Randomize object properties each episode
        # Agent doesn't know these — must infer from force sensor readings
        self.stiffness = np.random.uniform(0.03, 0.15)  # N/degree
        self.contact_angle = np.random.uniform(20.0, 80.0)  # degrees
        self.deform_rate = np.random.uniform(0.005, 0.02)  # decay per step
        self.deform_factor = 1.0  # resets each episode

        # Random start angle (agent doesn't always start from zero)
        self.angle = np.random.uniform(self.ANGLE_MIN, self.contact_angle)

        # Random target force
        if target_force is None:
            self.target_force = np.random.uniform(1.0, 6.0)
        else:
            self.target_force = float(target_force)

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
        """
        Physics model:
        - No force before contact
        - After contact: F = stiffness * (angle - contact_angle) * deform_factor
        - Small Gaussian noise on each sensor

        In real hardware: replace this method with serial port readings.
        """
        if self.angle <= self.contact_angle:
            # Phase 1: no contact yet
            base_force = 0.0
        else:
            # Phase 2: contact + deformation
            compression = self.angle - self.contact_angle
            base_force = self.stiffness * compression * self.deform_factor

        base_force = np.clip(base_force, 0.0, self.FORCE_MAX)

        # Each finger reads slightly different + sensor noise
        force1 = np.clip(
            base_force * 0.95 + np.random.normal(0, 0.05), 0, self.FORCE_MAX
        )
        force2 = np.clip(
            base_force * 1.05 + np.random.normal(0, 0.05), 0, self.FORCE_MAX
        )

        return force1, force2

    def _compute_reward(self, force1, force2):
        avg_force = (force1 + force2) / 2.0
        error = abs(avg_force - self.target_force)

        # Shaping: penalize distance from target
        shaping = -error * 2.0

        # Bonus: within threshold
        terminated = error < self.FORCE_THRESHOLD
        reach_bonus = 10.0 if terminated else 0.0

        # Overshoot penalty: applying too much force is dangerous
        overshoot_penalty = -5.0 if avg_force > self.target_force + 1.5 else 0.0

        # Small step penalty to encourage efficiency
        step_penalty = -0.01

        reward = shaping + reach_bonus + overshoot_penalty + step_penalty
        return reward, terminated

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
