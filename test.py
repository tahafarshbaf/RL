import torch
import numpy as np
from environment import GripperEnv


def test(agent, num_episodes=20):
    env = GripperEnv()
    successes   = 0
    total_steps = []

    print("\n=== Deterministic Policy Test ===")
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done  = False
        steps = 0

        while not done:
            action                                    = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated = env.step(action)
            done  = terminated or truncated
            state = next_state
            steps += 1

        # Read final forces after episode ends
        f1, f2    = env._simulate_forces()
        avg_force = (f1 + f2) / 2.0

        status = "✓" if terminated else "✗"
        total_steps.append(steps if terminated else env.max_steps)
        if terminated:
            successes += 1

        print(f"  Episode {ep:2d}: {status} | "
              f"steps={steps:2d} | "
              f"final_angle={env.angle:.1f}deg | "
              f"target_force={env.target_force:.2f}N | "
              f"avg_force={avg_force:.2f}N")

    print(f"\nResult: {successes}/{num_episodes} success "
          f"({successes/num_episodes*100:.0f}%) | "
          f"Step Avg: {sum(total_steps)/len(total_steps):.1f}")