from environment import Env2D

def test(agent, num_episodes=30):
    env = Env2D(world_min=-1.0, world_max=1.0, max_steps=50, reach_threshold=0.1)
    successes   = 0
    total_steps = []

    print("\n Determinstic Policy Test:")
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

        status = "✓" if terminated else "✗"
        total_steps.append(steps if terminated else 50)
        if terminated:
            successes += 1

        print(f"  Episode {ep:2d}: {status} | "
              f"steps={steps:2d} | "
              f"final_pos=[{env.position[0]:.2f},{env.position[1]:.2f}] | "
              f"target=[{env.target[0]:.2f},{env.target[1]:.2f}]")

    print(f"\nResult {successes}/{num_episodes} success "
          f"({successes/num_episodes*100:.0f}%) | "
          f"Step Avg. {sum(total_steps)/len(total_steps):.1f}")