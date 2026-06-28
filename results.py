import matplotlib.pyplot as plt
#import matplotlib.patches as patches
from environment import GripperEnv

def visualize_episodes(agent, num_episodes=6):
    env  = GripperEnv(max_steps=100, force_threshold=0.5)
    cols = 3
    rows = (num_episodes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for ep in range(num_episodes):
        state = env.reset()
        done  = False

        # Track angle and forces over time
        angles, forces1, forces2 = [env.angle], [], []

        while not done:
            action                                    = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated = env.step(action)
            done  = terminated or truncated
            state = next_state

            f1, f2 = env._simulate_forces()
            angles.append(env.angle)
            forces1.append(f1)
            forces2.append(f2)

        ax = axes[ep]
        steps = range(len(forces1))

        # Plot forces over time
        ax.plot(steps, forces1, 'b-', linewidth=1.5, label='Force 1')
        ax.plot(steps, forces2, 'r-', linewidth=1.5, label='Force 2')

        # Target force as horizontal line
        ax.axhline(y=env.target_force, color='green',
                   linestyle='--', linewidth=1.5, label=f'Target ({env.target_force:.1f}N)')

        # Threshold band around target
        ax.axhspan(env.target_force - env.force_threshold,
                   env.target_force + env.force_threshold,
                   alpha=0.1, color='green', label='Threshold band')

        ax.set_xlabel("Step")
        ax.set_ylabel("Force (N)")
        ax.set_ylim(0, env.FORCE_MAX)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        status = "Reached" if terminated else "Not Reached"
        ax.set_title(f"Episode {ep+1}: {status} ({len(forces1)} steps) | "
                     f"Angle: {env.angle:.1f}°")

    for ep in range(num_episodes, len(axes)):
        axes[ep].set_visible(False)

    plt.suptitle("Gripper Force Control (Deterministic Policy)", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_results(episode_rewards, success_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    def smooth(x, w=10):
        return [sum(x[max(0, i-w):i+1]) / len(x[max(0, i-w):i+1]) for i in range(len(x))]

    ax1.plot(episode_rewards, alpha=0.3, color='steelblue')
    ax1.plot(smooth(episode_rewards), color='steelblue', linewidth=2, label='Avg-10')
    ax1.set_title("Episode Reward"); ax1.set_xlabel("Episode")
    ax1.grid(alpha=0.3); ax1.legend()

    ax2.plot(success_history, alpha=0.2, color='seagreen')
    ax2.plot(smooth(success_history), color='seagreen', linewidth=2, label='Success rate')
    ax2.set_title("Success Rate"); ax2.set_xlabel("Episode")
    ax2.grid(alpha=0.3); ax2.legend()

    plt.tight_layout()
    plt.show()
