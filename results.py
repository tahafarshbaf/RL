import matplotlib.pyplot as plt
import matplotlib.patches as patches
from environment import Env2D

def visualize_episodes(agent, num_episodes=6):
    env  = Env2D(world_min=-1.0, world_max=1.0, max_steps=50, reach_threshold=0.1)
    cols = 3
    rows = (num_episodes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for ax in axes:
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
    

    for ep in range(num_episodes):
        state    = env.reset()
        start    = env.position.copy()    # ← موقعیت شروع واقعی
        trajectory = [start]
        done     = False

        while not done:
            action                                    = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated = env.step(action)
            trajectory.append(env.position.copy())
            done  = terminated or truncated
            state = next_state

        ax = axes[ep]
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_aspect('equal'); ax.grid(alpha=0.3)

        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]
        ax.plot(xs, ys, 'b-', linewidth=1.5, alpha=0.7)
        ax.plot(xs[0],  ys[0],  'ro', markersize=8,  label='start')
        ax.plot(xs[-1], ys[-1], 'bs', markersize=7,  label='end')

        target_circle = patches.Circle(env.target, radius=0.1, color='green', alpha=0.3)
        ax.add_patch(target_circle)
        ax.plot(env.target[0], env.target[1], 'g*', markersize=12, label='target')

        status = "Reached" if terminated else "Not Reached"
        ax.set_title(f"Episode {ep+1}: {status} ({len(trajectory)-1} step)")
        ax.legend(fontsize=7, loc='upper left')

    for ep in range(num_episodes, len(axes)):
        axes[ep].set_visible(False)

    plt.suptitle("Path Agent (Deterministic Policy)", fontsize=13)
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
