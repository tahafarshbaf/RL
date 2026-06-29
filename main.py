
from model import save_model
from results import plot_results, visualize_episodes
from test import test
from train import train

if __name__ == "__main__":
    print("Train Started\n")
    trained_agent, rewards, successes = train(num_episodes=800)
    plot_results(rewards, successes)
    save_model(trained_agent, "sac_2d_reach.pth")
    test(trained_agent, num_episodes=50)
    visualize_episodes(trained_agent, num_episodes=6)

"""
from environment import GripperEnv
from SAC import SACAgent


env = GripperEnv()
state = env.reset(target_force=3.0)
env.stiffness     = 0.08
env.contact_angle = 40.0
env.deform_factor = 1.0
# این بار angle رو دستی بذار داخل zone تماس
env.angle = 45.0

agent = SACAgent(state_dim=5, action_dim=1, max_action=3.0)

print(f"Target force: {env.target_force:.2f}N")
print(f"Start angle: {env.angle:.1f}deg | Contact angle: {env.contact_angle:.1f}deg")
print()
print(f"{'Step':>5} | {'Angle':>7} | {'Avg':>6} | {'Error':>6} | {'Action':>7} | {'Reward':>8}")
print("-" * 55)

state = env._get_state()
for step in range(30):
    f1, f2 = env._simulate_forces()
    avg    = (f1 + f2) / 2.0
    error  = avg - env.target_force

    action = agent.select_action(state, deterministic=False)
    next_state, reward, terminated, truncated = env.step(action)

    print(f"{step+1:>5} | {env.angle:>7.2f} | {avg:>6.3f} | {error:>6.3f} | {float(action):>7.3f} | {reward:>8.3f}")

    state = next_state
    if terminated:
        print("\n✓ Reached target!")
        break
    if truncated:
        print("\n✗ Timeout")
        break
"""