from environment import GripperEnv
from replayBuffer import ReplayBuffer
from SAC import SACAgent


def train(num_episodes=1000, batch_size=64, start_after=200):
    env = GripperEnv()
    agent = SACAgent(
        state_dim  = 5,     # [angle, force1, force2, force_error, target_force]
        action_dim = 1,     # delta_angle
        max_action = 5.0,   # max 3 degrees per step
    )
    buffer = ReplayBuffer(capacity=50_000)

    episode_rewards = []
    success_history = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated = env.step(action)

            done = terminated or truncated
            buffer.push(state, action, reward, next_state, terminated)

            if len(buffer) >= max(batch_size, start_after):
                agent.update(buffer.sample(batch_size))

            episode_reward += reward
            state = next_state

        episode_rewards.append(episode_reward)
        success_history.append(1.0 if terminated else 0.0)

        if episode % 50 == 0:
            avg_r = sum(episode_rewards[-10:]) / 10
            success = sum(success_history[-10:]) / 10 * 100
            print(
                f"Episode {episode:4d} | "
                f"Reward: {episode_reward:7.2f} | "
                f"Avg(10): {avg_r:7.2f} | "
                f"Success%: {success:5.1f} | "
                f"Alpha: {agent.alpha.item():.4f}"
            )

    return agent, episode_rewards, success_history
