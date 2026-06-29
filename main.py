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
