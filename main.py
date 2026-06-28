from train import train
from results import plot_results
from model import save_model
from test import test
from results import visualize_episodes


if __name__ == "__main__":
    print("Train Started\n")
    trained_agent, rewards, successes = train(num_episodes=1000)
    plot_results(rewards, successes)
    save_model(trained_agent, "sac_2d_reach.pth\n")
    test(trained_agent, num_episodes=20)
    visualize_episodes(trained_agent, num_episodes=6)