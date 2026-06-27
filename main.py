from train import *
from results import *
from model import *
from test import *


if __name__ == "__main__":
    print("Train Started /")
    trained_agent, rewards, successes = train(num_episodes=800)
    plot_results(rewards, successes)
    save_model(trained_agent, "sac_2d_reach.pth")
    test(trained_agent, num_episodes=20)
    visualize_episodes(trained_agent, num_episodes=6)