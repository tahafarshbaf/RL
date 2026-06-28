import os

import torch


def save_model(agent, path="sac_2d_reach.pth"):
    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
            "q1_target": agent.q1_target.state_dict(),
            "q2_target": agent.q2_target.state_dict(),
            "log_alpha": agent.log_alpha.data,
        },
        path,
    )
    print(f"Model Saved← {path}")


def load_model(agent, path="sac_2d_reach.pth"):
    if not os.path.exists(path):
        print(f"file {path} not found!")
        return
    checkpoint = torch.load(path, weights_only=True)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.q1.load_state_dict(checkpoint["q1"])
    agent.q2.load_state_dict(checkpoint["q2"])
    agent.q1_target.load_state_dict(checkpoint["q1_target"])
    agent.q2_target.load_state_dict(checkpoint["q2_target"])
    agent.log_alpha.data = checkpoint["log_alpha"]
    print(f"Model Loaded ← {path}")
