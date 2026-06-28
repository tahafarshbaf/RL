import torch.nn as nn
import torch.distributions as distributions
import torch

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """
    Critic (Q-network) network for SAC.
    State + Action → Q-value (how good was this action in this state?)
    Q(state, action) -> value
    Input: state, action
    Output: value
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.mlp = MLP(state_dim + action_dim, hidden_dim, 1)

    def forward(self, state, action):
        return self.mlp(torch.cat([state, action], dim=-1))
        # Concatenates the `state` and `action` tensors along the last dimension.


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, max_action=0.15):
        super().__init__()
        self.max_action = max_action
        self.feature_extractor = MLP(state_dim, hidden_dim, hidden_dim)
        self.mu_head      = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        features = self.feature_extractor(state)
        mu       = self.mu_head(features)
        log_std  = torch.clamp(self.log_std_head(features), min=-5, max=2)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std    = log_std.exp()
        normal = distributions.Normal(mu, std)
        a_raw  = normal.rsample()

        log_prob_raw = normal.log_prob(a_raw).sum(dim=-1, keepdim=True)
        action       = torch.tanh(a_raw)
        correction   = torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        log_prob     = log_prob_raw - correction

        return action * self.max_action, log_prob

    def deterministic_action(self, state):
        mu, _ = self.forward(state)
        return torch.tanh(mu) * self.max_action
