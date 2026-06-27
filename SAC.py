from neuralNetwork import Actor, Critic

import torch
import torch.nn as nn
import torch.optim as optim
import copy

class SACAgent:
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64,
                 max_action=0.15, lr=3e-4, gamma=0.99, tau=0.005):
        self.gamma      = gamma
        self.tau        = tau
        self.action_dim = action_dim

        self.actor     = Actor(state_dim, action_dim, hidden_dim, max_action)
        self.q1        = Critic(state_dim, action_dim, hidden_dim)
        self.q2        = Critic(state_dim, action_dim, hidden_dim)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        self.log_alpha      = torch.zeros(1, requires_grad=True)
        self.target_entropy = -float(action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer    = optim.Adam(self.q1.parameters(),    lr=lr)
        self.q2_optimizer    = optim.Adam(self.q2.parameters(),    lr=lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha],        lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = (self.actor.deterministic_action(state) if deterministic
                      else self.actor.sample(state)[0])
        return action.squeeze(0).numpy()

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch

        # قدم ۱: آپدیت Critic‌ها
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q_next    = torch.min(self.q1_target(next_states, next_actions),
                                  self.q2_target(next_states, next_actions))
            td_target = rewards + (1 - dones) * self.gamma * (q_next - self.alpha * next_log_probs)

        q1_loss = nn.functional.mse_loss(self.q1(states, actions), td_target)
        q2_loss = nn.functional.mse_loss(self.q2(states, actions), td_target)

        self.q1_optimizer.zero_grad(); q1_loss.backward(); self.q1_optimizer.step()
        self.q2_optimizer.zero_grad(); q2_loss.backward(); self.q2_optimizer.step()

        # قدم ۲: آپدیت Actor
        new_actions, log_probs = self.actor.sample(states)
        q_new      = torch.min(self.q1(states, new_actions), self.q2(states, new_actions))
        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()

        # قدم ۳: آپدیت Alpha
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad(); alpha_loss.backward(); self.alpha_optimizer.step()

        # قدم ۴: Soft update target networks
        for t, s in [(self.q1_target, self.q1), (self.q2_target, self.q2)]:
            for tp, sp in zip(t.parameters(), s.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        return {
            "q1_loss":    q1_loss.item(),
            "q2_loss":    q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha":      self.alpha.item(),
        }