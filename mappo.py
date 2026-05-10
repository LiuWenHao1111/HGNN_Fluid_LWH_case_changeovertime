import numpy as np
import torch
import torch.optim as optim
from hgnn import FEGRL_Network


class PPO_Agent(object):
    def __init__(
        self,
        op_dim,
        m_dim,
        edge_dim,
        lr=0.0001,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=4,
        batch_size=256,
        hidden_dim=64,
        num_layers=2
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy = FEGRL_Network(
            num_op_features=op_dim,
            num_m_features=m_dim,
            num_edge_features=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = torch.nn.MSELoss()
        self.buffer = []

    def select_action(self, state):
        self.policy.eval()
        with torch.no_grad():
            probs, val, _ = self.policy(state, self.device)
            if probs is None:
                return None, None, None
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            return action_idx.item(), float(dist.log_prob(action_idx).item()), float(val.item())

    def evaluate_action(self, state):
        self.policy.eval()
        with torch.no_grad():
            probs, _, _ = self.policy(state, self.device)
            return torch.argmax(probs).item() if probs is not None else None

    def store_transition(self, t):
        self.buffer.append(t)

    def _compute_returns(self, rewards, dones):
        returns = []
        running = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            running = float(r) + self.gamma * running * (1.0 - float(d))
            returns.append(running)
        returns.reverse()
        return returns

    def update(self):
        if len(self.buffer) == 0:
            return 0.0

        self.policy.train()
        states, actions, old_log_probs, rewards, dones, values = zip(*self.buffer)

        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        returns = torch.tensor(self._compute_returns(rewards, dones), dtype=torch.float32).to(self.device)
        advantages = returns - values

        total_loss, update_steps = 0.0, 0
        indices = np.arange(len(states))

        for _ in range(self.K_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                curr_log_probs_list = []
                curr_vals_list = []
                valid_batch_idx = []

                for i in batch_idx:
                    probs, val, _ = self.policy(states[i], self.device)
                    if probs is None:
                        continue
                    dist = torch.distributions.Categorical(probs)
                    curr_log_probs_list.append(dist.log_prob(actions[i]))
                    curr_vals_list.append(val.squeeze())
                    valid_batch_idx.append(i)

                if not curr_log_probs_list:
                    continue

                curr_log_probs = torch.stack(curr_log_probs_list)
                curr_vals = torch.stack(curr_vals_list)

                batch_adv = advantages[valid_batch_idx]
                batch_returns = returns[valid_batch_idx]
                batch_old_log_probs = old_log_probs[valid_batch_idx]

                ratios = torch.exp(curr_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_adv
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.mse_loss(curr_vals, batch_returns)
                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.item())
                update_steps += 1

        self.buffer = []
        return total_loss / update_steps if update_steps > 0 else 0.0