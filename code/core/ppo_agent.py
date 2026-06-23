import math

import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Shared-body actor-critic network.

    The actor produces one logit per flow (a factored Bernoulli policy), so the
    action space scales linearly with the number of flows instead of the
    exponential ``2**numberofFlowsPerAgent`` head used by the DQN. The critic
    produces a single scalar state-value used as the PPO baseline.
    """

    def __init__(self, num_flows, hidden_layers, hidden_size):
        super(ActorCritic, self).__init__()

        layers = []
        input_size = num_flows
        for _ in range(hidden_layers):
            linear = nn.Linear(input_size, hidden_size)
            linear.weight.data.normal_(0, 0.1)
            layers.append(linear)
            layers.append(nn.ReLU())
            input_size = hidden_size
        self.body = nn.Sequential(*layers)

        self.actor = nn.Linear(input_size, num_flows)
        self.actor.weight.data.normal_(0, 0.1)
        self.critic = nn.Linear(input_size, 1)
        self.critic.weight.data.normal_(0, 0.1)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h)


class PPOLearner:
    """Single-agent PPO learner with a factored per-flow Bernoulli policy.

    Exposes the same method surface the simulations expect from ``MultiAgentDQN``
    (``num_states``, ``select_actions``, ``convert_actions_to_binary``,
    ``store_transitions``, ``learn``, ``decay_epsilon_for_all_agents``,
    ``get_average_epsilon``, ``get_network_info``) so it is a drop-in learner.

    The reward design is shared: rewards are computed by ``RewardFunction``
    (honoring ``rewardAgingFactor`` and ``spatialReward``); PPO additionally
    discounts across steps through ``gamma`` / GAE.
    """

    def __init__(self, args, controller_table):
        self.args = args
        self.num_flows = len(controller_table)
        self.num_flows_per_agent = args.numberofFlowsPerAgent

        # State = previous per-flow install decision (degenerate but consistent
        # with how the simulation feeds ``current_state = next_state``).
        self.num_states = self.num_flows
        self.num_actions = self.num_flows

        # Device selection mirrors MultiAgentDQN.
        if hasattr(args, 'selected_device'):
            self.device = torch.device(args.selected_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hidden_layers = getattr(args, 'hidden_layers', 2)
        hidden_layer_size = getattr(args, 'hidden_layer_size', None)
        self.hidden_size = hidden_layer_size if hidden_layer_size is not None else 64
        self.hidden_layers = hidden_layers

        self.net = ActorCritic(self.num_flows, hidden_layers, self.hidden_size).to(self.device)

        # Hyperparameters (robust to missing args via getattr defaults).
        self.lr = float(getattr(args, 'ppo_lr', 3e-4))
        self.clip = float(getattr(args, 'ppo_clip', 0.2))
        self.epochs = int(getattr(args, 'ppo_epochs', 4))
        self.entropy_coef = float(getattr(args, 'ppo_entropy_coef', 0.01))
        self.value_coef = float(getattr(args, 'ppo_value_coef', 0.5))
        self.gae_lambda = float(getattr(args, 'ppo_gae_lambda', 0.95))
        self.gamma = float(getattr(args, 'gamma', 0.9))
        # Number of collected steps before a PPO update is performed.
        self.rollout_len = int(getattr(args, 'batch_size', 32))

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # Rollout buffer of dict transitions.
        self.buffer = []

        # Per-step cache populated during select_actions.
        self._last_state = None
        self._last_action = None
        self._last_log_prob = None
        self._last_value = None

        self.learn_step_counter = 0

    def _to_state_tensor(self, state):
        """Convert a numpy array / torch tensor state into a (1, num_flows) tensor."""
        if isinstance(state, torch.Tensor):
            arr = state.detach().cpu().numpy()
        else:
            arr = np.asarray(state, dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        # Defensive: enforce expected dimensionality.
        if arr.shape[0] != self.num_flows:
            fixed = np.zeros(self.num_flows, dtype=np.float32)
            n = min(self.num_flows, arr.shape[0])
            fixed[:n] = arr[:n]
            arr = fixed
        return torch.from_numpy(arr).float().unsqueeze(0).to(self.device)

    def select_actions(self, current_state):
        """Sample a per-flow binary install decision from the current policy."""
        state = self._to_state_tensor(current_state)
        with torch.no_grad():
            logits, value = self.net(state)
            dist = torch.distributions.Bernoulli(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)

        self._last_state = state.squeeze(0).cpu().numpy()
        self._last_action = action.squeeze(0).cpu().numpy().astype(int)
        self._last_log_prob = float(log_prob.item())
        self._last_value = float(value.item())

        return self._last_action

    def convert_actions_to_binary(self, action_list):
        """The factored policy already emits a per-flow binary mask."""
        return np.asarray(action_list, dtype=int)

    def store_transitions(self, current_state, action_list, agent_rewards, next_state, reward_list=None):
        """Append a transition to the rollout buffer.

        Uses the per-flow ``reward_list`` summed into a scalar step reward when
        available (the same signal the reward function produces), otherwise
        falls back to the aggregated ``agent_rewards``.
        """
        if reward_list is not None:
            reward = float(np.sum(np.asarray(reward_list, dtype=np.float32)))
        else:
            reward = float(np.sum(np.asarray(agent_rewards, dtype=np.float32)))

        if isinstance(next_state, torch.Tensor):
            next_state_np = next_state.detach().cpu().numpy()
        else:
            next_state_np = np.asarray(next_state, dtype=np.float32)

        self.buffer.append({
            'state': np.asarray(self._last_state, dtype=np.float32),
            'action': np.asarray(self._last_action, dtype=np.float32),
            'log_prob': self._last_log_prob,
            'value': self._last_value,
            'reward': reward,
            'next_state': np.asarray(next_state_np, dtype=np.float32),
        })

    def _compute_gae(self, rewards, values, bootstrap_value):
        """Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_value = bootstrap_value if t == n - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
        returns = advantages + np.asarray(values, dtype=np.float32)
        return advantages, returns

    def learn(self):
        """Run a PPO clipped-surrogate update once enough steps are collected."""
        if len(self.buffer) < self.rollout_len:
            return

        states = torch.from_numpy(
            np.stack([b['state'] for b in self.buffer])
        ).float().to(self.device)
        actions = torch.from_numpy(
            np.stack([b['action'] for b in self.buffer])
        ).float().to(self.device)
        old_log_probs = torch.tensor(
            [b['log_prob'] for b in self.buffer], dtype=torch.float32, device=self.device
        )
        values = [b['value'] for b in self.buffer]
        rewards = [b['reward'] for b in self.buffer]

        # Bootstrap value from the final next_state.
        with torch.no_grad():
            last_next = self._to_state_tensor(self.buffer[-1]['next_state'])
            _, bootstrap_v = self.net(last_next)
            bootstrap_value = float(bootstrap_v.item())

        advantages_np, returns_np = self._compute_gae(rewards, values, bootstrap_value)
        advantages = torch.from_numpy(advantages_np).float().to(self.device)
        returns = torch.from_numpy(returns_np).float().to(self.device)

        # Normalize advantages for stability.
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            logits, value_pred = self.net(states)
            dist = torch.distributions.Bernoulli(logits=logits)
            new_log_probs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1).mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(value_pred.squeeze(-1), returns)

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.learn_step_counter += 1
        self.buffer = []

    def decay_epsilon_for_all_agents(self):
        """PPO explores via policy entropy, not epsilon-greedy; no-op."""
        return

    def get_average_epsilon(self):
        """Provided for interface compatibility (PPO has no epsilon)."""
        return 0.0

    def get_network_info(self):
        """Network/hyperparameter info for logging."""
        return {
            'algorithm': 'ppo',
            'num_agents': 1,
            'input_size': self.num_states,
            'output_size': self.num_actions,
            'policy': 'factored_per_flow_bernoulli',
            'num_hidden_layers': self.hidden_layers,
            'hidden_layer_size': self.hidden_size,
            'total_parameters': sum(p.numel() for p in self.net.parameters()),
            'ppo_lr': self.lr,
            'ppo_clip': self.clip,
            'ppo_epochs': self.epochs,
            'ppo_entropy_coef': self.entropy_coef,
            'ppo_value_coef': self.value_coef,
            'ppo_gae_lambda': self.gae_lambda,
            'gamma': self.gamma,
            'rollout_len': self.rollout_len,
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
        }
