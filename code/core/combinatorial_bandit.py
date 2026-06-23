import math

import numpy as np


class CombinatorialBanditLearner:
    """Per-flow combinatorial semi-bandit learner (CombUCB1-style).

    Each controller flow is treated as an arm. Every round the learner selects a
    subset of arms (a binary install mask) by ranking them with an upper
    confidence bound, then observes per-arm reward (semi-bandit feedback) via the
    shared ``reward_list``.

    Exposes the same method surface the simulations expect from
    ``MultiAgentDQN`` so it is a drop-in learner.

    The reward design is shared: rewards come from ``RewardFunction`` (honoring
    ``rewardAgingFactor`` and ``spatialReward``). ``rewardAgingFactor`` is also
    reused here as the EWMA decay applied to each arm's estimate, so aging is
    honored inside the bandit's internal statistics.
    """

    def __init__(self, args, controller_table):
        self.args = args
        self.num_flows = len(controller_table)
        self.num_flows_per_agent = args.numberofFlowsPerAgent

        # Kept for interface compatibility (simulation sizes current_state with it).
        self.num_states = self.num_flows
        self.num_actions = self.num_flows

        # Exploration constant for the UCB bonus.
        self.c = float(getattr(args, 'bandit_c', 1.0))
        # Reuse the reward aging factor as the EWMA decay on per-arm estimates.
        self.reward_aging_factor = float(getattr(args, 'rewardAgingFactor', 0.95))
        # Budget: how many arms to install per round (bounded by the table size).
        self.budget = min(int(getattr(args, 'tablesize', self.num_flows)), self.num_flows)

        # Per-arm running estimate (EWMA mean) and pull counts.
        self.estimates = np.zeros(self.num_flows, dtype=np.float64)
        self.counts = np.zeros(self.num_flows, dtype=np.int64)
        self.round = 0

        # Cache last action mask for the update step.
        self._last_action = np.zeros(self.num_flows, dtype=int)

    def _ucb_scores(self):
        """Upper-confidence-bound score per arm; unpulled arms get +inf."""
        scores = np.empty(self.num_flows, dtype=np.float64)
        log_t = math.log(self.round + 1)
        for i in range(self.num_flows):
            if self.counts[i] == 0:
                scores[i] = float('inf')
            else:
                bonus = self.c * math.sqrt(log_t / self.counts[i])
                scores[i] = self.estimates[i] + bonus
        return scores

    def select_actions(self, current_state):
        """Select the top-``budget`` arms by UCB score; returns a binary mask."""
        self.round += 1
        scores = self._ucb_scores()

        # Indices of the highest-scoring arms (ties broken arbitrarily).
        if self.budget >= self.num_flows:
            selected = np.arange(self.num_flows)
        else:
            selected = np.argpartition(scores, -self.budget)[-self.budget:]

        action = np.zeros(self.num_flows, dtype=int)
        action[selected] = 1
        self._last_action = action
        return action

    def convert_actions_to_binary(self, action_list):
        """The bandit already emits a per-flow binary mask."""
        return np.asarray(action_list, dtype=int)

    def store_transitions(self, current_state, action_list, agent_rewards, next_state, reward_list=None):
        """Update per-arm estimates from observed per-flow rewards (EWMA)."""
        if reward_list is None:
            return

        rewards = np.asarray(reward_list, dtype=np.float64)
        action = np.asarray(action_list, dtype=int)

        # Guard against length mismatches.
        n = min(self.num_flows, rewards.shape[0], action.shape[0])
        aging = self.reward_aging_factor
        for i in range(n):
            if action[i] == 1:
                # Exponential moving average decayed by the reward aging factor.
                self.estimates[i] = aging * self.estimates[i] + (1.0 - aging) * rewards[i]
                self.counts[i] += 1

    def learn(self):
        """Estimates are updated online in store_transitions; nothing to do."""
        return

    def decay_epsilon_for_all_agents(self):
        """UCB handles exploration internally; no-op."""
        return

    def get_average_epsilon(self):
        """Provided for interface compatibility (UCB has no epsilon)."""
        return 0.0

    def get_network_info(self):
        """Bandit configuration info for logging."""
        return {
            'algorithm': 'bandit',
            'bandit_type': 'comb_ucb1',
            'num_agents': 1,
            'num_arms': self.num_flows,
            'budget_per_round': self.budget,
            'exploration_constant_c': self.c,
            'reward_aging_factor': self.reward_aging_factor,
            'device': 'cpu',
            'cuda_available': False,
        }
