# PPO Implementation (`--algorithm ppo`)

This document explains `PPOLearner` in `code/core/ppo_agent.py`. It assumes you
already understand the DQN learner (see `docs/dqn_implementation.md`) and
focuses on what changes relative to it.

---

## 1. One-line summary

PPO replaces DQN's value-based, `2^k`-action, multi-agent design with a
**single actor-critic network** whose actor emits **one install probability per
flow** (a factored Bernoulli policy). It learns on-policy with the
clipped-surrogate objective.

---

## 2. DQN vs PPO at a glance

| Aspect | DQN | PPO |
| --- | --- | --- |
| Family | value-based | policy-gradient (actor-critic) |
| Networks | `num_agents` Q-nets + target nets | one actor-critic net |
| Action space | one integer in `2^k` per agent | `num_flows` independent Bernoulli bits |
| Exploration | epsilon-greedy | policy entropy (stochastic sampling) |
| Data usage | off-policy replay buffer | on-policy rollout, then discarded |
| Update trigger | every LTI once buffer full | every `rollout_len` LTIs |
| Uses `reward_list`? | no (uses `agent_rewards`) | yes (summed to a scalar step reward) |

The action space is the headline change. With `num_flows = 25`, PPO's actor has
**25 outputs**, versus DQN's `3 x 1024`. PPO scales linearly in the number of
flows.

---

## 3. The network: `ActorCritic`

`code/core/ppo_agent.py` lines 8-37.

```
state (dim = num_flows)
      │
   [shared body: hidden_layers x Linear+ReLU]
      ├──> actor head  -> num_flows logits  (one Bernoulli logit per flow)
      └──> critic head -> 1 scalar value    (the PPO baseline)
```

- Input dim = `num_states = num_flows = 25` (the state is the previous install
  mask; see note below).
- Actor output = `num_flows` logits. Each logit `z_i` defines
  `P(install flow i) = sigmoid(z_i)`. The flows are **conditionally
  independent** given the state — this is the "factored" policy.
- Critic output = a single scalar `V(s)` used as a variance-reduction baseline.
- Hidden size defaults to 64 (`--hidden_layer_size` overrides), with
  `--hidden_layers` layers (default 2).

> State note: `num_states = num_flows` (line 60) so the simulation's
> `current_state = torch.zeros(num_states)` and `next_state =
> torch.FloatTensor(action_list)` line up dimensionally. The state is the
> previous binary decision — degenerate, like DQN's, but PPO does not depend on
> it being meaningful.

---

## 4. Acting: `select_actions`

`code/core/ppo_agent.py` lines 115-129.

```python
logits, value = self.net(state)
dist   = torch.distributions.Bernoulli(logits=logits)
action = dist.sample()                       # 0/1 per flow
log_prob = dist.log_prob(action).sum(dim=1)  # log pi(a|s) = sum over flows
```

It caches, for the step, the `state`, the sampled binary `action`, the scalar
`log_prob`, and the critic `value`. These four are needed later by the PPO
update. `convert_actions_to_binary` is the identity here (lines 131-133) — the
action already *is* the install mask, unlike DQN which had to bit-expand
integers.

Worked example (5 of 25 flows shown):

```
logits   = [ 2.1, -3.0, 0.4, -0.2, 1.5, ... ]
sigmoid  = [0.89, 0.05, 0.60, 0.45, 0.82, ...]
sample   = [   1,    0,    1,    0,    1, ...]   # this is the install mask
```

---

## 5. Collecting experience: `store_transitions`

`code/core/ppo_agent.py` lines 135-159.

PPO needs a scalar reward per step. It uses the **per-flow `reward_list`
summed** (the same per-flow signal the reward function produces), falling back
to `agent_rewards` only if `reward_list` is missing:

```python
reward = float(np.sum(reward_list))   # scalar step reward
```

It appends one dict to the rollout buffer:

```python
{ 'state', 'action', 'log_prob', 'value', 'reward', 'next_state' }
```

Unlike DQN's fixed ring buffer, this is a plain Python list that is **cleared
after every update** (on-policy).

---

## 6. Learning: `learn`

`code/core/ppo_agent.py` lines 161-225. Nothing happens until the buffer holds
`rollout_len` (= `--batch_size`, default 32) steps:

```python
if len(self.buffer) < self.rollout_len:
    return
```

### 6.1 Advantage estimation (GAE)
`_compute_gae` (lines 161-172) computes, walking backwards through the rollout:

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t     = delta_t + gamma * lambda * A_{t+1}
return_t = A_t + V(s_t)
```

The final step bootstraps with `V(next_state)` of the last transition
(lines 191-195). Advantages are then normalized to zero mean / unit std
(lines 201-203) for stability. This is where `gamma` (shared with DQN) and
`ppo_gae_lambda` come in — they provide the temporal discounting / aging that
DQN got from its Bellman bootstrap.

### 6.2 The clipped-surrogate update
For `ppo_epochs` passes over the same rollout (lines 205-222):

```python
ratio = exp(new_log_prob - old_log_prob)             # how much the policy moved
surr1 = ratio * advantage
surr2 = clamp(ratio, 1-clip, 1+clip) * advantage     # the "safety harness"
policy_loss = -min(surr1, surr2).mean()
value_loss  = MSE(V(s), return)
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

- `policy_loss` pushes probability toward actions with positive advantage, but
  the `clamp` to `[1-clip, 1+clip]` prevents any single update from moving the
  policy too far — this is the core PPO idea that keeps training stable.
- `value_loss` trains the critic toward the GAE returns.
- `-entropy_coef * entropy` rewards a less-certain policy, which is PPO's
  exploration mechanism (the analogue of DQN's epsilon).

After the epochs, `self.buffer = []` (line 225) discards the data — on-policy.

---

## 7. The interface no-ops

- `decay_epsilon_for_all_agents()` (lines 227-229): no-op. PPO explores via
  entropy, not epsilon.
- `get_average_epsilon()` (lines 231-233): returns 0.0 for logging
  compatibility.

---

## 8. End-to-end example of one rollout (rollout_len = 32)

1. For 32 consecutive LTIs: `select_actions` samples a 25-bit install mask and
   caches `(state, action, log_prob, value)`; the simulation installs and
   computes `reward_list`; `store_transitions` appends `sum(reward_list)`.
2. On the 32nd `learn()` call the buffer is full:
   - compute GAE advantages + returns over the 32 steps,
   - normalize advantages,
   - run `ppo_epochs` (4) gradient updates of the clipped objective,
   - clear the buffer.
3. Steps 1-2 repeat for the next 32 LTIs.

---

## 9. Relevant CLI parameters

PPO-only (prefixed `ppo_` in `code/main.py`):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--ppo_lr` | 3e-4 | Adam learning rate (note: far smaller than `--dqn_lr=0.5`) |
| `--ppo_clip` | 0.2 | clip range epsilon for the surrogate ratio |
| `--ppo_epochs` | 4 | gradient passes per rollout |
| `--ppo_entropy_coef` | 0.01 | exploration bonus weight |
| `--ppo_value_coef` | 0.5 | critic loss weight |
| `--ppo_gae_lambda` | 0.95 | GAE bias/variance trade-off |

Shared with DQN: `--gamma`, `--hidden_layers`, `--hidden_layer_size`,
`--batch_size` (reused as the rollout length), `--numberofFlowsPerAgent`.

Reward shaping (`--rewardAgingFactor`, `--spatialReward`) is unchanged and
applied by `RewardFunction`, exactly as for DQN.

---

## 10. Why PPO for this problem

- The factored Bernoulli policy makes the action space linear in `num_flows`
  instead of exponential, directly addressing DQN's main weakness.
- The critic baseline + clipping make policy-gradient learning stable even with
  the noisy, near-bandit reward signal.
- It is the de-facto modern policy-gradient baseline, so it is a natural
  comparison point against the value-based DQN and the non-deep bandit.
