# DQN Implementation (`--algorithm dqn`)

This document explains the default learner, `MultiAgentDQN`, defined in
`code/core/multiagent_dqn.py`. It is the reference design; the PPO and bandit
docs are written to contrast against it.

---

## 1. Where the learner sits in the framework

Every speculative learner is a *pluggable decision-maker*. Once per Learning
Time Interval (LTI), the simulation asks the learner **which flows to install
speculatively**, installs them, measures a reward, and lets the learner update
itself. The simulation (`code/simulation/speculative.py`,
`code/simulation/speculativereactive.py`) only ever calls this fixed interface:

| Method / attribute | Purpose |
| --- | --- |
| `num_states` | Used to size the initial `current_state` |
| `select_actions(current_state)` | Returns an action vector |
| `convert_actions_to_binary(action_list)` | Returns a binary install mask over all flows |
| `store_transitions(current_state, action_list, agent_rewards, next_state, reward_list=None)` | Hands the learner the result of the step |
| `learn()` | Trigger a parameter update |
| `decay_epsilon_for_all_agents()` | Anneal exploration |
| `get_average_epsilon()` / `get_network_info()` | Logging only |

The per-LTI orchestration in the simulation looks like this (simplified from
`_perform_speculative_learning`):

```python
action_list   = learner.select_actions(self.current_state)
agent_actions = learner.convert_actions_to_binary(action_list)   # binary mask over all flows
selected_flows = self._select_flows_based_on_actions(agent_actions)
# ... install selected flows, evicting low-age flows as needed ...

reward_list   = self.reward_function.calculate_rewards(...)        # ONE reward per flow
agent_rewards = self.reward_function.distribute_rewards_to_agents(reward_list, num_flows_per_agent)

next_state = torch.FloatTensor(action_list)
learner.store_transitions(self.current_state, action_list, agent_rewards, next_state, reward_list)
learner.learn()
learner.decay_epsilon_for_all_agents()
self.current_state = next_state
```

Key point: **the reward pipeline is identical for all three algorithms.**
`RewardFunction` applies `rewardAgingFactor` and `spatialReward` and produces a
per-flow `reward_list`; DQN consumes the aggregated `agent_rewards`, PPO/bandit
consume `reward_list`.

---

## 2. The problem as DQN frames it

DQN treats the task as a **multi-agent Markov Decision Process**:

- The `num_flows` controller flows are split into groups of
  `numberofFlowsPerAgent`. Each group is owned by **one independent DQN agent**.
- Each agent picks **one discrete action** out of `2^numberofFlowsPerAgent`.
  That integer's bits are the install/don't-install decisions for the agent's
  flows.

Running example used throughout these docs:

- `num_flows = 25`, `numberofFlowsPerAgent = 10`
- `num_agents = ceil(25 / 10) = 3`
- `num_actions = 2^10 = 1024`
- `num_states = ceil(25 / 10) = 3`

So we build **3 separate neural networks**, each mapping a 3-dim state to 1024
Q-values. See `MultiAgentDQN.__init__` (`code/core/multiagent_dqn.py` lines
129-146).

> Note on "state": the state fed in is literally the previous step's
> `action_list` (`next_state = torch.FloatTensor(action_list)`). There is no
> external environment observation, so the MDP is largely degenerate. This is
> exactly the observation that motivates the PPO and bandit alternatives.

---

## 3. The network: `NeuralNetwork`

`code/core/multiagent_dqn.py` lines 7-45.

- Input dimension = `num_states` (= 3 in our example).
- Output dimension = `num_actions` (= 1024). One Q-value per discrete action.
- Hidden layer sizing: if `--hidden_layer_size` is `None`, the sizes are
  interpolated between input and output with
  `size_i = num_states + ((num_actions - num_states) / (hidden_layers + 1)) * i`.
  With `hidden_layers = 2`, `num_states = 3`, `num_actions = 1024` the two
  hidden layers are ~344 and ~684 units. If `--hidden_layer_size` is set, all
  hidden layers use that uniform size.
- Weights initialized `N(0, 0.1)`; activations are ReLU; the output layer is
  linear (raw Q-values).

```python
def forward(self, x):
    for layer in self.layers:
        x = F.relu(layer(x))
    return self.out(x)   # shape (batch, num_actions)
```

---

## 4. One agent: `DQNAgent`

`code/core/multiagent_dqn.py` lines 47-127. This is textbook DQN with three
classic ingredients.

### 4.1 Two networks (the target trick)
`eval_net` is trained every step; `target_net` is a periodically frozen copy
used to compute stable learning targets. `target_net` is refreshed every
`dqn_target_replace_iter` learning steps (line 103-104).

### 4.2 Experience replay buffer
`self.memory` is a fixed NumPy array of shape
`(dqn_memory_capacity, num_states*2 + 2)`. Each row is one transition packed as:

```
[ state (num_states) | action (1) | reward (1) | next_state (num_states) ]
```

For our example a row has length `3 + 1 + 1 + 3 = 8`. `store_transition`
(lines 96-100) writes into the buffer in a ring (`index = counter % capacity`).

### 4.3 Epsilon-greedy action selection
`choose_action` (lines 83-94):

```python
if np.random.uniform() < self.epsilon:
    action = np.random.randint(0, num_actions)      # EXPLORE
else:
    action = argmax_a eval_net(state)               # EXPLOIT
```

`epsilon` starts at `dqn_epsilon_start` (1.0) and decays multiplicatively each
LTI by `dqn_epsilon_decay` down to `dqn_epsilon_end` (`decay_epsilon`, line 75-77).

### 4.4 The Q-learning update
`learn` (lines 102-127) samples a random minibatch of `batch_size` transitions
and applies the Bellman update:

```
q_eval   = eval_net(state).gather(action)              # Q(s,a)
q_next   = target_net(next_state).max(1)               # max_a' Q_target(s', a')
q_target = reward + gamma * q_next
loss     = MSE(q_eval, q_target)
```

Then Adam backprops `loss` into `eval_net` only.

---

## 5. The coordinator: `MultiAgentDQN`

`code/core/multiagent_dqn.py` lines 129-220. This wraps the 3 agents and
implements the shared interface.

### 5.1 `select_actions(current_state)` (lines 148-155)
Loops over the 3 agents, each returns one integer in `[0, 1023]`:

```
action_list = [5, 0, 257]      # one integer per agent
```

### 5.2 `convert_actions_to_binary(action_list)` (lines 157-169)
Expands each integer into its `numberofFlowsPerAgent` low bits and lays them out
across all flows. Worked example for agent 0 with action `5`:

```
5 in binary (LSB first, 10 bits) = 1 0 1 0 0 0 0 0 0 0
flows 0..9 install mask          = [1,0,1,0,0,0,0,0,0,0]
```

The 3 agents' bit-expansions are concatenated into a single length-25 mask:

```
agent_actions = [1,0,1,0,0,0,0,0,0,0,  0,0,0,...,0,  1,0,0,0,0]   # length 25
```

The simulation then installs the flows where the mask is 1.

### 5.3 `store_transitions(...)` (lines 171-179)
Splits the aggregated `agent_rewards` (one scalar per agent) back to each agent
and calls each agent's `store_transition`. **`reward_list` is accepted but
ignored** — that argument exists only so PPO/bandit can share the interface.

### 5.4 `learn()` (lines 181-185)
Gated: nothing happens until the replay buffer is full
(`memory_counter > memory_capacity`). After that, every agent does one
minibatch update per LTI.

### 5.5 `decay_epsilon_for_all_agents()` (lines 187-190)
Decays epsilon on all 3 agents once per LTI.

---

## 6. End-to-end example of a single LTI

With `num_flows=25`, 3 agents, `epsilon=0.3`:

1. `select_actions(state=[a,b,c])` → e.g. `[5, 0, 257]` (mostly exploit, maybe a random pick).
2. `convert_actions_to_binary([5,0,257])` → length-25 0/1 mask.
3. Simulation installs masked flows (evicting low-age flows if the table is full).
4. `reward_function.calculate_rewards(...)` → `reward_list` (length 25), e.g. hit-count deltas + spatial bonus.
5. `distribute_rewards_to_agents(reward_list, 10)` → `agent_rewards = [r0, r1, r2]` (sum of each group's flow rewards).
6. `store_transitions(state, [5,0,257], [r0,r1,r2], next_state=[5,0,257], reward_list)`:
   each agent appends `[state | action | reward | next_state]` to its buffer.
7. `learn()`: if buffers are full, each agent samples a minibatch and does a Bellman update.
8. `decay_epsilon_for_all_agents()`: `epsilon *= dqn_epsilon_decay`.
9. `current_state = next_state`.

---

## 7. Relevant CLI parameters

DQN-only (prefixed `dqn_` in `code/main.py`):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dqn_epsilon_start` | 1.0 | initial exploration rate |
| `--dqn_epsilon_end` | 0.01 | floor exploration rate |
| `--dqn_epsilon_decay` | 0.995 | per-LTI multiplicative decay |
| `--dqn_target_replace_iter` | 100 | target-net refresh period (learning steps) |
| `--dqn_memory_capacity` | 500 | replay buffer size |
| `--dqn_lr` | 0.5 | Adam learning rate |

Shared with other algorithms: `--numberofFlowsPerAgent`, `--gamma`,
`--hidden_layers`, `--hidden_layer_size`, `--batch_size`.

---

## 8. Strengths and weaknesses (for context)

- Strength: principled value-based RL with a stable target and replay.
- Weakness 1: the action head is `2^numberofFlowsPerAgent` (1024 here, but it
  explodes — e.g. 20 flows/agent → ~1M outputs). This is the central
  scalability complaint.
- Weakness 2: the "state" is a fake signal, so the `gamma * max Q(next)`
  bootstrap is bootstrapping off noise. The problem is really a
  (combinatorial) bandit, which is what the other two learners exploit.
