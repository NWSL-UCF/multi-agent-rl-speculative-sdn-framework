import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NeuralNetwork(nn.Module):
    def __init__(self, num_states, num_actions, hidden_layers=1, hidden_layer_size=None):
        super(NeuralNetwork, self).__init__()
        
        # Calculate hidden layer sizes
        self.hidden_layer_sizes = []
        
        if hidden_layer_size is None:
            # Use current implementation: dynamic sizing with formula
            # ith hidden layer size = x + ((y - x) / (hidden_layers + 1)) * i
            # where x = num_states, y = num_actions, i starts at 1
            for i in range(1, hidden_layers + 1):
                layer_size = num_states + ((num_actions - num_states) / (hidden_layers + 1)) * i
                # Round to nearest integer
                layer_size = round(layer_size)
                self.hidden_layer_sizes.append(layer_size)
        else:
            # Use uniform size for all hidden layers
            self.hidden_layer_sizes = [hidden_layer_size] * hidden_layers
        
        # Build the network layers
        self.layers = nn.ModuleList()
        input_size = num_states
        
        for hidden_size in self.hidden_layer_sizes:
            layer = nn.Linear(input_size, hidden_size)
            layer.weight.data.normal_(0, 0.1)
            self.layers.append(layer)
            input_size = hidden_size
        
        # Output layer
        self.out = nn.Linear(input_size, num_actions)
        self.out.weight.data.normal_(0, 0.1)
        
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        action_value = self.out(x)
        return action_value

class DQNAgent:
    def __init__(self, num_states, num_actions, args, device=None):
        # Set device (default to CPU if not provided)
        self.device = device if device is not None else torch.device("cpu")
        
        hidden_layers = getattr(args, 'hidden_layers', 1)
        hidden_layer_size = getattr(args, 'hidden_layer_size', None)
        self.eval_net = NeuralNetwork(num_states, num_actions, hidden_layers, hidden_layer_size).to(self.device)
        self.target_net = NeuralNetwork(num_states, num_actions, hidden_layers, hidden_layer_size).to(self.device)
        
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((args.memory_capacity, num_states*2+2))
        
        self.learning_rate = float(args.LR)
        # Epsilon decay parameters
        self.epsilon_start = float(args.epsilon_start)
        self.epsilon_end = float(args.epsilon_end)
        self.epsilon_decay = float(args.epsilon_decay)
        self.epsilon = self.epsilon_start  # Start with high exploration
        self.gamma = float(args.gamma)
        self.target_replace_iter = args.target_replace_iter
        self.memory_capacity = args.memory_capacity
        self.batch_size = args.batch_size
        
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
    
    def decay_epsilon(self):
        """Decay epsilon for exploration vs exploitation balance"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_current_epsilon(self):
        """Get current epsilon value for monitoring"""
        return self.epsilon
        
    def choose_action(self, state):
        state_tensor = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.device)
        
        if np.random.uniform() < self.epsilon:
            # EXPLORATION: Try random actions to discover new strategies
            action = np.random.randint(0, self.eval_net.out.out_features)
        else:
            # EXPLOITATION: Use learned knowledge to choose best action
            with torch.no_grad():  # No gradient computation needed for inference
                action_value = self.eval_net.forward(state_tensor)
                action = torch.max(action_value, 1)[1].cpu().data.numpy()[0]
        return action
    
    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        batch_memory = self.memory[sample_index, :]
        
        # Get input size from the first layer
        input_size = self.eval_net.layers[0].in_features if self.eval_net.layers else self.eval_net.out.in_features
        
        # Move tensors to the appropriate device
        batch_state = torch.FloatTensor(batch_memory[:, :input_size]).to(self.device)
        batch_action = torch.LongTensor(batch_memory[:, input_size:input_size+1]).to(self.device)
        batch_reward = torch.FloatTensor(batch_memory[:, input_size+1:input_size+2]).to(self.device)
        batch_next_state = torch.FloatTensor(batch_memory[:, -input_size:]).to(self.device)
        
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        
        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class MultiAgentDQN:
    def __init__(self, args, controller_table):
        self.args = args
        self.num_flows = len(controller_table)
        self.num_flows_per_agent = args.numberofFlowsPerAgent
        self.num_agents = math.ceil(self.num_flows / self.num_flows_per_agent)
        self.num_actions = pow(2, self.num_flows_per_agent)
        self.num_states = math.ceil(self.num_flows / self.num_flows_per_agent)
        
        # GPU support: use device from args or detect automatically
        if hasattr(args, 'selected_device'):
            self.device = torch.device(args.selected_device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.agents = []
        for agent in range(self.num_agents):
            self.agents.append(DQNAgent(self.num_states, self.num_actions, args, self.device))
    
    def select_actions(self, current_state):
        """Select actions for all agents"""
        action_list = np.zeros(self.num_agents)
        
        for agent, dqn in enumerate(self.agents):
            action_list[agent] = dqn.choose_action(current_state)
            
        return action_list
    
    def convert_actions_to_binary(self, action_list):
        """Convert agent actions to binary flow decisions"""
        agent_actions = np.zeros(self.num_flows, int)
        
        for i in range(len(action_list)):
            temp_action = action_list[i]
            for j in range(self.num_flows_per_agent):
                if (i * self.num_flows_per_agent + j == len(agent_actions)):
                    break
                agent_actions[i * self.num_flows_per_agent + j] = temp_action % 2
                temp_action = temp_action // 2
                
        return agent_actions
    
    def store_transitions(self, current_state, action_list, agent_rewards, next_state, reward_list=None):
        """Store transitions for all agents

        ``reward_list`` (per-flow semi-bandit feedback) is accepted for a common
        learner interface but is unused by DQN, which learns from the aggregated
        per-agent ``agent_rewards``.
        """
        for agent, dqn in enumerate(self.agents):
            dqn.store_transition(current_state, action_list[agent], agent_rewards[agent], next_state)
    
    def learn(self):
        """Trigger learning for all agents"""
        if self.agents[0].memory_counter > self.agents[0].memory_capacity:
            for dqn in self.agents:
                dqn.learn()
    
    def decay_epsilon_for_all_agents(self):
        """Decay epsilon for all agents to balance exploration vs exploitation"""
        for dqn in self.agents:
            dqn.decay_epsilon()
    
    def get_average_epsilon(self):
        """Get average epsilon value across all agents for monitoring"""
        if not self.agents:
            return 0.0
        return sum(dqn.get_current_epsilon() for dqn in self.agents) / len(self.agents)
    
    def get_network_info(self):
        """Get network architecture information for logging"""
        if not self.agents:
            return {}
        
        # Get network info from the first agent (all agents have same architecture)
        agent = self.agents[0]
        network_info = {
            'num_agents': self.num_agents,
            'input_size': self.num_states,
            'output_size': self.num_actions,
            'num_hidden_layers': len(agent.eval_net.hidden_layer_sizes),
            'hidden_layer_sizes': agent.eval_net.hidden_layer_sizes,
            'total_parameters_per_agent': sum(p.numel() for p in agent.eval_net.parameters()),
            'total_trainable_parameters': sum(p.numel() for p in agent.eval_net.parameters()) * self.num_states,
            'epsilon_start': self.agents[0].epsilon_start,
            'epsilon_end': self.agents[0].epsilon_end,
            'epsilon_decay': self.agents[0].epsilon_decay,
            'current_epsilon': self.get_average_epsilon(),
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available()
        }
        return network_info
