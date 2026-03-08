import numpy as np
import math

class RewardFunction:
    def __init__(self, args):
        self.args = args
        self.reward_aging_factor = float(args.rewardAgingFactor)
        self.spatial_reward_factor = float(args.spatialReward)
        self.switch_table = None  # Will be set during simulation
        
    def calculate_rewards(self, selected_flows, controller_table, old_controller_table, step, spatial_window_size):
        """Calculate rewards for all flows"""
        reward = np.zeros(len(controller_table))
        
        # Apply reward aging
        for i in range(len(controller_table)):
            controller_table.iloc[i, controller_table.columns.get_loc('accumulated_reward')] = self.reward_aging_factor * controller_table.iloc[i]['accumulated_reward']
        
        # Calculate hit rewards
        for i in range(len(controller_table)):
            if old_controller_table.iloc[i]['hit_count'] < controller_table.iloc[i]['hit_count']:
                reward[i] += controller_table.iloc[i]['hit_count'] - old_controller_table.iloc[i]['hit_count']
        
        # # Calculate miss rewards
        # for i in range(len(controller_table)):
        #     if old_controller_table.iloc[i]['miss_count'] < controller_table.iloc[i]['miss_count']:
        #         reward[i] += controller_table.iloc[i]['miss_count'] - old_controller_table.iloc[i]['miss_count']
        
        # Calculate spatial rewards (only if switch table is available)
        if self.switch_table is not None:
            reward = self._calculate_spatial_rewards(controller_table, old_controller_table, reward, spatial_window_size)
        
        # Accumulate rewards
        for i in range(len(controller_table)):
            controller_table.iloc[i, controller_table.columns.get_loc('accumulated_reward')] += reward[i]
        
        return reward
    
    def _find_controller_index(self, flow_source, flow_destination, controller_table):
        """Find the controller index for a given flow"""
        for j in range(len(controller_table)):
            if (controller_table.iloc[j]['Source'] == flow_source and 
                controller_table.iloc[j]['Destination'] == flow_destination):
                return j
        return -1
    
    def _is_flow_processed(self, controller_number, controller_table, old_controller_table):
        """Check if a flow has been processed and its counter has increased"""
        # Check if this flow has been processed (counter > 0)
        if controller_table.iloc[controller_number]['total_packet_count'] == 0:
            return False
            
        # Check if the counter has increased since last iteration
        return (old_controller_table.iloc[controller_number]['total_packet_count'] < 
                controller_table.iloc[controller_number]['total_packet_count'])
    
    def _calculate_base_reward(self, controller_number, controller_table):
        """Calculate base reward for a flow based on hit and miss counts"""
        # return (controller_table.iloc[controller_number]['hit_count'] + 
        #         controller_table.iloc[controller_number]['miss_count'])
        return controller_table.iloc[controller_number]['hit_count']
    
    def _distribute_spatial_rewards(self, controller_number, spatial_reward, reward, spatial_window_size):
        """Distribute spatial rewards to neighboring flows"""
        j = 0
        while j < math.ceil(spatial_window_size / 2):
            # Forward neighbors
            if (controller_number + j) < len(reward):
                reward[controller_number + j] += spatial_reward
            
            # Backward neighbors
            if (controller_number - j) >= 0:
                reward[controller_number - j] += spatial_reward
            
            j += 1
            # Apply spatial reward factor (decay)
            spatial_reward *= self.spatial_reward_factor
        
        return reward
    
    def _calculate_spatial_rewards(self, controller_table, old_controller_table, reward, spatial_window_size):
        """Calculate spatial rewards for neighboring flows based on original implementation"""
        # Calculate spatial rewards for flows in switch table
        for i in range(len(self.switch_table)):
            flow_source = self.switch_table.iloc[i]['Source']
            flow_destination = self.switch_table.iloc[i]['Destination']
            
            # Find the controller index for this flow
            controller_number = self._find_controller_index(flow_source, flow_destination, controller_table)
            
            if controller_number == -1:
                continue
                
            # Check if this flow has been processed and counter increased
            if self._is_flow_processed(controller_number, controller_table, old_controller_table):
                # Calculate base reward for this flow
                base_reward = self._calculate_base_reward(controller_number, controller_table)
                
                spatial_reward = base_reward
                
                # Distribute spatial rewards to neighboring flows
                reward = self._distribute_spatial_rewards(controller_number, spatial_reward, reward, spatial_window_size)
        
        return reward
    
    def distribute_rewards_to_agents(self, reward_list, number_of_flows_per_agent):
        """Distribute rewards to individual agents based on flow grouping"""
        # Calculate number of agents needed
        total_flows = len(reward_list)
        num_agents = math.ceil(total_flows / number_of_flows_per_agent)
        
        # Initialize agent rewards array
        agent_rewards = np.zeros(num_agents)
        
        # Distribute rewards to agents
        for flow_index, flow_reward in enumerate(reward_list):
            # Determine which agent this flow belongs to
            agent_index = flow_index // number_of_flows_per_agent
            
            # Add flow reward to the corresponding agent
            agent_rewards[agent_index] += flow_reward
        
        return agent_rewards
