import logging
import pandas as pd
import torch
import math
from collections import deque


class SpeculativeReactiveSimulation:
    def __init__(self, args, controller_table, switch_table, priority_policy, reward_function, learner, data_collector, logger=None):
        self.args = args
        self.controller_table = controller_table
        self.old_controller_table = controller_table.copy()
        self.switch_table = switch_table
        self.priority_policy = priority_policy
        self.reward_function = reward_function
        self.learner = learner
        self.data_collector = data_collector
        self.logger = logger
        
        self.table_size = args.tablesize
        self.learning_time_interval = float(args.LTI)
        # Fix LFU time interval calculation - should be multiple of LTI
        self.lfu_time_interval = float(args.LFUTimeInterval) * float(args.RTI)
        self.reactive_time_interval = float(args.RTI)
        self.num_flows_per_agent = args.numberofFlowsPerAgent
        
        # Initialize tracking variables
        self.packet_counter = 0
        self.trace_counter = 0
        self.least_frequently_used_counter = 0
        self.graph_counter = 0
        self.flow_rate_counter = 0
        
        # Initialize queues and lists
        self.flow_queue = deque()
        self.new_flow_list = []
        
        # State tracking
        self.current_state = torch.zeros(self.learner.num_states)
        self.action_history = []
        
    def run(self, dataset, value):
        """Main simulation loop for speculative-reactive mode"""
        lti_start_time = float(value.iloc[0]["Time"])  # Start time of first LTI
        lti_start_packet = 0  # Track packet count for LTI logging
        
        while True:
            # Process packets until learning time interval
            while True:
                self._process_single_packet(dataset, value)
                
                # Check if we should break for learning
                if self._should_break_for_learning(value):
                    break
                    
                # Check if we should stop simulation
                if self._should_stop_simulation(value):
                    # Record final LTI metrics
                    current_time = float(value.iloc[self.packet_counter].iloc[0])
                    self.data_collector.record_lti_metrics(lti_start_time, current_time, self.switch_table)
                    return
            
            # Apply aging to switch table
            self.switch_table, self.controller_table = self.priority_policy.apply_aging_to_switch_table(
                self.switch_table, self.controller_table
            )
            
            # Perform speculative learning and flow installation
            self._perform_speculative_learning(dataset, value)
            
            # Record LTI metrics and update performance metrics
            current_time = float(value.iloc[self.packet_counter].iloc[0])
            
            # Log LTI completion (before resetting counters)
            if self.logger:
                speculative_flows = sum(1 for _, flow in self.switch_table.iterrows() if 'is_speculative' in flow and flow['is_speculative']) if 'is_speculative' in self.switch_table.columns else 0
                reactive_flows = len(self.switch_table) - speculative_flows
                self.logger.lti_info(self.data_collector.current_lti, 
                                   f"Completed LTI. Packets: {self.packet_counter - lti_start_packet}, "
                                   f"Switch table size: {len(self.switch_table)}, "
                                   f"Speculative flows: {speculative_flows}, "
                                   f"Reactive flows: {reactive_flows}, "
                                   f"Evicted flows: {self.data_collector.lti_evicted_flows}")
            
            # Record LTI metrics (this resets counters)
            self.data_collector.record_lti_metrics(lti_start_time, current_time, self.switch_table)
            
            lti_start_time = current_time  # Start time for next LTI
            lti_start_packet = self.packet_counter  # Track packet count for next LTI
            self._update_performance_metrics(value)
    
    def _process_single_packet(self, dataset, value):
        """Process a single packet with both reactive and speculative components"""
        # Create packet data
        packet_data = pd.DataFrame(data={'Source': [dataset.iloc[self.packet_counter, 0]], 
                                       'Destination': [dataset.iloc[self.packet_counter, 1]]})
        
        # Get current time for data collection
        current_time = float(value.iloc[self.packet_counter].iloc[0])
        
        # Process flow queue (reactive component)
        self._process_flow_queue(value)
        
        # Check if packet matches switch table
        packet_found, is_reactive_hit = self._check_packet_match(packet_data)
        
        if packet_found:
            self._handle_packet_hit(packet_data)
            # Record hit with proper type tracking
            self.data_collector.record_packet_processing(current_time, was_hit=True, is_speculative=True, is_reactive_hit=is_reactive_hit)
        else:
            self._handle_packet_miss(packet_data, value)
            # Record mi ss
            self.data_collector.record_packet_processing(current_time, was_hit=False, is_speculative=True)
        
        self.packet_counter += 1
    
    def _process_flow_queue(self, value):
        """Process flow installation queue (reactive component)"""
        if len(self.flow_queue) > 0:
            current_time = float(value.iloc[self.packet_counter].iloc[0])
            # queue_head = self.flow_queue[0]
            
            while len(self.flow_queue) > 0 and self.flow_queue[0].iloc[0]['switchcopy'] <= current_time:
                # Install flow from queue
                queue_head = self.flow_queue[0]
                self.flow_queue.popleft()
                self._install_flow_from_queue(queue_head)
    
    def _check_packet_match(self, packet_data): # this can be optimized
        """Check if packet matches any flow in switch table - returns (found, is_reactive_hit)"""
        for j in range(len(self.switch_table)):
            if (self.switch_table.iloc[j]['Source'] == packet_data.iloc[0]['Source'] and 
                self.switch_table.iloc[j]['Destination'] == packet_data.iloc[0]['Destination']):
                # Check if this is a reactive or speculative flow
                is_reactive_hit = not ('is_speculative' in self.switch_table.columns and self.switch_table.iloc[j]['is_speculative'])
                return True, is_reactive_hit
        return False, False
    
    def _handle_packet_hit(self, packet_data):
        """Handle successful packet match"""
        for j in range(len(self.controller_table)):
            if (self.controller_table.iloc[j]['Source'] == packet_data.iloc[0]['Source'] and 
                self.controller_table.iloc[j]['Destination'] == packet_data.iloc[0]['Destination']):
                self.controller_table.iloc[j, self.controller_table.columns.get_loc('hit_count')] += 1
                self.controller_table.iloc[j, self.controller_table.columns.get_loc('was_hit_this_iteration')] = 1
                self.controller_table.iloc[j, self.controller_table.columns.get_loc('total_packet_count')] += 1
                break
    
    def _handle_packet_miss(self, packet_data, value):
        """Handle packet miss"""
        for j in range(len(self.controller_table)):
            if (self.controller_table.iloc[j]['Source'] == packet_data.iloc[0]['Source'] and 
                self.controller_table.iloc[j]['Destination'] == packet_data.iloc[0]['Destination']):
                self.controller_table.iloc[j, self.controller_table.columns.get_loc('miss_count')] += 1
                self.controller_table.iloc[j, self.controller_table.columns.get_loc('total_packet_count')] += 1
                
                # Add to installation queue (reactive component)
                self._add_to_installation_queue(packet_data, value)
                break
    
    def _add_to_installation_queue(self, packet_data, value):
        """Add flow to installation queue"""
        flow_entry = packet_data.copy()
        flow_entry['flow_age'] = self.args.reset_age
        flow_entry['switchcopy'] = float(value.iloc[self.packet_counter].iloc[0]) + self.reactive_time_interval
        flow_entry['is_speculative'] = False  # Reactive flows
        flow_entry = flow_entry[['Source', 'Destination', 'flow_age', 'switchcopy', 'is_speculative']]
        self.flow_queue.append(flow_entry)
    
    def _install_flow_from_queue(self, flow_entry):
        """Install flow from queue to switch table"""
        # Check if we need to evict flows to make space
        if len(self.switch_table) >= self.table_size:
            # Evict flows with age <= 0.4 to make space
            self.switch_table, evicted_count = self.priority_policy.evict_flows_with_low_age_optimized(
                self.switch_table, self.controller_table, 1, True, self.data_collector
            )
            
            # Check if eviction was successful
            if evicted_count == 0:
                print(f"Table is full ({len(self.switch_table)}/{self.table_size}) and no flows can be evicted, cannot install reactive flow")
                return
        
        # Final table size enforcement - ensure we don't exceed table capacity
        current_table_size = len(self.switch_table)
        if current_table_size >= self.table_size:
            print(f"Table is full ({current_table_size}/{self.table_size}), cannot install reactive flow")
            return
        
        # Install new flow
        flow_entry.iloc[0, flow_entry.columns.get_loc('flow_age')] = self.args.reset_age  # Set age to reset_age
        flow_entry.iloc[0, flow_entry.columns.get_loc('is_speculative')] = False  # Reactive flow
        self.switch_table = pd.concat([self.switch_table, flow_entry], ignore_index=True)
        self.switch_table = self.switch_table.drop_duplicates(subset=['Source', 'Destination'], keep='last')
        
        # Record reactive flow installation
        flow_data = {
            'Source': flow_entry.iloc[0]['Source'],
            'Destination': flow_entry.iloc[0]['Destination'],
            'flow_age': flow_entry.iloc[0]['flow_age']
        }
        self.data_collector.record_flow_installation(flow_data, is_speculative=False)
    
    def _perform_speculative_learning(self, dataset, value):
        """Perform speculative learning and flow installation"""
        # Select actions for all agents
        action_list = self.learner.select_actions(self.current_state)
        self.action_history.append(action_list)
        
        # Convert actions to binary flow decisions
        agent_actions = self.learner.convert_actions_to_binary(action_list)
        
        # Select flows based on agent decisions
        selected_flows = self._select_flows_based_on_actions(agent_actions)
        
        # Check if there are flows with age < threshold that can be evicted
        evictable_flows = self.priority_policy.count_evictable_flows(self.switch_table)
        
        # Check if we can install speculative flows
        available_slots = self.table_size - len(self.switch_table)
        
        if evictable_flows > 0 or available_slots > 0:
            self._install_speculative_flows(selected_flows)
        
        # Calculate rewards
        # Calculate spatial window size based on original formula: w = 2*(math.ceil(len(controller)/flowresult))
        # where flowresult is the total number of flows that have been processed
        total_flows_processed = self.controller_table['total_packet_count'].sum()
        if total_flows_processed > 0:
            spatial_window_size = 2 * math.ceil(len(self.controller_table) / total_flows_processed)
        else:
            spatial_window_size = 10  # Default value
        
        # Update reward function with current switch table
        self.reward_function.switch_table = self.switch_table
        
        reward_list = self.reward_function.calculate_rewards(
            selected_flows, self.controller_table, self.old_controller_table, 
            float(value.iloc[self.packet_counter].iloc[0]), spatial_window_size
        )
        
        
        # Distribute rewards to agents
        agent_rewards = self.reward_function.distribute_rewards_to_agents(
            reward_list, self.num_flows_per_agent
        )
        
        # self.logger.info(f"total reward {sum(agent_rewards)}")
        # if sum(agent_rewards) > 0:
        #     print("Agent rewards: ", agent_rewards)

        # Store transitions for learning
        next_state = torch.FloatTensor(action_list)
        self.learner.store_transitions(
            self.current_state, action_list, agent_rewards, next_state, reward_list
        )
        
        # Trigger learning
        self.learner.learn()
        
        # Decay epsilon for exploration vs exploitation balance
        self.learner.decay_epsilon_for_all_agents()
        
        # Update state
        self.current_state = next_state
        
        # Update old controller table
        self.old_controller_table = self.controller_table.copy()
        
        # Record reward data
        current_time = float(value.iloc[self.packet_counter].iloc[0])
        for reward in agent_rewards:
            self.data_collector.record_reward(reward, current_time)
        
        # Record total reward for LTI metrics
        total_reward = sum(agent_rewards)
        self.data_collector.record_lti_reward(total_reward)
    
    def _select_flows_based_on_actions(self, agent_actions):
        """Select flows based on agent binary decisions"""
        # Create flow selection matrix
        flow_selection = pd.DataFrame()
        flow_selection['No.'] = range(len(self.controller_table))
        flow_selection['Source'] = self.controller_table['Source'].values
        flow_selection['Destination'] = self.controller_table['Destination'].values
        
        # Filter based on agent actions
        selected_flows = flow_selection.iloc[agent_actions == 1]
        return selected_flows
    
    def _install_speculative_flows(self, selected_flows):
        """Install speculative flows in switch table"""
        # Sort flows by reward (ascending) for priority installation
        flow_priorities = selected_flows.copy()
        flow_priorities = flow_priorities.merge(
            self.controller_table[['Source', 'Destination', 'accumulated_reward']], 
            on=['Source', 'Destination']
        )
        flow_priorities = flow_priorities.sort_values(by='accumulated_reward', ascending=False)
        
        # Install flows up to table capacity
        flows_to_install = flow_priorities.head(self.table_size)
        
        # Calculate how many flows we need to evict upfront
        current_table_size = len(self.switch_table)
        flows_to_install_count = len(flows_to_install)
        
        # Use the counting method to find actual available space through eviction
        evictable_flows = self.priority_policy.count_evictable_flows(self.switch_table)
        available_space = self.table_size - current_table_size + evictable_flows
        
        if flows_to_install_count > available_space:
            # We need to evict flows to make space
            flows_to_evict = available_space
            
            # Use optimized method to evict all needed flows in one shot
            self.switch_table, evicted_count = self.priority_policy.evict_flows_with_low_age_optimized(
                self.switch_table, self.controller_table, flows_to_evict, False, self.data_collector
            )
            # If we couldn't evict enough flows, reduce the number of flows to install
            flows_to_install = flows_to_install.head(available_space)
            print(f"Could only evict {evicted_count} flows, installing {len(flows_to_install)} flows instead of {flows_to_install_count}")
        else:
            # No eviction needed, set evicted_count to 0
            evicted_count = 0
        
        # Final table size enforcement - ensure we don't exceed table capacity
        current_table_size = len(self.switch_table)
        if current_table_size >= self.table_size:
            # Table is still full, reduce flows to install
            remaining_space = self.table_size - current_table_size
            if remaining_space <= 0:
                print(f"Table is full ({current_table_size}/{self.table_size}), no flows can be installed")
                return
            flows_to_install = flows_to_install.head(remaining_space)
            print(f"Table enforcement: only installing {remaining_space} flows to maintain size limit")
        
        # Now install all flows at once (no need to check table size in loop)
        for _, flow in flows_to_install.iterrows():
            # Install the speculative flow
            new_flow = {
                'Source': flow['Source'],
                'Destination': flow['Destination'],
                'flow_age': self.args.speculative_reset_age,  # Initial age for speculative flows
                'is_speculative': True,
                'hit_count': 0
            }
            self.switch_table = pd.concat([self.switch_table, pd.DataFrame([new_flow])], ignore_index=True)
            
            # Record flow installation
            self.data_collector.record_flow_installation(new_flow, is_speculative=True)
    
    def _calculate_spatial_window_size(self):
        """Calculate spatial window size for reward calculation"""
        # This is a simplified version - original has complex calculation
        return 2
    
    def _should_break_for_learning(self, value):
        """Check if we should break for learning interval"""
        if self.packet_counter >= len(value) - 1:
            return True
            
        current_time = float(value.iloc[self.packet_counter].iloc[0])
        learning_start_time = float(value.iloc[self.trace_counter].iloc[0])
        
        return (current_time - learning_start_time) > self.learning_time_interval
    
    def _should_stop_simulation(self, value):
        """Check if simulation should stop"""
        if self.packet_counter >= len(value) - 1:
            return True
            
        current_time = float(value.iloc[self.packet_counter].iloc[0])
        
        # Hardcoded to dataset 1
        return current_time > self.args.simulation_time
    
    def _update_performance_metrics(self, value):
        """Update performance tracking metrics"""
        # Reset counters for next interval
        self.trace_counter = self.packet_counter
        
        # Update LFU counter
        if ((float(value.iloc[self.packet_counter].iloc[0]) - float(value.iloc[self.least_frequently_used_counter].iloc[0])) > self.lfu_time_interval):
            self.least_frequently_used_counter = self.packet_counter
            self.controller_table['total_packet_count'] = 0
        
        # Final table size enforcement - ensure table never exceeds configured size
        current_table_size = len(self.switch_table)
        if current_table_size > self.table_size:
            print(f"CRITICAL: Table size {current_table_size} exceeds limit {self.table_size}. Enforcing size limit...")
            # Force evict excess flows based on packet count (least frequently used first)
            excess_flows = current_table_size - self.table_size
            self.switch_table, evicted_count = self.priority_policy.evict_flows_with_low_age_optimized(
                self.switch_table, self.controller_table, excess_flows, True, self.data_collector
            )
            print(f"Table size enforcement: evicted {evicted_count} flows. New table size: {len(self.switch_table)}")
