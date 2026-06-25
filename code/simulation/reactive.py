import pandas as pd
from collections import deque
class ReactiveSimulation:
    def __init__(self, args, controller_table, switch_table, priority_policy, reward_function, data_collector, logger=None):
        self.args = args
        self.controller_table = controller_table
        self.switch_table = switch_table
        self.priority_policy = priority_policy
        self.reward_function = reward_function
        self.data_collector = data_collector
        self.logger = logger
        
        self.table_size = args.tablesize
        self.learning_time_interval = float(args.LTI)
        # Fix LFU time interval calculation - should be multiple of LTI
        self.lfu_time_interval = float(args.LFUTimeInterval) * float(args.RTI)
        self.reactive_time_interval = float(args.RTI)
        
        # Initialize tracking variables
        self.packet_counter = 0
        self.trace_counter = 0
        self.least_frequently_used_counter = 0
        self.graph_counter = 0
        self.flow_rate_counter = 0
        
        # Initialize queues and lists
        self.flow_queue = deque()
        self.new_flow_list = []
        
    def run(self, dataset, value):
        """Main simulation loop for reactive mode"""
        lti_start_time = float(value.iloc[0].iloc[0])  # Start time of first LTI
        lti_start_packet = 0  # Track packet count for LTI logging
        
        while True: # LTI loop
            # Process packets until learning time interval
            while True: # Packet processing loop
                self._process_single_packet(dataset, value)
                
                # Check if we should break for learning
                if self._should_break_for_learning(value):
                    break 
                    # this break is for learning interval 
                    # even though reactive mode does not use learning interval
                    # we need to break for learning interval to update the performance metrics
                
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
            
            # Record LTI metrics and update performance metrics
            current_time = float(value.iloc[self.packet_counter].iloc[0])
            
            # Log LTI completion (before resetting counters)
            if self.logger:
                self.logger.lti_info(self.data_collector.current_lti, 
                                   f"Completed LTI. Packets: {self.packet_counter - lti_start_packet}, "
                                   f"Switch table size: {len(self.switch_table)}, "
                                   f"Evicted flows: {self.data_collector.lti_evicted_flows}")
            
            self.data_collector.record_lti_metrics(lti_start_time, current_time, self.switch_table)
            
            lti_start_time = current_time  # Start time for next LTI
            lti_start_packet = self.packet_counter  # Track packet count for next LTI
            self._update_performance_metrics(value)
    
    def _process_single_packet(self, dataset, value):
        """Process a single packet"""
        # Create packet data
        packet_data = pd.DataFrame(data={'Source': [dataset.iloc[self.packet_counter, 0]], 
                                       'Destination': [dataset.iloc[self.packet_counter, 1]]})
        
        # Get current time for data collection
        current_time = float(value.iloc[self.packet_counter].iloc[0])
        
        # Process flow queue
        self._process_flow_queue(value)
        
        # Check if packet matches switch table
        packet_found = self._check_packet_match(packet_data)
        
        if packet_found:
            self._handle_packet_hit(packet_data)
            # Record hit (all hits in reactive mode are against reactive flows)
            self.data_collector.record_packet_processing(current_time, was_hit=True, is_speculative=False, is_reactive_hit=True)
        else:
            self._handle_packet_miss(packet_data, value)
            # Record miss
            self.data_collector.record_packet_processing(current_time, was_hit=False, is_speculative=False)
        
        self.packet_counter += 1
    
    def _process_flow_queue(self, value):
        """Process flow installation queue"""
        if len(self.flow_queue) > 0:
            current_time = float(value.iloc[self.packet_counter].iloc[0])
            # queue_head = self.flow_queue[0]
            
            while len(self.flow_queue) > 0 and self.flow_queue[0].iloc[0]['switchcopy'] <= current_time:
                # Install flow from queue
                queue_head = self.flow_queue[0]
                self.flow_queue.popleft()
                self._install_flow_from_queue(queue_head)
    
    def _check_packet_match(self, packet_data):
        """Check if packet matches any flow in switch table - OPTIMIZED"""
        if len(self.switch_table) == 0:
            return False
            
        packet_source = packet_data.iloc[0]['Source']
        packet_dest = packet_data.iloc[0]['Destination']
        
        # Use vectorized operations for faster matching
        matches = (self.switch_table['Source'] == packet_source) & (self.switch_table['Destination'] == packet_dest)
        return matches.any()
    
    def _handle_packet_hit(self, packet_data):
        """Handle successful packet match - OPTIMIZED"""
        packet_source = packet_data.iloc[0]['Source']
        packet_dest = packet_data.iloc[0]['Destination']
        
        # Use vectorized operations for faster matching
        matches = (self.controller_table['Source'] == packet_source) & (self.controller_table['Destination'] == packet_dest)
        if matches.any():
            match_idx = matches.idxmax()
            self.controller_table.loc[match_idx, 'hit_count'] += 1
            self.controller_table.loc[match_idx, 'was_hit_this_iteration'] = 1
            self.controller_table.loc[match_idx, 'total_packet_count'] += 1
    
    def _handle_packet_miss(self, packet_data, value):
        """Handle packet miss - OPTIMIZED"""
        packet_source = packet_data.iloc[0]['Source']
        packet_dest = packet_data.iloc[0]['Destination']
        
        # Use vectorized operations for faster matching
        matches = (self.controller_table['Source'] == packet_source) & (self.controller_table['Destination'] == packet_dest)
        if matches.any():
            match_idx = matches.idxmax()
            self.controller_table.loc[match_idx, 'miss_count'] += 1
            self.controller_table.loc[match_idx, 'total_packet_count'] += 1
            
            # Add to installation queue
            self._add_to_installation_queue(packet_data, value)
    
    def _add_to_installation_queue(self, packet_data, value):
        """Add flow to installation queue"""
        flow_entry = packet_data.copy()
        flow_entry['flow_age'] = self.args.reset_age
        flow_entry['switchcopy'] = float(value.iloc[self.packet_counter].iloc[0]) + self.reactive_time_interval
        flow_entry['is_speculative'] = False  # Reactive flows
        flow_entry = flow_entry[['Source', 'Destination', 'flow_age', 'switchcopy', 'is_speculative']]
        self.flow_queue.append(flow_entry)
    
    def _evict_flow_if_needed(self, flow_entry):
        """Evict a flow if the switch table is full"""
        # before evicting, we need to check if the flow is in the switch already
        # write a loop to check if the flow is in the switch already
        for i in range(len(self.switch_table)):
            if flow_entry.iloc[0]['Source'] == self.switch_table.iloc[i]['Source'] and flow_entry.iloc[0]['Destination'] == self.switch_table.iloc[i]['Destination']:
                print(f"Flow already in switch table: {flow_entry.iloc[0]['Source']} {flow_entry.iloc[0]['Destination']}")
                return False
        
        if len(self.switch_table) >= self.table_size:
            flow_to_evict, _ = self.priority_policy.find_least_frequently_used_flow(
                self.switch_table, self.controller_table, is_speculative=False
            )
            
            if flow_to_evict != -1:
                self.switch_table = self.switch_table.drop(flow_to_evict)
                if self.data_collector:
                    self.data_collector.record_evicted_flows(1)
            else:
                return False
        return True # we have sufficient space to install the flow
    
    def _create_new_flow_dataframe(self, flow_entry):
        """Create a new flow DataFrame from the flow entry"""
        # Convert Series to DataFrame if needed
        if isinstance(flow_entry, pd.Series):
            flow_entry = flow_entry.to_frame().T
        
        # Extract values once to avoid repeated .iloc[0] calls
        source = flow_entry.iloc[0]['Source']
        destination = flow_entry.iloc[0]['Destination']
        switchcopy = flow_entry.iloc[0]['switchcopy']
        
        return pd.DataFrame([{
            'Source': source,
            'Destination': destination,
            'flow_age': self.args.reset_age,
            'switchcopy': switchcopy,
            'is_speculative': False
        }])
    
    def _install_flow_to_switch_table(self, new_flow, flow_entry):
        """Install the new flow to the switch table and record the installation"""
        # Concatenate and remove duplicates in one operation
        self.switch_table = pd.concat([self.switch_table, new_flow], ignore_index=True)
        self.switch_table.drop_duplicates(subset=['Source', 'Destination'], keep='last', inplace=True)
        
        # Record installation with extracted values
        if self.data_collector:
            self.data_collector.record_flow_installation({
                'Source': flow_entry.iloc[0]['Source'],
                'Destination': flow_entry.iloc[0]['Destination'],
                'flow_age': self.args.reset_age
            }, is_speculative=False)
    
    def _create_and_install_flow(self, flow_entry):
        """Create and install a new flow to the switch table"""
        new_flow = self._create_new_flow_dataframe(flow_entry)
        self._install_flow_to_switch_table(new_flow, flow_entry)
    
    def _install_flow_from_queue(self, flow_entry):
        """Install flow from queue to switch table"""
        if self._evict_flow_if_needed(flow_entry):
            self._create_and_install_flow(flow_entry)

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
        
        # Check against simulation time limit
        return current_time > self.args.simulation_time
            
        return False
    
    def _update_performance_metrics(self, value):
        """Update performance tracking metrics"""
        # Reset counters for next interval
        self.trace_counter = self.packet_counter
        
        # Update LFU counter
        if ((float(value.iloc[self.packet_counter].iloc[0]) - float(value.iloc[self.least_frequently_used_counter].iloc[0])) > self.lfu_time_interval):
            self.least_frequently_used_counter = self.packet_counter
            self.controller_table['total_packet_count'] = 0
