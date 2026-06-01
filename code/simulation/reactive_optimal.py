import pandas as pd
import numpy as np
from collections import defaultdict, deque
import math

class ReactiveOptimalSimulation:
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
        self.reactive_time_interval = float(args.RTI)
        
        # Initialize tracking variables
        self.packet_counter = 0
        self.trace_counter = 0
        self.graph_counter = 0
        self.flow_rate_counter = 0
        
        # Time tracking
        self.current_time = 0.0
        self.last_metrics_time = 0.0
        self.metrics_interval = 1.0
        self.lti_start_time = 0.0
        self.lti_end_time = 0.0
        
        # Flow tracking for reactive optimal eviction
        self.flow_next_packet_time = {}  # Maps flow to next packet arrival time
        self.flow_last_packet_time = {}  # Maps flow to last packet arrival time
        self.flow_packet_count = defaultdict(int)  # Count packets per flow
        
        # Precomputed future packet information
        self.flow_future_packets = {}  # Maps flow to list of future packet times
        self.current_packet_index = 0
        
        # Eviction tracking
        self.total_evictions = 0
    
    def _precompute_future_packet_times(self, dataset, value):
        """Precompute when each flow will have its next packet"""
        if self.logger:
            self.logger.info("Precomputing future packet times for reactive optimal eviction...")
        
        # Group packets by flow and time
        flow_packets = defaultdict(list)
        
        for idx, row in dataset.iterrows():
            flow_key = (row['Source'], row['Destination'])
            packet_time = float(value.iloc[idx].iloc[0])
            flow_packets[flow_key].append((idx, packet_time))
        
        # Store future packet times for each flow
        for flow_key, packets in flow_packets.items():
            # Sort packets by time
            packets.sort(key=lambda x: x[1])
            self.flow_future_packets[flow_key] = packets
        
        if self.logger:
            self.logger.info(f"Precomputed future packets for {len(self.flow_future_packets)} flows")
    
    def _get_next_packet_time(self, flow_key, current_time):
        """Get the next packet time for a flow after current_time"""
        if flow_key not in self.flow_future_packets:
            return None
        
        # Find the next packet after current_time within simulation duration
        for idx, packet_time in self.flow_future_packets[flow_key]:
            if packet_time > current_time and packet_time <= self.args.simulation_time:
                return packet_time
        
        return None  # No more packets for this flow
    
    def _update_flow_next_packet_time(self, flow_key, current_time):
        """Update the next packet time for a flow"""
        next_time = self._get_next_packet_time(flow_key, current_time)
        if next_time is not None:
            self.flow_next_packet_time[flow_key] = next_time
        else:
            # No more packets for this flow - remove from tracking
            if flow_key in self.flow_next_packet_time:
                del self.flow_next_packet_time[flow_key]
    
    def _find_flow_to_evict(self):
        """Find the flow that will have its next packet farthest in the future"""
        if not self.flow_next_packet_time:
            return None
        
        # Find flow with the latest next packet time
        latest_flow = max(self.flow_next_packet_time.items(), 
                         key=lambda x: x[1] if x[1] is not None else 0)
        
        return latest_flow[0] if latest_flow[1] is not None else None
    
    def _evict_flow(self, flow_key):
        """Evict a flow from the switch table"""
        if flow_key in self.switch_table.index:
            self.switch_table = self.switch_table.drop(flow_key)
            self.total_evictions += 1
            if self.logger:
                self.logger.info(f"Reactive optimal eviction: Removed flow {flow_key}")
    
    def _add_flow_to_table(self, flow_data):
        """Add a new flow to the switch table"""
        flow_key = (flow_data['Source'], flow_data['Destination'])
        
        new_flow = pd.DataFrame([{
            'Source': flow_data['Source'],
            'Destination': flow_data['Destination'],
            'flow_age': 1.0,
            'is_speculative': False,
        }], index=[flow_key])
        
        self.switch_table = pd.concat([self.switch_table, new_flow], ignore_index=False)
        
        # Update controller table
        if flow_key in self.controller_table.index:
            self.controller_table.loc[flow_key, 'miss_count'] += 1
            self.controller_table.loc[flow_key, 'total_packet_count'] += 1
    
    def _handle_packet_miss(self, packet_data, packet_time):
        """Handle a packet miss with reactive optimal flow management"""
        flow_key = (packet_data['Source'], packet_data['Destination'])
        
        # Check if table is full
        if len(self.switch_table) >= self.table_size:
            # Find the reactive optimal flow to evict
            flow_to_evict = self._find_flow_to_evict()
            if flow_to_evict:
                self._evict_flow(flow_to_evict)
                # Remove from future tracking
                if flow_to_evict in self.flow_next_packet_time:
                    del self.flow_next_packet_time[flow_to_evict]
        
        # Add the new flow
        self._add_flow_to_table(packet_data)
        
        # Update future packet time tracking
        self._update_flow_next_packet_time(flow_key, packet_time)
        self.flow_last_packet_time[flow_key] = packet_time
        self.flow_packet_count[flow_key] += 1
        
        # Record reactive flow installation
        self.data_collector.record_flow_installation(packet_data, is_speculative=False)
    
    def _handle_packet_hit(self, packet_data, packet_time):
        """Handle a packet hit"""
        flow_key = (packet_data['Source'], packet_data['Destination'])
        
        # Update controller table
        if flow_key in self.controller_table.index:
            self.controller_table.loc[flow_key, 'hit_count'] += 1
            self.controller_table.loc[flow_key, 'total_packet_count'] += 1
            self.controller_table.loc[flow_key, 'was_hit_this_iteration'] = 1
        
        # Update future packet time tracking
        self.flow_last_packet_time[flow_key] = packet_time
        self.flow_packet_count[flow_key] += 1
        
        # Update next packet time
        self._update_flow_next_packet_time(flow_key, packet_time)
    
    def run(self, dataset, value):
        """Main simulation loop for reactive optimal mode"""
        if self.logger:
            self.logger.info("Starting reactive optimal SDN simulation with future information...")
        
        # Precompute future packet times
        self._precompute_future_packet_times(dataset, value)
        
        # Process each packet
        for idx, row in dataset.iterrows():
            packet_data = {
                'Source': row['Source'],
                'Destination': row['Destination']
            }
            
            packet_time = float(value.iloc[idx].iloc[0])
            self.current_time = packet_time
            
            # Stop simulation at configured duration
            if packet_time > self.args.simulation_time:
                if self.logger:
                    self.logger.info(f"Stopping simulation at {packet_time:.2f} seconds (simulation_time limit reached)")
                break
            
            # Check if packet hits in switch table
            flow_key = (packet_data['Source'], packet_data['Destination'])
            hit = flow_key in self.switch_table.index
            
            if hit:
                # Packet hit - update flow information
                self._handle_packet_hit(packet_data, packet_time)
                self.data_collector.record_packet_processing(packet_time, True, is_speculative=False, is_reactive_hit=True)
            else:
                # Packet miss - install flow with reactive optimal eviction
                self._handle_packet_miss(packet_data, packet_time)
                self.data_collector.record_packet_processing(packet_time, False, is_speculative=False)
            
            # Collect metrics at regular intervals
            if packet_time - self.last_metrics_time >= self.metrics_interval:
                self.lti_end_time = packet_time
                self._collect_lti_metrics(self.lti_start_time, self.lti_end_time)
                self.lti_start_time = packet_time
                self.last_metrics_time = packet_time
            
            self.packet_counter += 1
        
        if self.logger:
            self.logger.info(f"Reactive optimal simulation completed. Processed {self.packet_counter} packets.")
            self.logger.info(f"Total reactive optimal evictions: {self.total_evictions}")
    
    def _collect_lti_metrics(self, lti_start_time, lti_end_time):
        """Collect Learning Time Interval metrics"""
        # Use the data collector's built-in LTI metrics recording
        self.data_collector.record_lti_metrics(lti_start_time, lti_end_time, self.switch_table)
