



class PriorityPolicy:
    def __init__(self, args):
        self.args = args
        self.aging_factor = args.agingfactor  # Fixed aging factor as specified
        
    def apply_aging_to_switch_table(self, switch_table, controller_table):
        """Apply aging factor to switch table flows - CORRECTED"""
        # Check if switch table is empty
        if len(switch_table) == 0:
            return switch_table, controller_table
        
        # Create a lookup dictionary for faster matching
        controller_lookup = {}
        for idx, row in controller_table.iterrows():
            key = (row['Source'], row['Destination'])
            controller_lookup[key] = idx
        
        # Process switch table entries efficiently
        for k in range(len(switch_table)):
            switch_row = switch_table.iloc[k]
            key = (switch_row['Source'], switch_row['Destination'])
            
            if key in controller_lookup:
                controller_idx = controller_lookup[key]
                if controller_table.loc[controller_idx]['was_hit_this_iteration'] != 1:
                    # Apply aging factor multiplication
                    current_age = switch_table.iloc[k]['flow_age']
                    switch_table.iloc[k, switch_table.columns.get_loc('flow_age')] = current_age * self.aging_factor
                else:
                    # Reset age to reset_age when hit occurs
                    switch_table.iloc[k, switch_table.columns.get_loc('flow_age')] = self.args.reset_age
                    # Reset hit flag
                    controller_table.loc[controller_idx, 'was_hit_this_iteration'] = 0
                    
        return switch_table, controller_table
    
    def find_least_frequently_used_flow(self, switch_table, controller_table, is_speculative=False):
        """Find the least frequently used flow for eviction - CORRECTED"""
        if len(switch_table) == 0:
            return -1, False
            
        temp = float('inf')
        no = -1
        breaknew = False
        
        # Create lookup for faster access
        controller_lookup = {}
        for idx, row in controller_table.iterrows():
            key = (row['Source'], row['Destination'])
            controller_lookup[key] = idx
        
        for k in range(len(switch_table)):
            switch_row = switch_table.iloc[k]
            key = (switch_row['Source'], switch_row['Destination'])
            
            if key in controller_lookup:
                controller_idx = controller_lookup[key]
                controller_row = controller_table.loc[controller_idx]
                flow_age = switch_row['flow_age']
                
                # Different policies based on mode
                if not is_speculative:
                    # Pure reactive mode - use total packet count only
                    packet_count = controller_row['total_packet_count']
                    if temp > packet_count:
                        temp = packet_count
                        no = switch_table.index[k]
                    if temp == 0:
                        temp = packet_count
                        no = switch_table.index[k]
                        breaknew = True
                        break
                else:
                    # Speculative mode - evict flows with age <= 0.4 first, then LFU
                    if flow_age <= 0.4:
                        packet_count = controller_row['total_packet_count']
                        if temp > packet_count:
                            temp = packet_count
                            no = switch_table.index[k]
                        if temp == 0:
                            temp = packet_count
                            no = switch_table.index[k]
                            breaknew = True
                            break
                    else:
                        continue
                    
        return no, breaknew
    
    def evict_flows_with_low_age(self, switch_table, controller_table, required_slots, data_collector=None):
        """Evict flows with age <= 0.4 to make space for new speculative flows"""
        evicted_count = 0
        
        # Create lookup for controller table
        controller_lookup = {}
        for idx, row in controller_table.iterrows():
            key = (row['Source'], row['Destination'])
            controller_lookup[key] = idx
        
        # First pass: evict flows with age < 0.5 (after aging from 0.5)
        flows_to_evict = []
        for k in range(len(switch_table)):
            switch_row = switch_table.iloc[k]
            key = (switch_row['Source'], switch_row['Destination'])
            
            if switch_row['flow_age'] <= self.args.speculative_reset_age and k not in flows_to_evict:
                # Get LFU counter from controller table
                lfu_counter = 0
                if key in controller_lookup:
                    controller_idx = controller_lookup[key]
                    lfu_counter = controller_table.loc[controller_idx]['total_packet_count']
                
                flows_to_evict.append({
                    'index': k,
                    'age': switch_row['flow_age'],
                    'lfu_counter': lfu_counter
                })
        
        # Sort by age first, then by LFU counter (ascending - evict least frequently used first)
        flows_to_evict.sort(key=lambda x: (x['age'], x['lfu_counter']))
        
        # Evict only enough flows to make space
        evicted_indices = []
        for flow_info in flows_to_evict:
            if len(evicted_indices) >= required_slots:
                break
            evicted_indices.append(flow_info['index'])
        
        # Remove evicted flows from switch table (in reverse order to maintain indices)
        evicted_indices.sort(reverse=True)
        for idx in evicted_indices:
            switch_table.drop(switch_table.index[idx], inplace=True)
            evicted_count += 1
        
        # Record evicted flows if data collector is provided
        if data_collector and evicted_count > 0:
            data_collector.record_evicted_flows(evicted_count)
        
        return switch_table, evicted_count
    
    def evict_flows_with_low_age_optimized(self, switch_table, controller_table, required_slots, from_reactive_fallback, data_collector=None):
        # First, check if we can evict any flows using the counting method
        evictable_count = self.count_evictable_flows(switch_table)
        
        # Handle reactive fallback logic
        if evictable_count == 0:
            if from_reactive_fallback:
                # Force eviction based on packet count for reactive fallback
                print(f"Reactive fallback: forcing eviction based on packet count. Required slots: {required_slots}, Current table size: {len(switch_table)}")
                return self._force_evict_by_packet_count(switch_table, controller_table, required_slots, data_collector)
            else:
                # No eviction possible and not forced
                print(f"No eviction possible: evictable_count={evictable_count}, from_reactive_fallback={from_reactive_fallback}")
                return switch_table, 0
        
        # If we have fewer evictable flows than required, adjust required_slots
        actual_slots_to_evict = min(required_slots, evictable_count)
        
        # Vectorized filtering: get flows with age <= speculative_reset_age
        age_mask = switch_table['flow_age'] <= self.args.speculative_reset_age
        
        # Create a DataFrame with only eligible flows for efficient processing
        eligible_flows = switch_table.loc[age_mask].copy()
        
        # Vectorized merge with controller table to get packet counts
        # Create merge keys
        eligible_flows['merge_key'] = eligible_flows['Source'].astype(str) + '_' + eligible_flows['Destination'].astype(str)
        controller_merge = controller_table.copy()
        controller_merge['merge_key'] = controller_merge['Source'].astype(str) + '_' + controller_merge['Destination'].astype(str)
        
        # Merge to get packet counts (left join to keep all eligible flows)
        merged_flows = eligible_flows.merge(
            controller_merge[['merge_key', 'total_packet_count']], 
            on='merge_key', 
            how='left'
        )
        
        # Fill NaN values with 0 for flows not in controller table
        merged_flows['total_packet_count'] = merged_flows['total_packet_count'].fillna(0)
        
        # Sort by age first, then by packet count (ascending - evict least frequently used first)
        merged_flows_sorted = merged_flows.sort_values(['flow_age', 'total_packet_count'])
        
        # Take only the required number of flows
        flows_to_evict = merged_flows_sorted.head(actual_slots_to_evict)
        
        # Get the original indices to remove from switch table using position-based approach
        evicted_positions = flows_to_evict.index.tolist()
        
        # Get the actual row indices from the original switch table that match our criteria
        age_mask_indices = switch_table.index[age_mask].tolist()
        evicted_indices = [age_mask_indices[pos] for pos in evicted_positions]
        
        # Remove evicted flows from switch table
        switch_table = switch_table.drop(evicted_indices)
        evicted_count = len(evicted_indices)
        
        # Record evicted flows if data collector is provided
        if data_collector and evicted_count > 0:
            data_collector.record_evicted_flows(evicted_count)
        
        return switch_table, evicted_count
    
    def _force_evict_by_packet_count(self, switch_table, controller_table, required_slots, data_collector=None):
        """Force eviction based on packet count when reactive fallback requires it"""
        evicted_count = 0
        print(f"Force evicting {required_slots} flows by packet count. Current table size: {len(switch_table)}")
        
        # Create merge keys for all flows in switch table
        switch_merge = switch_table.copy()
        switch_merge['merge_key'] = switch_merge['Source'].astype(str) + '_' + switch_merge['Destination'].astype(str)
        
        controller_merge = controller_table.copy()
        controller_merge['merge_key'] = controller_merge['Source'].astype(str) + '_' + controller_merge['Destination'].astype(str)
        
        # Merge to get packet counts for all flows
        all_flows_with_packets = switch_merge.merge(
            controller_merge[['merge_key', 'total_packet_count']], 
            on='merge_key', 
            how='left'
        )
        
        # Fill NaN values with 0 for flows not in controller table
        all_flows_with_packets['total_packet_count'] = all_flows_with_packets['total_packet_count'].fillna(0)
        
        # Sort by packet count (ascending - evict least frequently used first)
        flows_sorted_by_packets = all_flows_with_packets.sort_values('total_packet_count')
        
        # Take the required number of flows with lowest packet count
        flows_to_evict = flows_sorted_by_packets.head(required_slots)
        
        # Get the actual row indices from the original switch table that match our criteria
        # The flows_to_evict DataFrame has the same index as switch_table, so we can use them directly
        evicted_indices = flows_to_evict.index.tolist()
        
        # Remove evicted flows from switch table
        switch_table = switch_table.drop(evicted_indices)
        evicted_count = len(evicted_indices)
        
        # Record evicted flows if data collector is provided
        if data_collector and evicted_count > 0:
            data_collector.record_evicted_flows(evicted_count)
        
        return switch_table, evicted_count
    
    def count_evictable_flows(self, switch_table):
        """Efficiently count how many flows can be evicted based on age"""
        # Vectorized filtering: get flows with age <= speculative_reset_age
        age_mask = switch_table['flow_age'] <= self.args.speculative_reset_age
        
        if not age_mask.any():
            return 0
        
        # Get eligible flows
        eligible_flows = switch_table.loc[age_mask]
        
        if len(eligible_flows) == 0:
            return 0
        
        # Return total count of eligible flows
        return len(eligible_flows)
    
    def evict_flow_from_switch_table(self, switch_table, controller_table, is_speculative=False):
        """Evict least frequently used flow from switch table"""
        if len(switch_table) >= self.args.tablesize:
            no, breaknew = self.find_least_frequently_used_flow(switch_table, controller_table, is_speculative)
            if no != -1:
                switch_table = switch_table.drop([no])
        return switch_table

    def evict_flows_in_reactive_mode(self, switch_table, tablesize, controller_table, required_slots, data_collector=None):
        """Evict flows based on age and packet count priority"""
        if required_slots > 1: 
            print(f"Evicting {required_slots} flows in reactive mode")
        
        evicted_count = 0
        
        # Create lookup for controller table
        controller_lookup = {}
        for idx, row in controller_table.iterrows():
            key = (row['Source'], row['Destination'])
            controller_lookup[key] = idx
        
        # Collect all flows with their age and packet count
        all_flows_info = []
        for k in range(len(switch_table)):
            switch_row = switch_table.iloc[k]
            key = (switch_row['Source'], switch_row['Destination'])
            
            # Get packet count from controller table
            packet_count = 0
            if key in controller_lookup:
                controller_idx = controller_lookup[key]
                packet_count = controller_table.loc[controller_idx]['total_packet_count']
            
            all_flows_info.append({
                'index': k,
                'age': switch_row['flow_age'],
                'packet_count': packet_count
            })
        
        # Primary eviction: flows with age <= 0.5, sorted by packet count (lowest first)
        primary_eviction_candidates = [f for f in all_flows_info if f['age'] <= self.args.speculative_reset_age]
        primary_eviction_candidates.sort(key=lambda x: x['packet_count'])
        
        flows_to_evict = []
        
        # If we have enough flows with age <= 0.5, use only those
        if len(primary_eviction_candidates) >= required_slots:
            flows_to_evict = primary_eviction_candidates[:required_slots]
        else:
            # Use all flows with age <= 0.5
            flows_to_evict.extend(primary_eviction_candidates)
            
            # Check if table is full to determine if we can evict more flows
            remaining_slots_needed = required_slots - len(flows_to_evict)
            
            if remaining_slots_needed > 0 and len(switch_table) >= tablesize:
                # Table is full, evict additional flows based on packet count only
                remaining_flows = [f for f in all_flows_info if f not in flows_to_evict]
                remaining_flows.sort(key=lambda x: x['packet_count'])  # Sort by packet count (lowest first)
                
                # Add the required number of remaining flows
                flows_to_evict.extend(remaining_flows[:remaining_slots_needed])
        
        # Evict the selected flows
        evicted_indices = [flow_info['index'] for flow_info in flows_to_evict]
        
        # Remove evicted flows from switch table (in reverse order to maintain indices)
        evicted_indices.sort(reverse=True)
        for idx in evicted_indices:
            switch_table.drop(switch_table.index[idx], inplace=True)
            evicted_count += 1
        
        # Record evicted flows if data collector is provided
        if data_collector and evicted_count > 0:
            data_collector.record_evicted_flows(evicted_count)
        
        return switch_table, evicted_count
