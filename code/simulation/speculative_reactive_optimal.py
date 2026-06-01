import pandas as pd

from simulation.reactive_optimal import ReactiveOptimalSimulation


class SpeculativeReactiveOptimalSimulation(ReactiveOptimalSimulation):
    def __init__(self, args, controller_table, switch_table, priority_policy, reward_function, data_collector, logger=None):
        super().__init__(args, controller_table, switch_table, priority_policy, reward_function, data_collector, logger)
        self.trace_counter = 0
        self.total_proactive_removals = 0
        self.total_speculative_installs = 0
        self.lti_maintenance_counter = 0

    def _add_flow_to_table(self, flow_data, is_speculative=False):
        """Add a new flow to the switch table"""
        flow_key = (flow_data['Source'], flow_data['Destination'])

        new_flow = pd.DataFrame([{
            'Source': flow_data['Source'],
            'Destination': flow_data['Destination'],
            'flow_age': self.args.speculative_reset_age if is_speculative else 1.0,
            'is_speculative': is_speculative,
        }])

        self.switch_table = pd.concat([self.switch_table, new_flow], ignore_index=True)
        self.switch_table = self.switch_table.drop_duplicates(
            subset=['Source', 'Destination'], keep='last'
        )

        if not is_speculative and flow_key in self.controller_table.index:
            self.controller_table.loc[flow_key, 'miss_count'] += 1
            self.controller_table.loc[flow_key, 'total_packet_count'] += 1

    def _check_packet_match(self, flow_key):
        """Check if packet matches a flow in the switch table"""
        source, destination = flow_key
        matches = self.switch_table[
            (self.switch_table['Source'] == source) &
            (self.switch_table['Destination'] == destination)
        ]
        if matches.empty:
            return False, False

        is_speculative = bool(matches.iloc[-1]['is_speculative'])
        return True, not is_speculative

    def _handle_packet_miss(self, packet_data, packet_time):
        """Handle a packet miss with reactive optimal flow management"""
        flow_key = (packet_data['Source'], packet_data['Destination'])

        if len(self.switch_table) >= self.table_size:
            flow_to_evict = self._find_flow_to_evict()
            if flow_to_evict:
                self._evict_flow(flow_to_evict)
                if flow_to_evict in self.flow_next_packet_time:
                    del self.flow_next_packet_time[flow_to_evict]

        self._add_flow_to_table(packet_data, is_speculative=False)

        self._update_flow_next_packet_time(flow_key, packet_time)
        self.flow_last_packet_time[flow_key] = packet_time
        self.flow_packet_count[flow_key] += 1

        self.data_collector.record_flow_installation(packet_data, is_speculative=False)

    def _process_single_packet(self, dataset, value):
        """Process a single packet with reactive optimal and speculative hit tracking"""
        row = dataset.iloc[self.packet_counter]
        packet_data = {
            'Source': row['Source'],
            'Destination': row['Destination'],
        }
        packet_time = float(value.iloc[self.packet_counter].iloc[0])
        self.current_time = packet_time

        flow_key = (packet_data['Source'], packet_data['Destination'])
        hit, is_reactive_hit = self._check_packet_match(flow_key)

        if hit:
            self._handle_packet_hit(packet_data, packet_time)
            self.data_collector.record_packet_processing(
                packet_time, True, is_speculative=True, is_reactive_hit=is_reactive_hit
            )
        else:
            self._handle_packet_miss(packet_data, packet_time)
            self.data_collector.record_packet_processing(packet_time, False, is_speculative=True)

        self.packet_counter += 1

        if packet_time - self.last_metrics_time >= self.metrics_interval:
            self._collect_lti_metrics(self.lti_start_time, packet_time)
            self.lti_start_time = packet_time
            self.last_metrics_time = packet_time

    def _remove_flows_without_future_packets(self, current_time):
        """Remove SFT entries for flows with no future packets within simulation_time"""
        flows_to_remove = []
        for _, row in self.switch_table.iterrows():
            flow_key = (row['Source'], row['Destination'])
            if self._get_next_packet_time(flow_key, current_time) is None:
                flows_to_remove.append(flow_key)

        for flow_key in flows_to_remove:
            self._evict_flow(flow_key)
            if flow_key in self.flow_next_packet_time:
                del self.flow_next_packet_time[flow_key]

        self.total_proactive_removals += len(flows_to_remove)
        return len(flows_to_remove)

    def _fill_empty_slots_with_speculative_flows(self, current_time):
        """Fill empty SFT slots with flows whose next packet is soonest and not yet installed"""
        available_slots = self.table_size - len(self.switch_table)
        if available_slots <= 0:
            return 0

        candidates = []
        for flow_key in self.flow_future_packets:
            if self._flow_in_switch_table(flow_key):
                continue
            next_time = self._get_next_packet_time(flow_key, current_time)
            if next_time is not None:
                candidates.append((flow_key, next_time))

        candidates.sort(key=lambda item: item[1])

        installed = 0
        for flow_key, _ in candidates[:available_slots]:
            flow_data = {
                'Source': flow_key[0],
                'Destination': flow_key[1],
            }
            self._add_flow_to_table(flow_data, is_speculative=True)
            self._update_flow_next_packet_time(flow_key, current_time)
            self.data_collector.record_flow_installation(flow_data, is_speculative=True)
            installed += 1

        self.total_speculative_installs += installed
        return installed

    def _perform_lti_maintenance(self, current_time):
        """Run oracle speculative maintenance at each LTI boundary"""
        removed = self._remove_flows_without_future_packets(current_time)
        installed = self._fill_empty_slots_with_speculative_flows(current_time)

        if self.logger:
            speculative_flows = sum(
                1 for _, flow in self.switch_table.iterrows()
                if flow.get('is_speculative', False)
            )
            reactive_flows = len(self.switch_table) - speculative_flows
            self.logger.info(
                f"LTI maintenance at {current_time:.4f}s: removed {removed} finished flows, "
                f"installed {installed} speculative flows "
                f"(reactive: {reactive_flows}, speculative: {speculative_flows})"
            )

    def _should_break_for_learning(self, value):
        """Check if we should break for the next LTI boundary"""
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
        return current_time > self.args.simulation_time

    def _last_processed_time(self, value):
        """Return timestamp of the most recently processed packet"""
        if self.packet_counter == 0:
            return float(value.iloc[0].iloc[0])
        return float(value.iloc[self.packet_counter - 1].iloc[0])

    def run(self, dataset, value):
        """Main simulation loop for speculative reactive optimal mode"""
        if self.logger:
            self.logger.info(
                "Starting speculative reactive optimal SDN simulation with future information..."
            )

        self._precompute_future_packet_times(dataset, value)

        self.lti_start_time = 0.0
        self.last_metrics_time = 0.0
        lti_maintenance_start_packet = 0
        self.trace_counter = 0

        while True:
            while True:
                if self.packet_counter >= len(value):
                    if self.logger:
                        self.logger.info(
                            f"Speculative reactive optimal simulation completed. "
                            f"Processed {self.packet_counter} packets."
                        )
                        self.logger.info(f"Total reactive optimal evictions: {self.total_evictions}")
                        self.logger.info(f"Total proactive removals: {self.total_proactive_removals}")
                        self.logger.info(f"Total speculative installs: {self.total_speculative_installs}")
                    return

                self._process_single_packet(dataset, value)

                if self._should_break_for_learning(value):
                    break

                if self._should_stop_simulation(value):
                    if self.logger:
                        self.logger.info(
                            f"Speculative reactive optimal simulation completed. "
                            f"Processed {self.packet_counter} packets."
                        )
                        self.logger.info(f"Total reactive optimal evictions: {self.total_evictions}")
                        self.logger.info(f"Total proactive removals: {self.total_proactive_removals}")
                        self.logger.info(f"Total speculative installs: {self.total_speculative_installs}")
                    return

            current_time = self._last_processed_time(value)
            self._perform_lti_maintenance(current_time)

            self.lti_maintenance_counter += 1
            if self.logger:
                speculative_flows = sum(
                    1 for _, flow in self.switch_table.iterrows()
                    if flow.get('is_speculative', False)
                )
                reactive_flows = len(self.switch_table) - speculative_flows
                self.logger.lti_info(
                    self.lti_maintenance_counter,
                    f"Completed LTI. Packets: {self.packet_counter - lti_maintenance_start_packet}, "
                    f"Switch table size: {len(self.switch_table)}, "
                    f"Speculative flows: {speculative_flows}, "
                    f"Reactive flows: {reactive_flows}, "
                    f"Evicted flows: {self.data_collector.lti_evicted_flows}",
                )

            lti_maintenance_start_packet = self.packet_counter
            self.trace_counter = self.packet_counter
