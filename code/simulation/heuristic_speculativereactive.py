from collections import deque, defaultdict

import pandas as pd

from simulation.speculativereactive import SpeculativeReactiveSimulation


class HeuristicSpeculativeReactiveSimulation(SpeculativeReactiveSimulation):
    """Speculative-reactive simulation whose speculative installs are driven by
    a heuristic over a sliding window of recent LTI hit history instead of RL.

    The reactive path (RTI-delayed install queue, eviction) and the aging logic
    (age reset on hit, eviction when ``flow_age <= speculative_reset_age``) are
    inherited unchanged from :class:`SpeculativeReactiveSimulation`. Only the
    per-LTI speculative selection differs: each LTI the flows that generated the
    most hits over the last ``speculative_window_size`` intervals are used to
    fill unoccupied switch-table slots.
    """

    def __init__(self, args, controller_table, switch_table, priority_policy,
                 reward_function, heuristic_learner, data_collector, logger=None):
        # Bypass the RL learner / torch-state setup in the parent __init__.
        self.args = args
        self.controller_table = controller_table
        self.old_controller_table = controller_table.copy()
        self.switch_table = switch_table
        self.priority_policy = priority_policy
        self.reward_function = reward_function
        self.heuristic_learner = heuristic_learner
        self.data_collector = data_collector
        self.logger = logger

        self.table_size = args.tablesize
        self.learning_time_interval = float(args.LTI)
        self.lfu_time_interval = float(args.LFUTimeInterval) * float(args.RTI)
        self.reactive_time_interval = float(args.RTI)
        self.num_flows_per_agent = args.numberofFlowsPerAgent

        self.packet_counter = 0
        self.trace_counter = 0
        self.least_frequently_used_counter = 0
        self.graph_counter = 0
        self.flow_rate_counter = 0

        self.flow_queue = deque()
        self.new_flow_list = []

        # Per-LTI hit accumulator: {(Source, Destination): hits_this_lti}
        self._lti_hits = defaultdict(int)

    def _handle_packet_hit(self, packet_data):
        """Record the hit and accumulate it for the current LTI window entry."""
        super()._handle_packet_hit(packet_data)
        key = (packet_data.iloc[0]['Source'], packet_data.iloc[0]['Destination'])
        self._lti_hits[key] += 1

    def _perform_speculative_learning(self, dataset, value):
        """Replace RL selection with the sliding-window heuristic selection."""
        # Commit the just-finished LTI's hits to the window, then reset.
        self.heuristic_learner.observe_lti(self._lti_hits)
        self._lti_hits = defaultdict(int)

        ranked_flows = self.heuristic_learner.rank_flows(self.controller_table)

        available_slots = self.table_size - len(self.switch_table)
        evictable_flows = self.priority_policy.count_evictable_flows(self.switch_table)

        if evictable_flows > 0 or available_slots > 0:
            self._install_speculative_flows(ranked_flows)

    def _install_speculative_flows(self, ranked_flows):
        """Fill unoccupied slots with heuristic-ranked flows (best-first)."""
        # Evict low-age speculative flows first (same mechanic as the RL path).
        evictable_flows = self.priority_policy.count_evictable_flows(self.switch_table)
        if evictable_flows > 0:
            self.switch_table, _ = self.priority_policy.evict_flows_with_low_age_optimized(
                self.switch_table, self.controller_table, evictable_flows, False, self.data_collector
            )

        remaining_space = self.table_size - len(self.switch_table)
        if remaining_space <= 0:
            return

        # Only fill empty spots: skip flows already installed.
        existing = set(zip(self.switch_table['Source'], self.switch_table['Destination']))

        installed = 0
        for _, flow in ranked_flows.iterrows():
            if installed >= remaining_space:
                break
            key = (flow['Source'], flow['Destination'])
            if key in existing:
                continue

            new_flow = {
                'Source': flow['Source'],
                'Destination': flow['Destination'],
                'flow_age': self.args.speculative_reset_age,
                'is_speculative': True,
                'hit_count': 0,
            }
            self.switch_table = pd.concat(
                [self.switch_table, pd.DataFrame([new_flow])], ignore_index=True
            )
            self.data_collector.record_flow_installation(new_flow, is_speculative=True)
            existing.add(key)
            installed += 1
