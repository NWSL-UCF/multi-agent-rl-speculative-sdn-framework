import pandas as pd
from collections import defaultdict

from simulation.reactive import ReactiveSimulation


class ReactiveOptimalSimulation(ReactiveSimulation):
    """Reactive simulation with oracle eviction.

    Flow insertion matches :class:`ReactiveSimulation` exactly (RTI-delayed queue).
    On eviction, remove the installed flow whose next matching packet is farthest
    in the future.
    """

    def __init__(
        self,
        args,
        controller_table,
        switch_table,
        priority_policy,
        reward_function,
        data_collector,
        logger=None,
    ):
        super().__init__(
            args,
            controller_table,
            switch_table,
            priority_policy,
            reward_function,
            data_collector,
            logger,
        )
        self.flow_future_packets: dict[tuple, list[tuple[int, float]]] = {}
        self.flow_next_packet_time: dict[tuple, float] = {}
        self.total_evictions = 0
        self._current_packet_time = 0.0

    def run(self, dataset, value):
        self._precompute_future_packet_times(dataset, value)
        super().run(dataset, value)
        if self.logger:
            self.logger.info(f"Total reactive optimal evictions: {self.total_evictions}")

    def _process_single_packet(self, dataset, value):
        self._current_packet_time = float(value.iloc[self.packet_counter].iloc[0])
        super()._process_single_packet(dataset, value)

    def _process_flow_queue(self, value):
        self._current_packet_time = float(value.iloc[self.packet_counter].iloc[0])
        super()._process_flow_queue(value)

    def _precompute_future_packet_times(self, dataset, value):
        if self.logger:
            self.logger.info("Precomputing future packet times for reactive optimal eviction...")

        flow_packets: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
        for idx, row in dataset.iterrows():
            flow_key = (row["Source"], row["Destination"])
            packet_time = float(value.iloc[idx].iloc[0])
            flow_packets[flow_key].append((idx, packet_time))

        self.flow_future_packets = {
            flow_key: sorted(packets, key=lambda item: item[1])
            for flow_key, packets in flow_packets.items()
        }

        if self.logger:
            self.logger.info(
                f"Precomputed future packets for {len(self.flow_future_packets)} flows"
            )

    def _get_next_packet_time(self, flow_key, current_time):
        if flow_key not in self.flow_future_packets:
            return None

        for _, packet_time in self.flow_future_packets[flow_key]:
            if packet_time > current_time:
                return packet_time
        return None

    def _update_flow_next_packet_time(self, flow_key, current_time):
        next_time = self._get_next_packet_time(flow_key, current_time)
        if next_time is not None:
            self.flow_next_packet_time[flow_key] = next_time
        elif flow_key in self.flow_next_packet_time:
            del self.flow_next_packet_time[flow_key]

    def _flow_in_switch_table(self, flow_key):
        source, destination = flow_key
        return (
            (self.switch_table["Source"] == source)
            & (self.switch_table["Destination"] == destination)
        ).any()

    def _next_packet_rank(self, flow_key, current_time):
        next_time = self._get_next_packet_time(flow_key, current_time)
        return next_time if next_time is not None else float("inf")

    def _find_flow_to_evict(self, current_time=None):
        current_time = self._current_packet_time if current_time is None else current_time
        best_flow = None
        best_next_time = None

        for _, row in self.switch_table.iterrows():
            flow_key = (row["Source"], row["Destination"])
            candidate = self._next_packet_rank(flow_key, current_time)
            if best_next_time is None or candidate > best_next_time:
                best_next_time = candidate
                best_flow = flow_key

        return best_flow

    def _find_oracle_eviction_index(self, current_time):
        best_idx = None
        best_next_time = None

        for idx, row in self.switch_table.iterrows():
            flow_key = (row["Source"], row["Destination"])
            candidate = self._next_packet_rank(flow_key, current_time)
            if best_next_time is None or candidate > best_next_time:
                best_next_time = candidate
                best_idx = idx

        return best_idx

    def _evict_flow(self, flow_key):
        source, destination = flow_key
        mask = (
            (self.switch_table["Source"] == source)
            & (self.switch_table["Destination"] == destination)
        )
        if not mask.any():
            return False

        self.switch_table = self.switch_table[~mask]
        self.total_evictions += 1
        if flow_key in self.flow_next_packet_time:
            del self.flow_next_packet_time[flow_key]
        if self.data_collector:
            self.data_collector.record_evicted_flows(1)
        if self.logger:
            self.logger.info(f"Reactive optimal eviction: Removed flow {flow_key}")
        return True

    def _evict_flow_if_needed(self, flow_entry):
        for i in range(len(self.switch_table)):
            if (
                flow_entry.iloc[0]["Source"] == self.switch_table.iloc[i]["Source"]
                and flow_entry.iloc[0]["Destination"] == self.switch_table.iloc[i]["Destination"]
            ):
                print(
                    f"Flow already in switch table: {flow_entry.iloc[0]['Source']} "
                    f"{flow_entry.iloc[0]['Destination']}"
                )
                return False

        if len(self.switch_table) >= self.table_size:
            flow_to_evict = self._find_oracle_eviction_index(self._current_packet_time)
            if flow_to_evict is not None:
                self.switch_table = self.switch_table.drop(flow_to_evict)
                self.total_evictions += 1
                if self.data_collector:
                    self.data_collector.record_evicted_flows(1)
            else:
                return False
        return True
