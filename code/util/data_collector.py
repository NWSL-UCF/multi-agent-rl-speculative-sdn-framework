import pandas as pd
import os
import json
from datetime import datetime

from util.environment import save_environment_json

class DataCollector:
    def __init__(self, args, output_dir="./results", logger=None):
        self.args = args
        self.output_dir = output_dir
        self.logger = logger
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize data storage
        self.hit_rate_data = []
        self.miss_rate_data = []
        self.traffic_rate_data = []
        self.speculated_flow_data = []
        self.reward_data = []
        self.lti_metrics_data = []  # New LTI-specific metrics
        
        # Performance counters
        self.total_packets = 0
        self.total_hits = 0
        self.total_misses = 0
        self.total_speculative_flows = 0
        self.total_reactive_flows = 0
        self.total_evicted_flows = 0  # Track total evicted flows
        
        # LTI-specific counters
        self.lti_packets = 0
        self.lti_hits = 0
        self.lti_misses = 0
        self.lti_reactive_hits = 0
        self.lti_speculative_hits = 0
        self.lti_evicted_flows = 0  # Track evicted flows per LTI
        self.lti_start_time = 0.0
        self.current_lti = 0
        
        # LTI reward tracking
        self.lti_total_reward = 0.0
        self.previous_lti_total_reward = 0.0
        
        # Time tracking
        self.current_time = 0.0
        self.last_metrics_time = 0.0
        self.metrics_interval = 1.0  # Collect metrics every second
        
        # Wall-clock time tracking
        self.wall_clock_start_time = None
        self.wall_clock_end_time = None
        self.total_wall_clock_time = 0.0
        
        # Flow table tracking
        self.switch_table = pd.DataFrame()
        
    def set_switch_table(self, switch_table):
        """Update the current switch table for LTI metrics"""
        self.switch_table = switch_table.copy()
    
    def set_wall_clock_time(self, total_wall_clock_time):
        """Set the total wall-clock time for the simulation"""
        self.total_wall_clock_time = total_wall_clock_time
    
    def record_packet_processing(self, packet_time, was_hit, is_speculative=False, is_reactive_hit=False):
        """Record packet processing results"""
        self.total_packets += 1
        self.lti_packets += 1
        self.current_time = packet_time
        
        if was_hit:
            self.total_hits += 1
            self.lti_hits += 1
            if is_reactive_hit:
                self.lti_reactive_hits += 1
            elif is_speculative:
                self.lti_speculative_hits += 1
        else:
            self.total_misses += 1
            self.lti_misses += 1
            
        # Collect metrics at regular intervals
        if packet_time - self.last_metrics_time >= self.metrics_interval:
            self._collect_metrics(packet_time, is_speculative)
            self.last_metrics_time = packet_time
    
    def record_flow_installation(self, flow_data, is_speculative=False):
        """Record flow installation"""
        if is_speculative:
            self.total_speculative_flows += 1
            self.speculated_flow_data.append({
                'timestamp': self.current_time,
                'source': flow_data['Source'],
                'destination': flow_data['Destination'],
                'flow_age': flow_data.get('flow_age', 1.0),
                'is_speculative': True
            })
        else:
            self.total_reactive_flows += 1
    
    def record_evicted_flows(self, evicted_count):
        """Record flows evicted due to age < 0.5"""
        self.total_evicted_flows += evicted_count
        self.lti_evicted_flows += evicted_count
    
    def record_lti_reward(self, total_reward):
        """Record total reward for current LTI"""
        self.lti_total_reward = total_reward
    
    def record_reward(self, reward, timestamp):
        """Record DQN reward data"""
        self.reward_data.append({
            'timestamp': timestamp,
            'reward': reward
        })
    
    def record_lti_metrics(self, lti_start_time, lti_end_time, switch_table):
        """Record metrics for a specific Learning Time Interval"""
        self.set_switch_table(switch_table)
        
        # Count flows in switch table
        total_flows = len(switch_table)
        
        # Check if is_speculative column exists, if not assume all flows are reactive
        if 'is_speculative' in switch_table.columns:
            reactive_flows = len(switch_table[switch_table['is_speculative'] == False])
            speculative_flows = len(switch_table[switch_table['is_speculative'] == True])
        else:
            # If is_speculative column doesn't exist, all flows are reactive
            reactive_flows = total_flows
            speculative_flows = 0
        
        # Calculate metrics
        hit_rate = (self.lti_hits / max(1, self.lti_packets)) * 100
        speculation_efficiency = 0.0
        
        if self.lti_speculative_hits > 0 and self.lti_reactive_hits > 0 and speculative_flows > 0 and total_flows > 0:
            # (speculative_hit_count/reactive_hit_count)/(speculative_flows_count/total_flows_count)
            speculation_efficiency = (self.lti_speculative_hits / self.lti_reactive_hits) / (speculative_flows / total_flows)
        
        # Calculate reward metrics
        current_total_reward = self.lti_total_reward
        delta_reward = current_total_reward - self.previous_lti_total_reward
        
        # Record LTI metrics
        lti_metrics = {
            'lti_number': self.current_lti,
            'lti_start_time': lti_start_time,
            'lti_end_time': lti_end_time,
            'lti_duration': lti_end_time - lti_start_time,
            'total_packets': self.lti_packets,
            'total_hits': self.lti_hits,
            'total_misses': self.lti_misses,
            'reactive_hits': self.lti_reactive_hits,
            'speculative_hits': self.lti_speculative_hits,
            'total_flows': total_flows,
            'reactive_flows': reactive_flows,
            'speculative_flows': speculative_flows,
            'hit_rate': hit_rate,
            'speculation_efficiency': speculation_efficiency,
            'total_evicted_flows': self.lti_evicted_flows,
            'reward': current_total_reward,
            'delta_reward': delta_reward
        }
        
        self.lti_metrics_data.append(lti_metrics)
        
        # Reset LTI counters for next interval
        self.lti_packets = 0
        self.lti_hits = 0
        self.lti_misses = 0
        self.lti_reactive_hits = 0
        self.lti_speculative_hits = 0
        self.lti_evicted_flows = 0  # Reset evicted flows counter
        
        # Store current reward for next delta calculation and reset
        self.previous_lti_total_reward = self.lti_total_reward
        self.lti_total_reward = 0.0
        
        self.current_lti += 1
    
    def _collect_metrics(self, timestamp, is_speculative=False):
        """Collect performance metrics at regular intervals"""
        # Calculate hit rate
        hit_rate = (self.total_hits / max(1, self.total_packets)) * 100
        miss_rate = (self.total_misses / max(1, self.total_packets)) * 100
        
        # Calculate traffic rate (packets per second)
        traffic_rate = self.total_packets / max(1, timestamp)
        
        # Record metrics
        self.hit_rate_data.append({
            'timestamp': timestamp,
            'hit_rate': hit_rate,
            'total_packets': self.total_packets,
            'total_hits': self.total_hits,
            'total_misses': self.total_misses
        })
        
        self.miss_rate_data.append({
            'timestamp': timestamp,
            'miss_rate': miss_rate,
            'total_packets': self.total_packets,
            'total_hits': self.total_hits,
            'total_misses': self.total_misses
        })
        
        self.traffic_rate_data.append({
            'timestamp': timestamp,
            'traffic_rate': traffic_rate,
            'total_packets': self.total_packets
        })
    
    def save_results(self):
        """Save all collected data to CSV files"""
        if self.logger:
            self.logger.results_saving(self.output_dir)
        
        # Save hit rate data
        # if self.hit_rate_data:
        #     hit_rate_df = pd.DataFrame(self.hit_rate_data)
        #     hit_rate_df.to_csv(os.path.join(self.output_dir, "hit_rate.csv"), index=False)
        #     if self.logger:
        #         self.logger.file_saved("Hit rate data", len(self.hit_rate_data))
        
        # Save miss rate data
        # if self.miss_rate_data:
        #     miss_rate_df = pd.DataFrame(self.miss_rate_data)
        #     miss_rate_df.to_csv(os.path.join(self.output_dir, "miss_rate.csv"), index=False)
        #     if self.logger:
        #         self.logger.file_saved("Miss rate data", len(self.miss_rate_data))
        
        # Save traffic rate data
        # if self.traffic_rate_data:
        #     traffic_rate_df = pd.DataFrame(self.traffic_rate_data)
        #     traffic_rate_df.to_csv(os.path.join(self.output_dir, "traffic_rate.csv"), index=False)
        #     if self.logger:
        #         self.logger.file_saved("Traffic rate data", len(self.traffic_rate_data))
        
        # Save LTI metrics data
        if self.lti_metrics_data:
            lti_metrics_df = pd.DataFrame(self.lti_metrics_data)
            lti_metrics_df.to_csv(os.path.join(self.output_dir, "lti_metrics.csv"), index=False)
            if self.logger:
                self.logger.file_saved("LTI metrics data", len(self.lti_metrics_data))
        
        # Save speculated flow data (for speculative modes)
        # if self.speculated_flow_data:
        #     speculated_flow_df = pd.DataFrame(self.speculated_flow_data)
        #     speculated_flow_df.to_csv(os.path.join(self.output_dir, "speculatedflowplot.csv"), index=False)
        #     if self.logger:
        #         self.logger.file_saved("Speculated flow data", len(self.speculated_flow_data))
        
        # Save reward data (for DQN modes)
        # if self.reward_data:
        #     reward_df = pd.DataFrame(self.reward_data)
        #     reward_df.to_csv(os.path.join(self.output_dir, "reward_data.csv"), index=False)
        #     if self.logger:
        #         self.logger.file_saved("Reward data", len(self.reward_data))
        
        # Save arguments/configuration
        self._save_arguments()
        
        # Save summary statistics
        self._save_summary()

        # Save installed package versions from requirements.txt
        save_environment_json(
            self.output_dir,
            selected_device=getattr(self.args, "selected_device", None),
            logger=self.logger,
        )
        
        if self.logger:
            self.logger.all_results_saved()
    
    def _save_arguments(self):
        """Save simulation arguments to JSON file"""
        args_dict = vars(self.args)
        args_dict['timestamp'] = datetime.now().isoformat()
        
        # Add network architecture information if available
        if hasattr(self.args, 'network_architecture'):
            args_dict['network_architecture'] = self.args.network_architecture
        elif hasattr(self.args, 'hidden_layers') and self.args.hidden_layers > 0:
            # Fallback: basic hidden layer info
            args_dict['hidden_layer_info'] = {
                'num_hidden_layers': self.args.hidden_layers,
                'layer_size_formula': 'x + ((y - x) / (hidden_layers + 1)) * i',
                'note': 'Actual layer sizes calculated dynamically based on num_states and num_actions'
            }
        
        with open(os.path.join(self.output_dir, "args.json"), 'w') as f:
            json.dump(args_dict, f, indent=2)
        if self.logger:
            self.logger.file_saved("Arguments", 1)

    def _compute_summary_hit_and_speculation_metrics(self):
        """Compute aggregate hit rate and speculation efficiency metrics for summary."""
        num_lti = len(self.lti_metrics_data)
        hitrate = 0.0
        speculation_efficiency = 0.0
        average_speculation_efficiency = 0.0

        if num_lti > 0:
            hitrate = sum(m['hit_rate'] for m in self.lti_metrics_data) / num_lti
            average_speculation_efficiency = (
                sum(m['speculation_efficiency'] for m in self.lti_metrics_data) / num_lti
            )

            speculative_hits = sum(m.get('speculative_hits', 0) for m in self.lti_metrics_data)
            reactive_hits = sum(m.get('reactive_hits', 0) for m in self.lti_metrics_data)

            speculation_rate_sum = 0.0
            for m in self.lti_metrics_data:
                total_flows = m.get('total_flows', 0)
                if total_flows > 0:
                    speculation_rate_sum += m.get('speculative_flows', 0) / total_flows

            avg_speculation_rate = speculation_rate_sum / num_lti
            if speculative_hits > 0 and reactive_hits > 0 and avg_speculation_rate > 0:
                speculation_efficiency = (
                    (speculative_hits / reactive_hits) / avg_speculation_rate
                )

        return {
            'hitrate': hitrate,
            'speculation_efficiency': speculation_efficiency,
            'average_speculation_efficiency': average_speculation_efficiency,
        }
    
    def _save_summary(self):
        """Save summary statistics"""
        aggregate_metrics = self._compute_summary_hit_and_speculation_metrics()

        def seconds_to_hms(seconds):
            """Convert seconds to hour:min:sec format"""
            if seconds <= 0:
                return "00:00:00"

            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)

            return f"{hours:02d}:{minutes:02d}:{secs:02d}"

        total_run_time_hms = seconds_to_hms(self.current_time)
        wall_clock_hms = seconds_to_hms(self.total_wall_clock_time)

        summary = {
            'total_packets': self.total_packets,
            'total_hits': self.total_hits,
            'total_misses': self.total_misses,
            'total_speculative_flows': self.total_speculative_flows,
            'total_reactive_flows': self.total_reactive_flows,
            'overall_hit_rate': (self.total_hits / max(1, self.total_packets)) * 100,
            'hitrate': aggregate_metrics['hitrate'],
            'overall_miss_rate': (self.total_misses / max(1, self.total_packets)) * 100,
            'speculation_efficiency': aggregate_metrics['speculation_efficiency'],
            'average_speculation_efficiency': aggregate_metrics['average_speculation_efficiency'],
            'simulation_duration_seconds': self.current_time,
            'total_run_time': total_run_time_hms,
            'wall_clock_time_seconds': self.total_wall_clock_time,
            'wall_clock_time': wall_clock_hms,
            'total_lti_intervals': self.current_lti,
            'timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(self.output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        if self.logger:
            self.logger.file_saved("Summary", 1)

    def get_final_metrics(self):
        """Get final performance metrics"""
        aggregate_metrics = self._compute_summary_hit_and_speculation_metrics()

        return {
            'total_packets': self.total_packets,
            'total_hits': self.total_hits,
            'total_misses': self.total_misses,
            'hit_rate': (self.total_hits / max(1, self.total_packets)) * 100,
            'hitrate': aggregate_metrics['hitrate'],
            'miss_rate': (self.total_misses / max(1, self.total_packets)) * 100,
            'total_speculative_flows': self.total_speculative_flows,
            'total_reactive_flows': self.total_reactive_flows,
            'speculation_efficiency': aggregate_metrics['speculation_efficiency'],
            'average_speculation_efficiency': aggregate_metrics['average_speculation_efficiency'],
            'simulation_duration': self.current_time
        }
