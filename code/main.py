import argparse
import numpy as np
import torch
import sys
import time

# Import modular components
from util.data_loader import DataLoader
from core.priority_policy import PriorityPolicy
from core.reward_function import RewardFunction
from core.multiagent_dqn import MultiAgentDQN
from simulation.reactive import ReactiveSimulation
from simulation.speculative import SpeculativeSimulation
from simulation.speculativereactive import SpeculativeReactiveSimulation
from simulation.optimal import OptimalSimulation
from util.data_collector import DataCollector
from util.logger import SDNLogger

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Modular Speculative SDN Simulation')
    
    # DQN Parameters
    parser.add_argument('--numberofFlowsPerAgent', type=int, default=10, help='Number of flows per agent')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Starting epsilon value for exploration')
    parser.add_argument('--epsilon_end', type=float, default=0.01, help='Final epsilon value for exploitation')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate per LTI')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor gamma')
    parser.add_argument('--target_replace_iter', type=int, default=5, help='Target network replacement interval')
    parser.add_argument('--memory_capacity', type=int, default=50, help='Memory capacity for experience replay')
    parser.add_argument('--LR', type=float, default=0.75, help='Learning rate')
    parser.add_argument('--hidden_layers', type=int, default=1, help='Number of hidden layers in DQN')
    parser.add_argument('--hidden_layer_size', type=int, default=None, help='Size of hidden layers (None: use current implementation, int: uniform size for all hidden layers)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for experience replay')
    
    # Simulation Parameters
    parser.add_argument('--tablesize', type=int, default=70, help='Switch table size')
    parser.add_argument('--LFUTimeInterval', type=int, default=10, help='LFU time interval')
    parser.add_argument('--agingfactor', type=float, default=0.995, help='Aging factor')
    parser.add_argument('--rewardAgingFactor', type=float, default=0.9, help='Reward aging factor')
    parser.add_argument('--spatialReward', type=float, default=0.9, help='Spatial reward factor')
    
    # SDN Mode
    parser.add_argument('--mode', type=str, default='speculativereactive', 
                       choices=['reactive', 'speculative', 'speculativereactive', 'optimal'],
                       help='SDN mode: reactive, speculative, speculativereactive, or optimal')
    
    # Device Configuration
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1'],
                       help='Device to use: auto (detect automatically), cpu, cuda, cuda:0, cuda:1')
    
    # Data Parameters
    parser.add_argument('--seed', type=int, default=101, help='Random seed')
    parser.add_argument('--ordering', type=str, default='source', 
                       choices=['trace', 'source', 'destination'],
                       help='SDN sorting mode')
    parser.add_argument('--trace', type=int, default=1, choices=[1, 2, 3], help='Trace type')
    parser.add_argument('--LTI', type=float, default=0.1, help='Learning Time Interval')
    parser.add_argument('--RTI', type=float, default=0.01, help='Reactive Time Interval')
    
    # Data Path Configuration
    parser.add_argument('--pcap_base_path', type=str, 
                       default='/home/rouf/data/raw/Pcap',
                       help='Base path for pcap CSV data files')
    
    # Output Configuration
    parser.add_argument('--base_path', type=str, default='./results_speculativereactive',
                       help='Directory to save simulation results')
    
    # Simulation Constants
    parser.add_argument('--reset_age', type=float, default=1.0, help='Reset age for reactive flows')
    parser.add_argument('--speculative_reset_age', type=float, default=0.2, help='Reset age for speculative flows')
    parser.add_argument('--simulation_time', type=float, default=20, help='Simulation duration in seconds')
    
    return parser.parse_args()

def main():
    """Main function to run the simulation"""
    # Start timing the simulation
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize logger
    logger = SDNLogger(args.base_path)
    
    # Device selection logic
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device selection: AUTO")
    elif args.device == 'cpu':
        device = torch.device("cpu")
        logger.info("Device selection: FORCED CPU")
    elif args.device.startswith('cuda'):
        if torch.cuda.is_available():
            device = torch.device(args.device)
            logger.info(f"Device selection: FORCED {args.device.upper()}")
        else:
            device = torch.device("cpu")
            logger.warning(f"CUDA not available, falling back to CPU (requested: {args.device})")
    else:
        device = torch.device("cpu")
        logger.warning(f"Unknown device '{args.device}', falling back to CPU")
    
    # Log device information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    logger.info(f"Using device: {device}")
    
    # Store device in args for passing to MultiAgentDQN (convert to string for JSON serialization)
    args.selected_device = str(device)
    
    
    
    # Log simulation start
    logger.simulation_start(args.mode, args.tablesize, args.trace)
    logger.info(f"Pcap base path: {args.pcap_base_path}")
    logger.info(f"Output directory: {args.base_path}")
    
    try:
        # Load data
        data_loader = DataLoader(args, logger)
        controller_table, switch_table, dataset, value = data_loader.load_data()
        
        logger.data_loading("data file", len(controller_table), dataset.shape)
        
        # Initialize components
        priority_policy = PriorityPolicy(args)
        reward_function = RewardFunction(args)
        data_collector = DataCollector(args, args.base_path, logger)
        
        # Run simulation based on mode
        if args.mode == 'reactive':
            simulation = ReactiveSimulation(args, 
                                            controller_table, 
                                            switch_table, 
                                            priority_policy, 
                                            reward_function, 
                                            data_collector, 
                                            logger)
            simulation.run(dataset, value)
            
        elif args.mode == 'speculative':
            # Initialize DQN only for speculative modes
            multiagent_dqn = MultiAgentDQN(args, controller_table)
            
            # Add network architecture info to args for logging
            network_info = multiagent_dqn.get_network_info()
            if network_info:
                args.network_architecture = network_info
            
            simulation = SpeculativeSimulation(args, 
                                               controller_table, 
                                               switch_table, 
                                               priority_policy, 
                                               reward_function, 
                                               multiagent_dqn, 
                                               data_collector, 
                                               logger)
            simulation.run(dataset, value)
            
        elif args.mode == 'speculativereactive':
            # Initialize DQN only for speculative modes
            multiagent_dqn = MultiAgentDQN(args, controller_table)
            
            # Add network architecture info to args for logging
            network_info = multiagent_dqn.get_network_info()
            if network_info:
                args.network_architecture = network_info
            
            simulation = SpeculativeReactiveSimulation(args, 
                                                       controller_table, 
                                                       switch_table, 
                                                       priority_policy, 
                                                       reward_function, 
                                                       multiagent_dqn, 
                                                       data_collector, 
                                                       logger)
            simulation.run(dataset, value)
            
        elif args.mode == 'optimal':
            # Optimal mode uses future information for perfect eviction decisions
            simulation = OptimalSimulation(args, 
                                          controller_table, 
                                          switch_table, 
                                          priority_policy, 
                                          reward_function, 
                                          data_collector, 
                                          logger)
            simulation.run(dataset, value)
        
        # Calculate total wall-clock time
        end_time = time.time()
        total_wall_clock_time = end_time - start_time
        
        # Set wall-clock time in data collector
        data_collector.set_wall_clock_time(total_wall_clock_time)
        
        # Save results
        data_collector.save_results()
        
        # Log final metrics
        final_metrics = data_collector.get_final_metrics()
        logger.simulation_end(final_metrics)
        
        # Log total execution time
        logger.info(f"Total execution time: {total_wall_clock_time:.2f} seconds")
        
    except FileNotFoundError as e:
        logger.error_with_troubleshooting(str(e))
        sys.exit(1)
    except Exception as e:
        logger.unexpected_error(str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

# python main.py --mode speculativereactive --trace 1 --seed 101 --reset_age 1.0 --speculative_reset_age 0.3 --simulation_time 600 --device auto