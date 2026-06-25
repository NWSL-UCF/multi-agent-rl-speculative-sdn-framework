import argparse
import numpy as np
import torch
import sys
import time
from pathlib import Path

ENABLE_JD = False

if ENABLE_JD:
    from jd import jd_job_dir, jd_upload

# Import modular components
from util.data_loader import DataLoader
from core.priority_policy import PriorityPolicy
from core.reward_function import RewardFunction
from core.multiagent_dqn import MultiAgentDQN
from core.ppo_agent import PPOLearner
from core.combinatorial_bandit import CombinatorialBanditLearner
from simulation.reactive import ReactiveSimulation
from simulation.speculative import SpeculativeSimulation
from simulation.speculativereactive import SpeculativeReactiveSimulation
from simulation.reactive_optimal import ReactiveOptimalSimulation
from simulation.speculative_reactive_optimal import SpeculativeReactiveOptimalSimulation
from util.data_collector import DataCollector
from util.logger import SDNLogger


def build_learner(args, controller_table, logger=None):
    """Construct the speculative-flow learner selected by ``--algorithm``.

    All learners share the same method interface expected by the speculative
    simulations, so they are interchangeable. Defaults to the original
    multi-agent DQN.
    """
    if args.algorithm == 'ppo':
        return PPOLearner(args, controller_table)
    elif args.algorithm == 'bandit':
        return CombinatorialBanditLearner(args, controller_table)
    else:
        return MultiAgentDQN(args, controller_table, logger)


def upload_job_outputs(job_dir, logger):
    """Upload all generated files to the jd server after the simulation."""
    if not ENABLE_JD:
        return
    for path in sorted(job_dir.iterdir()):
        if path.is_file():
            jd_upload(path)
            logger.info(f"Uploaded to job server: {path.name}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Modular Speculative SDN Simulation')
    
    # Shared Learner / Agent Parameters (used by multiple algorithms)
    parser.add_argument('--numberofFlowsPerAgent', type=int, default=10, help='Number of flows per agent')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor gamma (DQN and PPO)')
    parser.add_argument('--hidden_layers', type=int, default=2, help='Number of hidden layers (DQN and PPO)')
    parser.add_argument('--hidden_layer_size', type=int, default=None, help='Size of hidden layers (None: use current implementation, int: uniform size for all hidden layers)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (DQN experience replay / PPO rollout length)')
    
    # DQN Parameters (used when --algorithm dqn)
    parser.add_argument('--dqn_epsilon_start', type=float, default=1.0, help='Starting epsilon value for exploration')
    parser.add_argument('--dqn_epsilon_end', type=float, default=0.01, help='Final epsilon value for exploitation')
    parser.add_argument('--dqn_epsilon_decay', type=float, default=0.995, help='Epsilon decay rate per LTI')
    parser.add_argument('--dqn_target_replace_iter', type=int, default=100, help='Target network replacement interval')
    parser.add_argument('--dqn_memory_capacity', type=int, default=500, help='Memory capacity for experience replay')
    parser.add_argument('--dqn_learning_start_size', type=int, default=100,
                       help='Minimum replay-buffer transitions before DQN training starts (must be >= batch_size)')
    parser.add_argument('--dqn_lr', type=float, default=0.5, help='DQN learning rate')
    
    # Learning Algorithm Selection (applies to speculative / speculativereactive modes)
    parser.add_argument('--algorithm', type=str, default='ppo',
                       choices=['dqn', 'ppo', 'bandit'],
                       help='Speculative-flow learner: dqn (default), ppo (actor-critic), or bandit (combinatorial UCB)')
    
    # PPO Parameters (used when --algorithm ppo)
    parser.add_argument('--ppo_lr', type=float, default=3e-4, help='PPO learning rate')
    parser.add_argument('--ppo_clip', type=float, default=0.2, help='PPO clipped-surrogate ratio epsilon')
    parser.add_argument('--ppo_epochs', type=int, default=4, help='PPO update epochs per rollout')
    parser.add_argument('--ppo_entropy_coef', type=float, default=0.01, help='PPO entropy bonus coefficient')
    parser.add_argument('--ppo_value_coef', type=float, default=0.5, help='PPO value loss coefficient')
    parser.add_argument('--ppo_gae_lambda', type=float, default=0.95, help='PPO GAE lambda')
    
    # Combinatorial Bandit Parameters (used when --algorithm bandit)
    parser.add_argument('--bandit_c', type=float, default=1.0, help='Combinatorial bandit UCB exploration constant')
    
    # Simulation Parameters
    parser.add_argument('--tablesize', type=int, default=70, help='Switch table size')
    parser.add_argument('--LFUTimeInterval', type=int, default=10, help='LFU time interval')
    parser.add_argument('--agingfactor', type=float, default=0.995, help='Aging factor')
    parser.add_argument('--rewardAgingFactor', type=float, default=0.95, help='Reward aging factor')
    parser.add_argument('--spatialReward', type=float, default=0.75, help='Spatial reward factor')
    
    # SDN Mode
    parser.add_argument('--mode', type=str, default='speculativereactive', 
                       choices=['reactive', 'speculative', 'speculativereactive', 'reactiveoptimal', 'speculativereactiveoptimal'],
                       help='SDN mode: reactive, speculative, speculativereactive, reactiveoptimal, or speculativereactiveoptimal')
    
    # Device Configuration
    parser.add_argument('--device', type=str, default='cpu', 
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
                       default='/home/ab823254/data/multi-agent-rl-speculative-sdn-framework/Pcap',
                       help='Base path for pcap CSV data files')
    
    # Output Configuration
    parser.add_argument('--base_path', type=str, default='/home/ab823254/data/multi-agent-rl-speculative-sdn-framework/results/debug',
                       help='Directory to save simulation results')
    
    # Simulation Constants
    parser.add_argument('--reset_age', type=float, default=1.0, help='Reset age for reactive flows')
    parser.add_argument('--speculative_reset_age', type=float, default=0.5, help='Reset age for speculative flows')
    parser.add_argument('--simulation_time', type=float, default=200.0, help='Simulation duration in seconds')
    
    return parser.parse_args()

def main():
    """Main function to run the simulation"""
    # Start timing the simulation
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()

    if ENABLE_JD:
        output_dir = jd_job_dir()
    else:
        output_dir = Path(args.base_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.base_path = str(output_dir)
    
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
            # Initialize the selected learner only for speculative modes
            learner = build_learner(args, controller_table, logger)
            
            # Add network architecture info to args for logging
            network_info = learner.get_network_info()
            if network_info:
                args.network_architecture = network_info
            
            simulation = SpeculativeSimulation(args, 
                                               controller_table, 
                                               switch_table, 
                                               priority_policy, 
                                               reward_function, 
                                               learner, 
                                               data_collector, 
                                               logger)
            simulation.run(dataset, value)
            
        elif args.mode == 'speculativereactive':
            # Initialize the selected learner only for speculative modes
            learner = build_learner(args, controller_table, logger)
            
            # Add network architecture info to args for logging
            network_info = learner.get_network_info()
            if network_info:
                args.network_architecture = network_info
            
            simulation = SpeculativeReactiveSimulation(args, 
                                                       controller_table, 
                                                       switch_table, 
                                                       priority_policy, 
                                                       reward_function, 
                                                       learner, 
                                                       data_collector, 
                                                       logger)
            simulation.run(dataset, value)
            
        elif args.mode == 'reactiveoptimal':
            # Reactive optimal mode uses future information for perfect eviction decisions
            simulation = ReactiveOptimalSimulation(args, 
                                          controller_table, 
                                          switch_table, 
                                          priority_policy, 
                                          reward_function, 
                                          data_collector, 
                                          logger)
            simulation.run(dataset, value)

        elif args.mode == 'speculativereactiveoptimal':
            simulation = SpeculativeReactiveOptimalSimulation(args,
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

        # if ENABLE_JD:
        #     upload_job_outputs(output_dir, logger)
        
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
