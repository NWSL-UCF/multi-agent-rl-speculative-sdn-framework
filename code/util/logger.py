import logging
import os

class SDNLogger:
    def __init__(self, output_dir="./results", log_level=logging.INFO):
        """
        Initialize SDN Logger
        
        Args:
            output_dir: Directory to save log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.output_dir = output_dir
        self.log_level = log_level
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('SDN_Simulation')
        self.logger.setLevel(log_level)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        
        # Console handler (simple format)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        
        # File handler (detailed format)
        log_file = os.path.join(output_dir, "info.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Log initialization
        self.logger.info(f"SDN Logger initialized. Log file: {log_file}")
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)
    
    def lti_info(self, lti_number, message):
        """Log LTI-specific info message"""
        self.logger.info(f"[LTI {lti_number}] {message}")
    
    def lti_debug(self, lti_number, message):
        """Log LTI-specific debug message"""
        self.logger.debug(f"[LTI {lti_number}] {message}")
    
    def simulation_start(self, mode, tablesize, trace):
        """Log simulation start"""
        self.logger.info(f"Starting {mode} SDN simulation...")
        self.logger.info(f"Configuration: Table Size={tablesize}, Trace={trace}")
    
    def simulation_end(self, final_metrics):
        """Log simulation end with final metrics"""
        self.logger.info("="*50)
        self.logger.info("FINAL SIMULATION RESULTS")
        self.logger.info("="*50)
        self.logger.info(f"Total Packets: {final_metrics['total_packets']:,}")
        self.logger.info(f"Total Hits: {final_metrics['total_hits']:,}")
        self.logger.info(f"Total Misses: {final_metrics['total_misses']:,}")
        self.logger.info(f"Overall Hit Rate: {final_metrics['overall_hit_rate']:.2f}%")
        self.logger.info(f"Average Hit Rate (per LTI): {final_metrics['average_hitrate_per_lti']:.2f}%")
        self.logger.info(f"Overall Miss Rate: {final_metrics['overall_miss_rate']:.2f}%")
        self.logger.info(f"Simulation Duration: {final_metrics['simulation_duration']:.2f} seconds")
        
        if 'total_speculative_flows' in final_metrics and final_metrics['total_speculative_flows'] > 0:
            self.logger.info(f"Speculative Flows: {final_metrics['total_speculative_flows']:,}")
            self.logger.info(f"Overall Speculation Efficiency: {final_metrics['overall_speculation_efficiency']:.2f}")
            self.logger.info(f"Average Speculation Efficiency (per LTI): {final_metrics['average_speculation_efficiency_per_lti']:.2f}")
        
        if 'total_reactive_flows' in final_metrics and final_metrics['total_reactive_flows'] > 0:
            self.logger.info(f"Reactive Flows: {final_metrics['total_reactive_flows']:,}")
        
        self.logger.info("="*50)
        self.logger.info("Simulation completed successfully!")
    
    def data_loading(self, file_path, flow_count, dataset_shape):
        """Log data loading information"""
        self.logger.info(f"Loading dataset from: {file_path}")
        self.logger.info(f"Loaded dataset with {flow_count} unique flows")
        self.logger.info(f"Dataset shape: {dataset_shape}")
    
    def results_saving(self, output_dir):
        """Log results saving information"""
        self.logger.info(f"Saving results to: {output_dir}")
    
    def file_saved(self, filename, record_count):
        """Log when a file is saved"""
        self.logger.info(f"✓ {filename} saved: {record_count} records")
    
    def all_results_saved(self):
        """Log when all results are saved"""
        self.logger.info("All results saved successfully!")
    
    def error_with_troubleshooting(self, error_message):
        """Log error with troubleshooting tips"""
        self.logger.error(f"Error: {error_message}")
        self.logger.error("\nTroubleshooting tips:")
        self.logger.error("1. Check if the CSV files exist in the expected locations")
        self.logger.error("2. Use --pcap_base_path to specify the correct path to your data files")
        self.logger.error("3. Ensure the file names match: pcap.csv, pcap_.csv, pcapfile_.csv, etc.")
    
    def unexpected_error(self, error_message):
        """Log unexpected error"""
        self.logger.error(f"Unexpected error: {error_message}")
