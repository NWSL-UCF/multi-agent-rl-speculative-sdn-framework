import pandas as pd
import os

class DataLoader:
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        # Get pcap base path from command line argument or use default, expand tilde
        self.base_pcap_path = os.path.expanduser(getattr(args, 'pcap_base_path', '~/Pcap'))
        
    def load_data(self):
        """Load and preprocess dataset based on arguments"""
        # Load dataset based on arguments (hardcoded to dataset 1)
        # Map trace number to filename
        filename = f"{self.args.trace}.csv"
        dataset = self._load_dataset_file(filename)
        
        # Remove columns
        rem = ['No.', 'Time', 'Protocol', 'Length', 'Info'] 
        remove = ['No.', 'Source', 'Destination', 'Protocol', 'Length', 'Info']
        
        # Create value dataframe (time information)
        value = dataset.copy()
        value.drop(remove, axis=1, inplace=True)
        
        # Create main dataset
        dataset.drop(rem, axis=1, inplace=True)
        table = dataset.drop_duplicates()
        
        # Create controller table with clear column structure
        controller_table = self._create_controller_table(table)
        
        # Create switch table
        switch_table = self._create_switch_table()
        
        return controller_table, switch_table, dataset, value
    
    def _load_dataset_file(self, filename):
        """Load dataset file with simple path handling"""
        # Try common locations
        possible_paths = [
            filename,  # Current directory
            os.path.join(self.base_pcap_path, filename),  # Base pcap path
            os.path.join('..', 'Pcap', filename),  # Parent Pcap directory
        ]
        
        for file_path in possible_paths:
            if os.path.exists(file_path):
                if self.logger:
                    self.logger.info(f"Loading dataset from: {file_path}")
                return pd.read_csv(file_path)
        
        raise FileNotFoundError(
            f"Could not find dataset file '{filename}' in any of these locations:\n" +
            "\n".join(f"  - {path}" for path in possible_paths) +
            f"\n\nPlease ensure the file exists or set the correct base path using --pcap_base_path"
        )
    
    def _create_controller_table(self, table):
        """Create controller table with clear, readable column structure"""
        # Start with the base table
        controller_table = table.copy()
        
        # Add flow tracking columns with clear names
        controller_table['hit_count'] = 0
        controller_table['miss_count'] = 0
        
        # Sort controller table based on SDN mode
        if self.args.ordering == "trace":
            pass
        elif self.args.ordering == "source":
            controller_table = controller_table.sort_values(by='Source', ascending=True)
        elif self.args.ordering == "destination":
            controller_table = controller_table.sort_values(by='Destination', ascending=True)
        
        # Add performance tracking columns
        controller_table['accumulated_reward'] = 0.0
        controller_table['total_packet_count'] = 0
        controller_table['was_hit_this_iteration'] = 0
        controller_table['spatial_reward_component'] = 0.0
        controller_table['is_speculated_flow'] = 0
        
        return controller_table.reset_index(drop=True)
    
    def _create_switch_table(self):
        """Create switch table with clear column structure"""
        switch_table = pd.DataFrame({
            'Source': [],
            'Destination': [],
            'flow_age': [],
            'is_speculative': [],  # Add is_speculative column by default
            'hit_count': []  # Add this line
        })
        return switch_table
    

