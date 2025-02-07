import pandas as pd
import numpy as np

class DataLoader:
    """
    A class for preprocessing a dataset, including loading, cleaning, and handling missing values.
    """

    def __init__(self, filepath, logger):
        """
        Initializes the DataPreprocessor with a dataset filepath and logger.
        """
        self.filepath = filepath
        self.logger = logger
        self.data = None
    
    def load_dataset(self):
        """
        Loads the dataset from the specified filepath.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            self.logger.info("Dataset loaded successfully.")
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            return None  # Return None if there's an error loading the dataset
        