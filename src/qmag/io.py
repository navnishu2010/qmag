import pandas as pd
from pathlib import Path
from typing import List, Optional

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")

    def load(self, names: Optional[List[str]] = None):
        """
        Loads instrument data, handling metadata comments (%) and semi-colon delimiters.
        
        Args:
            names: List of column names (e.g. ['time', 'frequency']).
                   If None, it defaults to ['time', 'signal'].
        """
        if names is None:
            names = ['time', 'signal']

        # 1. comment='%': Skips all lines starting with %
        # 2. sep=';': Tells pandas to split data at semicolons
        # 3. header=None: Tells pandas there is no header row (since we skipped it)
        df = pd.read_csv(
            self.filepath, 
            sep=';', 
            comment='%', 
            header=None, 
            names=names,
            engine='python' # Safer for variable whitespace
        )
        
        return df
