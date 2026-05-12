import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# DataLoader class to handle data loading, distribution plotting, and scaling
class DataLoader:
    # Initialize with the file name and set up the base directory
    def __init__(self, file_name="clean.csv"):
        try:
            self.base_dir = Path(__file__).resolve().parent.parent / "data"
        except NameError:
            self.base_dir = Path.cwd() / "data"
            
        self.file_path = self.base_dir / file_name
        self.df = None

    # Load data from the specified CSV file
    def load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Cannot find file at {self.file_path}")
        
        self.df = pd.read_csv(self.file_path)
        return self.df

    # Plot the distribution of specified columns using seaborn
    def plot_distribution(self, columns):
        if self.df is None: return

        plt.figure(figsize=(16, 10))
        for col in columns:
            plt.subplot(1, len(columns), columns.index(col) + 1)
            sns.histplot(self.df[col], kde=True, color='skyblue', bins=10)
            plt.title(f'Distribution of {col}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    # Scale the specified columns using either StandardScaler or MinMaxScaler
    def scale_data(self, columns, method='standard'):
        if self.df is None: return None
        
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        
        df_scaled = self.df.copy()
        df_scaled[columns] = scaler.fit_transform(self.df[columns])
        
        return df_scaled



# Example usage
if __name__ == "__main__":
    loader = DataLoader("clean.csv")

    try:
        df = loader.load_data()
        columns = ['food_weight_g', 'sleep_hours']
        loader.plot_distribution(columns)

        df_scaled = loader.scale_data(['food_weight_g', 'sleep_hours'], method='standard')
        print(df_scaled.head())
    except Exception as e: print(e)