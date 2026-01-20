import pandas as pd
import os
from src.utils.io_utils import load_csv, save_csv
from src.config.config import DATA_RAW_PATH, DATA_CLEAN_PATH


# src/data_preprocessing.py

def preprocess_run():
    # Load raw data2
    # data_path = "data/raw/sales.csv"  # adjust path if needed
    # if not os.path.exists(data_path):
    #     raise FileNotFoundError(f"{data_path} not found")

    df = pd.read_csv(DATA_RAW_PATH)

    # Example preprocessing
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna()  # simple drop NA for demo

    return df  # <-- IMPORTANT: return df


