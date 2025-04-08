# data_loader.py
import pandas as pd

# 集中管理所有数据集路径
DATA_PATHS = {
    # Physics
    "college_physics": "./Data/college_physics.csv",
    "high_school_physics": "./Data/high_school_physics.csv",
    "gpqa_physics": "./Data/gpqa_physics.csv",

    # Bio
    "college_biology": "./Data/college_biology.csv",
    "high_school_biology": "./Data/high_school_biology.csv",
    "gpqa_biology": "./Data/gpqa_biology.csv",

    # Chem
    "college_chemistry": "./Data/college_chemistry.csv",
    "high_school_chemistry": "./Data/high_school_chemistry.csv",
    "gpqa_chemistry": "./Data/gpqa_chemistry.csv"

    #
}


def load_dataset(dataset_name, sample_size=100):

    if dataset_name not in DATA_PATHS:
        raise ValueError(f"Unknown {dataset_name}, select: {list(DATA_PATHS.keys())}")

    df = pd.read_csv(DATA_PATHS[dataset_name]).iloc[:sample_size]
    required_columns = ['question', 'A', 'B', 'C', 'D', 'answer']
    return df[required_columns]

def load_datasets():
    return {
        "college_physics": load_dataset("college_physics", sample_size=100),
        "high_school_physics": load_dataset("high_school_physics", sample_size=150),
        "college_biology": load_dataset("college_biology", sample_size=140),
        "high_school_biology": load_dataset("high_school_biology", sample_size=310),
        "college_chemistry": load_dataset("college_chemistry", sample_size=100),
        "high_school_chemistry": load_dataset("high_school_chemistry", sample_size=200),
        "gpqa_physics": load_dataset("gpqa_physics", sample_size=180),
        "gpqa_chemistry": load_dataset("gpqa_chemistry", sample_size=180),
        "gpqa_biology": load_dataset("gpqa_biology", sample_size=70),
        "test": load_dataset("college_physics", sample_size=10),
    }