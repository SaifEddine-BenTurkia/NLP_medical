def load_data(file_path):
    import pandas as pd
    return pd.read_csv(file_path)

def save_data(df, file_path):
    df.to_csv(file_path, index=False)

def print_dataframe_info(df):
    print("DataFrame Info:")
    print(df.info())
    print("First few rows:")
    print(df.head())

def check_missing_values(df):
    return df.isnull().sum()

def drop_missing_values(df):
    return df.dropna()