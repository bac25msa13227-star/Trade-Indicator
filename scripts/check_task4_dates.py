"""Check task4 dataset date range."""
import pandas as pd

# Check task4 dataset
df = pd.read_csv("outputs/task4_training_dataset_acc1.csv", usecols=["time", "trade_side"], nrows=3)
print("First rows:")
print(df)

# Full date range (just time column)
df2 = pd.read_csv("outputs/task4_training_dataset_acc1.csv", usecols=["time"])
df2["time"] = pd.to_datetime(df2["time"])
t_min = df2["time"].min()
t_max = df2["time"].max()
print(f"Date range: {t_min} to {t_max}")
print(f"Total rows: {len(df2)}")
print(f"Duration: {(t_max - t_min).days} days")
