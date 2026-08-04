import polars as pl

# Load datasets
train_df = pl.read_csv("train.csv")
test_df = pl.read_csv("test.csv")

# Combine train and test for overall analysis
df = pl.concat([train_df, test_df])

print(train_df.shape);