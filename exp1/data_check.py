import polars as pl

df = pl.read_csv("dataset.csv")

duplicates = df.height - df.unique().height
print(duplicates)