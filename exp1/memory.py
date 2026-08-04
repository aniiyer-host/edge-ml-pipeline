import polars as pl

df = pl.read_csv("dataset.csv")

samples = df.shape[0]
features = df.shape[1]-1

print("Samples = ",samples," Features = ",features)

memory = samples * features * 4

print(memory/1024 , "KB")