import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pl.read_csv("iot_telemetry_data.csv")

# print(df.head())

# print(df.shape)

# rows, cols = df.shape
# print("Number of Samples:", rows)
# print("Number of Features:", cols)

# print(df.dtypes)

# memory = df.estimated_size()
# print(f"Memory Usage: {memory/1024:.2f} KB")

# print(df.describe())

# print(df.null_count())


pdf = df.to_pandas()


# sensor_cols = ["co", "humidity", "lpg", "smoke", "temp"]
# for col in sensor_cols:
#     plt.figure(figsize=(6,4))
#     plt.hist(pdf[col], bins=20)
#     plt.title(f"Distribution of {col}")
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
#     plt.grid(True)
#     plt.show()


# corr = pdf.corr(numeric_only=True)
# print(corr)
# plt.figure(figsize=(8,6))
# plt.imshow(corr)
# plt.colorbar()
# plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
# plt.yticks(range(len(corr.columns)), corr.columns)
# plt.title("Correlation Matrix")
# plt.show()
