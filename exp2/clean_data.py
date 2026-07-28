import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pl.read_csv("iot_telemetry_data.csv")
pdf = df.to_pandas()

# print(df.null_count())
# print("Total Missing Values:", df.null_count().sum_horizontal())

# duplicates = df.height - df.unique().height
# print(duplicates)

# before = len(pdf)
# duplicates = pdf.duplicated().sum()
# print("Duplicate Records:", duplicates)
# pdf = pdf.drop_duplicates()
# after = len(pdf)
# print("Rows Before:", before)
# print("Rows After :", after)
# print("Duplicates Removed:", before-after)

# 3 point rolling mean filter 
sensor_cols = ["co","humidity","lpg","smoke","temp"]
# for col in sensor_cols:
#     pdf[col] = pdf[col].rolling(window=3, min_periods=1).mean()


# Z- Score
# from scipy.stats import zscore
# import numpy as np
# z = np.abs(zscore(pdf[sensor_cols]))
# outliers_z = (z > 3).any(axis=1)
# print("Outliers (Z-score):", outliers_z.sum())

#IQR
# outliers = 0
# for col in sensor_cols:
#     Q1 = pdf[col].quantile(0.25)
#     Q3 = pdf[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
#     outliers += ((pdf[col] < lower) | (pdf[col] > upper)).sum()
# print("Possible Outliers (IQR):", outliers)

#Isolation forest
# from sklearn.ensemble import IsolationForest
# iso = IsolationForest(
#     contamination=0.02,
#     random_state=42
# )
# pred = iso.fit_predict(pdf[sensor_cols])
# print("Outliers (Isolation Forest):", (pred == -1).sum())