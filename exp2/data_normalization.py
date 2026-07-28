import polars as pl
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler
)
import time

df = pl.read_csv("iot_telemetry_data.csv")
pdf = df.to_pandas()
sensor_cols = ["co", "humidity", "lpg", "smoke", "temp"]
X = pdf[sensor_cols].copy()


# Min-Max Scaling
# start = time.perf_counter()
# minmax = MinMaxScaler()
# X_minmax = minmax.fit_transform(X)
# end = time.perf_counter()
# print("Min-Max Time:", end-start)

#Standardization
# start = time.perf_counter()
# standard = StandardScaler()
# X_standard = standard.fit_transform(X)
# end = time.perf_counter()
# print("Standard Time:", end-start)

#Robust Scaling
# start = time.perf_counter()
# robust = RobustScaler()
# X_robust = robust.fit_transform(X)
# end = time.perf_counter()
# print("Robust Time:", end-start)

#Max Abs scaling time
# start = time.perf_counter()
# maxabs = MaxAbsScaler()
# X_maxabs = maxabs.fit_transform(X)
# end = time.perf_counter()
# print("MaxAbs Time:", end-start)

# Memory Usage
# print("Original :", X.memory_usage(deep=True).sum()/1024, "KB")
# print("MinMax   :", X_minmax.nbytes/1024, "KB")
# print("Standard :", X_standard.nbytes/1024, "KB")
# print("Robust   :", X_robust.nbytes/1024, "KB")
# print("MaxAbs   :", X_maxabs.nbytes/1024, "KB")