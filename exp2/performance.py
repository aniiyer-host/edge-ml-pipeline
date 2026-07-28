import time
import psutil
import pandas as pd

original = pd.read_csv("iot_telemetry_data.csv")
engineered = pd.read_csv("environment_feature_engineered.csv")

#time
start = time.perf_counter()
df = pd.read_csv("environment_feature_engineered.csv")
end = time.perf_counter()
execution_time = end - start
print(f"Execution Time: {execution_time:.6f} seconds")

#memory usage
memory = engineered.memory_usage(deep=True).sum()
print(f"Memory Usage: {memory/1024:.2f} KB")

#cpu
cpu = psutil.cpu_percent(interval=1)
print(f"CPU Utilization: {cpu}%")

#features
print("Original Features :", original.shape[1])
print("Engineered Features :", engineered.shape[1])

#Size reduction
original_size = original.memory_usage(deep=True).sum()
engineered_size = engineered.memory_usage(deep=True).sum()
print("Original Size :", original_size/1024, "KB")
print("Engineered Size :", engineered_size/1024, "KB")