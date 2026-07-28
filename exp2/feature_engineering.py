import numpy as np
import time
import pandas as pd
import polars as pl
from scipy.stats import entropy
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.fft import fft
from scipy.signal import periodogram
df = pl.read_csv("iot_telemetry_data.csv")
pdf = df.to_pandas()
pdf = pdf.drop_duplicates()
pdf = pdf.sort_values("ts")
pdf = pdf.sort_values("ts")
sensor_cols = ["co", "humidity", "lpg", "smoke", "temp"]

start = time.perf_counter()

#statistical features
statistical_features = {}
for col in sensor_cols:
    signal = pdf[col].values
    statistical_features[col] = {
        "Mean": np.mean(signal),
        "Median": np.median(signal),
        "Variance": np.var(signal),
        "Standard Deviation": np.std(signal),
        "RMS": np.sqrt(np.mean(signal**2)),
        "Energy": np.sum(signal**2),
        "Entropy": entropy(np.histogram(signal, bins=20)[0] + 1),
        "Kurtosis": kurtosis(signal),
        "Skewness": skew(signal)
    }
stat_df = pd.DataFrame(statistical_features).T
# print(stat_df)


#time domain features
time_features = {}
for col in sensor_cols:
    signal = pdf[col].values
    peak = np.max(signal)
    peak_to_peak = np.ptp(signal)
    zero_crossings = np.where(np.diff(np.sign(signal - np.mean(signal))))[0]
    sma = np.sum(np.abs(signal))
    time_features[col] = {
        "Peak": peak,
        "Peak-to-Peak": peak_to_peak,
        "Zero Crossing Rate": len(zero_crossings) / len(signal),
        "Signal Magnitude Area": sma
    }
time_df = pd.DataFrame(time_features).T
# print(time_df)

#FFT
frequency_features = {}
for col in sensor_cols:
    signal = pdf[col].values
    fft_values = np.abs(fft(signal))
    dominant_frequency = np.argmax(fft_values[1:]) + 1
    spectral_energy = np.sum(fft_values**2)
    spectral_entropy = entropy(
        fft_values / np.sum(fft_values)
    )
    frequencies, psd = periodogram(signal)
    average_psd = np.mean(psd)
    frequency_features[col] = {
        "Dominant Frequency": dominant_frequency,
        "Spectral Energy": spectral_energy,
        "Spectral Entropy": spectral_entropy,
        "Power Spectral Density": average_psd
    }
freq_df = pd.DataFrame(frequency_features).T
# print(freq_df)

windows = [10,20,50,100]
for window in windows:
    for col in sensor_cols:
        pdf[f"{col}_MA_{window}"] = (
            pdf[col]
            .rolling(window)
            .mean()
        )
for window in windows:
    for col in sensor_cols:
        pdf[f"{col}_VAR_{window}"] = (
            pdf[col]
            .rolling(window)
            .var()
        )
for window in windows:
    for col in sensor_cols:
        pdf[f"{col}_EMA_{window}"] = (
            pdf[col]
            .ewm(span=window)
            .mean()
        )
# print(pdf.head())

end = time.perf_counter()

print(f"Feature Extraction Time: {end-start:.4f} seconds")

# pdf.to_csv(
#     "environment_feature_engineered.csv",
#     index=False
# )