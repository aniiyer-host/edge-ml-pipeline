from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import polars as pl
import pandas as pd


df = pl.read_csv("iot_telemetry_data.csv")
pdf = df.to_pandas()


# label_df = pdf.copy()
# le = LabelEncoder()
# label_df["device"] = le.fit_transform(label_df["device"])
# print(label_df.head())

# onehot_df = pd.get_dummies(
#     pdf,
#     columns=["device"]
# )
# print(onehot_df.head())

# freq_df = pdf.copy()
# freq = freq_df["device"].value_counts()
# freq_df["device"] = freq_df["device"].map(freq)
# print(freq_df.head())

pdf["light"] = pdf["light"].astype(int)
pdf["motion"] = pdf["motion"].astype(int)