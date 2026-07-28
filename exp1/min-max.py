import polars as pl
import matplotlib.pyplot as plt

df = pl.read_csv("dataset.csv")
features = ["acc_x", "acc_y", "acc_z"]

# Min-Max Scaling
df_minmax = df.clone()

for feature in features:

    minimum = df[feature].min()
    maximum = df[feature].max()

    df_minmax = df_minmax.with_columns(
        (
            (pl.col(feature) - minimum) /
            (maximum - minimum)
        ).alias(feature)
    )

print(df_minmax.head())

# Standardization 
df_standard = df.clone()

for feature in features:

    mean = df[feature].mean()
    std = df[feature].std()

    df_standard = df_standard.with_columns(
        (
            (pl.col(feature) - mean) /
            std
        ).alias(feature)
    )

print(df_standard.head())

for feature in features:

    plt.figure(figsize=(15,4))

    plt.subplot(1,3,1)
    plt.hist(df[feature].to_numpy(), bins=50)
    plt.title(f"{feature} Original")

    plt.subplot(1,3,2)
    plt.hist(df_minmax[feature].to_numpy(), bins=50)
    plt.title(f"{feature} Min-Max")

    plt.subplot(1,3,3)
    plt.hist(df_standard[feature].to_numpy(), bins=50)
    plt.title(f"{feature} Standardized")

    plt.tight_layout()
    plt.show()


