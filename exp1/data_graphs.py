import matplotlib.pyplot as plt
import polars as pl

df = pl.read_csv("dataset.csv")

# Histogram
# features = ["acc_x", "acc_y", "acc_z"]

# for feature in features:
#     plt.figure(figsize=(6,4))
#     plt.hist(df[feature].to_numpy(), bins=50)
#     plt.title(f"Distribution of {feature}")
#     plt.xlabel(feature)
#     plt.ylabel("Frequency")
#     plt.grid(alpha=0.3)
#     plt.show()


# Correlation Matrix
# corr = df.select(["acc_x", "acc_y", "acc_z"]).corr()

# plt.figure(figsize=(6,5))

# plt.imshow(corr.to_numpy(), cmap="coolwarm")

# plt.xticks([0,1,2], ["x","y","z"])
# plt.yticks([0,1,2], ["x","y","z"])

# plt.colorbar(label="Correlation")
# plt.title("Correlation Matrix")

# plt.show()


# Class Balance
# activity_counts = (
#     df.group_by("Activity")
#       .len()
#       .sort("Activity")
# )

# print(activity_counts)

# plt.figure(figsize=(8,5))

# plt.bar(
#     activity_counts["Activity"].to_list(),
#     activity_counts["len"].to_list()
# )

# plt.xlabel("Activity")
# plt.ylabel("Count")
# plt.title("Class Distribution")

# plt.show()


#Outliers Detection
for feature in ["acc_x", "acc_y", "acc_z"]:

    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df.filter(
        (pl.col(feature) < lower) |
        (pl.col(feature) > upper)
    )

    print(f"{feature}: {outliers.height} outliers")