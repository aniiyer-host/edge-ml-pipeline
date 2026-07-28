import polars as pl

# Load datasets
train_df = pl.read_csv("train.csv")
test_df = pl.read_csv("test.csv")

# Combine train and test for overall analysis
df = pl.concat([train_df, test_df])

# -----------------------------
# Dataset Shape
# -----------------------------
print("Dataset Shape")
print(f"Rows (Samples): {df.height}")
print(f"Columns (Features): {df.width}")

# -----------------------------
# Feature Names
# -----------------------------
print("\nFeature Names:")
for feature in df.columns:
    print(feature)

# -----------------------------
# Activity Labels
# -----------------------------
activity_column = "Activity"   # Change if your column has a different name

activities = df[activity_column].unique().sort()

print("\nActivity Labels:")
for activity in activities:
    print(activity)

# -----------------------------
# Class Distribution
# -----------------------------
print("\nClass Distribution:")
class_distribution = (
    df.group_by(activity_column)
      .len()
      .sort(activity_column)
)

print(class_distribution)

# -----------------------------
# Dataset Summary
# -----------------------------
print("\nDataset Summary")
print(f"Number of Samples : {df.height}")
print(f"Number of Features: {df.width - 1}")   # excluding activity label
print(f"Number of Activity Classes: {len(activities)}")
