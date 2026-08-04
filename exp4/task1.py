import pandas as pd

# ==========================================================
# Task 1 - Dataset Loading
# Tiny Neural Network using Iris Dataset
# ==========================================================

# Load Dataset
df = pd.read_csv("Iris.csv")

print("=" * 60)
print("IRIS DATASET SUMMARY")
print("=" * 60)

# Display first few rows
print("\nFirst 5 Rows")
print(df.head())

# Dataset Shape
print("\nDataset Shape :", df.shape)

# Feature Names
features = list(df.columns[1:-1])   # Ignore Id and Species

print("\nFeature Names")
for feature in features:
    print("-", feature)

# Activity/Class Labels
print("\nSpecies Labels")
species = sorted(df["Species"].unique())

for label in species:
    print(label)

# Number of Samples
print("\nNumber of Samples :", len(df))

# Number of Features
print("Number of Features :", len(features))

# Number of Classes
print("Number of Classes :", len(species))

# Class Distribution
print("\nClass Distribution")
print(df["Species"].value_counts())

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

print("\nTask 1 Completed Successfully.")