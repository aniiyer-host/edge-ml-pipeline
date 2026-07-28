import polars as pl

# Load datasets
train_df = pl.read_csv("train.csv")
test_df = pl.read_csv("test.csv")

# Name of the activity column
label_column = "Activity"  

# Separate features and labels
X_train = train_df.drop(label_column)
y_train = train_df[label_column]

X_test = test_df.drop(label_column)
y_test = test_df[label_column]

# Display dataset sizes
print("Training Dataset")
print(f"Samples : {X_train.height}")
print(f"Features: {X_train.width}")

print("\nTesting Dataset")
print(f"Samples : {X_test.height}")
print(f"Features: {X_test.width}")

# Verify class balance
print("\nTraining Class Distribution")
print(train_df.group_by(label_column).len().sort(label_column))

print("\nTesting Class Distribution")
print(test_df.group_by(label_column).len().sort(label_column))