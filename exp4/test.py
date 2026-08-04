# import os

# print("Current Working Directory:")
# print(os.getcwd())

# print("\nFiles in current directory:")
# print(os.listdir())

# import os

# print("Does train folder exist?", os.path.exists("train"))

# if os.path.exists("train"):
#     print("Number of files:", len(os.listdir("train")))
#     print("First 10 files:")
#     print(os.listdir("train")[:10])

# import matplotlib.pyplot as plt
# import numpy as np

# X = np.load("X_train.npy")
# y = np.load("y_train.npy")

# print(X.shape)
# print(X.min(), X.max())
# print(y[:20])

# plt.imshow(X[0])
# plt.title(f"Label = {y[0]}")
# plt.show()

# import pandas as pd

# TRAIN_CSV = "Human Action Recognition/Training_set.csv"
# TRAIN_DIR = "Human Action Recognition/train"

# train_df = pd.read_csv(TRAIN_CSV)

# print(train_df.head())