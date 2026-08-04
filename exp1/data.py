import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

df = pl.read_csv("dataset.csv")

sns.boxplot(df["seconds_elapsed"])