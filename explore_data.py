import pandas as pd

df = pd.read_csv("data/online_shoppers_intention.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset dimensions:")
print(df.shape)

print("\nColumn names and data types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDataset summary:")
print(df.describe())

print("\nRevenue distribution:")
print(df["Revenue"].value_counts())

print("\nRevenue percentages:")
print(df["Revenue"].value_counts(normalize=True) * 100)

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nActual dataset dimensions:")
print(df.shape)

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nExamples of duplicate rows:")
print(df[df.duplicated(keep=False)].head(10))

print("\nData types:")
print(df.dtypes)

print("\nUnique values in Revenue:")
print(df["Revenue"].unique())

# Compare browsing behaviour for sessions with and without purchases

print("\nAverage product-related page duration:")
print(df.groupby("Revenue")["ProductRelated_Duration"].mean())

print("\nAverage number of product-related pages visited:")
print(df.groupby("Revenue")["ProductRelated"].mean())

print("\nAverage page value:")
print(df.groupby("Revenue")["PageValues"].mean())

print("\nPurchase rate by visitor type:")
print(
df.groupby("VisitorType")["Revenue"]
.mean()
.mul(100)
.round(2)
)
