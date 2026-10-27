import pandas as pd

df = pd.read_csv("data/online_shoppers_intention.csv")

print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\nNumerical summary:")
print(df.describe().round(2))

print("\nUnique values in categorical columns:")
for col in ["Month", "VisitorType", "Weekend", "Revenue"]:
    print(f"\n{col}:")
    print(df[col].value_counts())
