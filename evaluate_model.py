import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("data/online_shoppers_intention.csv")
df = df.drop_duplicates()

X = df.drop(columns=["Revenue", "PageValues"])
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

predictions = baseline.predict(X_test)

print("Baseline accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))
