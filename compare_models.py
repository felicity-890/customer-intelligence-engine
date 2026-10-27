import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
)

# 1. Load and prepare the data

df = pd.read_csv("data/online_shoppers_intention.csv")
df = df.drop_duplicates()

X = df.drop(columns=["Revenue", "PageValues"])
y = df["Revenue"]

# 2. Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y,
)

# 3. Identify column types

categorical_cols = X.select_dtypes(
include=["object", "str", "bool", "category"]
).columns.tolist()

numerical_cols = X.select_dtypes(
include=["number"]
).columns.tolist()

# 4. Prepare the numerical and categorical features

preprocessor = ColumnTransformer(
transformers=[
("num", StandardScaler(), numerical_cols),
("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
]
)

# 5. Define both models

models = {
"Baseline": DummyClassifier(strategy="most_frequent"),
"Logistic Regression": Pipeline(
steps=[
("preprocessor", preprocessor),
(
"classifier",
LogisticRegression(
max_iter=1000,
class_weight="balanced",
),
),
]
),
}

# 6. Train and evaluate each model

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    results.append(
    {
        "Model": name,
        "Accuracy": round(
            accuracy_score(y_test, predictions), 3
        ),
        "Purchase Precision": round(
            precision_score(
                y_test, predictions, zero_division=0
            ),
            3,
        ),
        "Purchase Recall": round(
            recall_score(y_test, predictions), 3
        ),
        "Purchase F1": round(
            f1_score(y_test, predictions), 3
        ),
    }
)

# 7. Display the comparison

results_df = pd.DataFrame(results)

print("\nModel comparison:")
print(results_df.to_string(index=False))

#Saving the results
results_df.to_csv("model_comparison.csv", index=False)
print("\nResults saved to model_comparison.csv")