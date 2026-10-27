import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load data

df = pd.read_csv("data/online_shoppers_intention.csv")

# 2. Remove exact duplicate rows for this experiment

df = df.drop_duplicates()

# 3. Separate features (X) and target (y)

X = df.drop(columns=["Revenue","PageValues"])
y = df["Revenue"]

# 4. Identify numerical and categorical columns

categorical_cols = X.select_dtypes(
    include=["object", "str", "bool", "category"]
).columns.tolist()
numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

# 5. Prepare each type of feature

preprocessor = ColumnTransformer(
transformers=[
("num", StandardScaler(), numerical_cols),
("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
]
)

# 6. Create a machine-learning pipeline

model = Pipeline([
("preprocessor", preprocessor),
("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

# 7. Split data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Train the model

model.fit(X_train, y_train)

# 9. Evaluate it on unseen test data

y_pred = model.predict(X_test)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
