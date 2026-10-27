# Customer Intelligence & Purchase Analytics

## Objective

Explore e-commerce browsing behaviour and develop a machine-learning model to predict whether a browsing session results in a purchase.

## Dataset

Online Shoppers Purchasing Intention Dataset from the UCI Machine Learning Repository.

* 12,330 browsing sessions
* 18 columns
* 1,908 purchasing sessions
* 10,422 non-purchasing sessions
* 0 missing values
* 125 exact duplicate rows in the original dataset

## Exploratory Findings

* Approximately 15.47% of sessions resulted in purchases.
* Purchasing sessions had higher average product-page duration.
* Purchasing sessions visited more product-related pages on average.
* Purchase rates differed across visitor types.

These are observed associations, not evidence of causation.

## Initial Model

A logistic regression classifier was trained using a stratified train/test split.

### Model comparison

| Metric             | With PageValues | Without PageValues |
| ------------------ | --------------: | -----------------: |
| Accuracy           |            0.85 |               0.63 |
| Purchase precision |            0.52 |               0.27 |
| Purchase recall    |            0.79 |               0.76 |
| Purchase F1-score  |            0.62 |               0.40 |

The results show that model performance is sensitive to the inclusion of PageValues. Further investigation is required to establish whether this feature is appropriate for the intended prediction scenario.

## Limitations

* Model performance has only been evaluated using one train/test split.
* Feature leakage and prediction-time availability require further investigation.
* The relationship between browsing behaviour and purchases does not establish causation.
* The initial model requires further evaluation before practical use.
