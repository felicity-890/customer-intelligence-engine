### Customer Intelligence \& Purchase Analytics



##### Overview



This project explores e-commerce browsing behaviour and develops a machine-learning model to predict whether a browsing session results in a purchase.



It covers data validation, exploratory data analysis (EDA), visualisation, model training, and performance comparison using Python.



##### Objectives



\* Investigate patterns in e-commerce browsing behaviour.

\* Identify behavioural features associated with purchases.

\* Build and evaluate a classification model.

\* Compare model performance with and without the PageValues feature.

\* Establish a reproducible foundation for further experimentation.



##### Dataset



Source: Online Shoppers Purchasing Intention Dataset — UCI Machine Learning Repository

https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention



The dataset contains browsing-session information from an e-commerce website.



Property: Value

Browsing sessions: 12,330

Features, including target: 18

Purchasing sessions: 1,908

Non-purchasing sessions: 10,422

Missing values: 0

Exact duplicate rows in original dataset: 125



The target variable is Revenue, which indicates whether a browsing session resulted in a purchase.



##### Exploratory Data Analysis



The initial analysis identified several patterns:



\* Approximately 15.47% of sessions resulted in purchases.

\* Purchasing sessions had higher average ProductRelated\_Duration.

\* Purchasing sessions visited more product-related pages on average.

\* Purchase rates differed across visitor types.



These findings describe associations in the dataset and do not establish causation.



##### Visualisation



The project includes a comparison of average product-related page duration between purchasing and non-purchasing sessions.



!\[Average product-related page duration by purchase outcome](product\_duration\_by\_revenue.png)



##### Model Development



Logistic regression was used as an initial classification model. Numerical and categorical features were preprocessed using scikit-learn pipelines, with numerical scaling and categorical encoding.



The data was divided into training and testing sets using a stratified split, with an 80:20 ratio and a fixed random seed.



The model used class weighting to account for the imbalance between purchasing and non-purchasing sessions.



##### Logistic Regression: Feature Comparison



Metric: Accuracy | Purchase precision | Purchase recall | Purchase F1-score



With PageValues: 0.85 | 0.52 | 0.79 | 0.62



Without PageValues: 0.63 | 0.27 | 0.76 | 0.40



The results indicate that performance is sensitive to the inclusion of PageValues.



Purchase recall measures the proportion of actual purchasing sessions correctly identified by the model. Purchase precision measures how many sessions predicted to result in a purchase actually resulted in one. The F1-score balances precision and recall.



PageValues requires further investigation to determine whether it is available at the intended prediction time and appropriate for the prediction scenario.



##### Baseline Comparison



A majority-class baseline was also evaluated against logistic regression without PageValues.



Model: Accuracy | Purchase precision | Purchase recall | Purchase F1-score



Majority-class baseline: 0.844 | 0.000 | 0.000 | 0.000



Logistic regression: 0.634 | 0.266 | 0.764 | 0.395



The baseline achieves higher accuracy by predicting that every session will result in no purchase. However, it fails to identify any purchasing sessions.



Logistic regression identifies a larger proportion of purchasing sessions, but generates false-positive predictions and has lower overall accuracy. This illustrates why accuracy alone is insufficient for evaluating an imbalanced classification problem.



##### Repository Contents



check\_data.py — Data validation and quality checks



explore\_data.py — Exploratory data analysis



visualize\_data.py — Data visualisation



train\_model.py — Logistic regression training



evaluate\_model.py — Model evaluation



compare\_models.py — Baseline and logistic regression comparison without PageValues



model\_comparison.csv — Saved baseline comparison results



product\_duration\_by\_revenue.png — Product-related page duration visualisation



requirements.txt — Python dependencies



.gitignore — Git ignore rules



##### Getting Started



###### Prerequisites



\* Python 3.13.5

\* Git



###### Installation



Clone the repository:



git clone https://github.com/felicity-890/customer-intelligence-engine.git



cd customer-intelligence-engine



Create and activate a virtual environment.



For Windows Command Prompt:



python -m venv .venv



.venv\\Scripts\\activate



For macOS or Linux:



python -m venv .venv



source .venv/bin/activate



Install dependencies:



pip install -r requirements.txt



###### Dataset Setup



Download the Online Shoppers Purchasing Intention Dataset from the UCI Machine Learning Repository and place online\_shoppers\_intention.csv inside the data/ directory.



The dataset is not included in this repository.



###### Running the Scripts



Run the scripts from the project root, one at a time:



python check\_data.py



python explore\_data.py



python visualize\_data.py



python train\_model.py



python evaluate\_model.py



python compare\_models.py



Run each script independently to inspect its output.



##### Limitations



\* Evaluation currently relies on a single train/test split.

\* The predictive suitability and availability of PageValues require further investigation.

\* The baseline and logistic regression results illustrate a trade-off between accuracy and identifying purchasing sessions.

\* The impact of duplicate-row handling requires careful consideration.

\* Observed relationships do not establish causation.

\* The model has not been validated for deployment in a real-world business environment.



##### Future Improvements



\* Add automated data-quality checks and clearer validation reporting.

\* Evaluate models using cross-validation and additional classification metrics.

\* Investigate feature availability and potential leakage.

\* Improve reproducibility and automate the data-processing workflow.

\* Explore alternative classification models and decision thresholds.



##### Acknowledgements



Dataset: Online Shoppers Purchasing Intention Dataset, UCI Machine Learning Repository.



