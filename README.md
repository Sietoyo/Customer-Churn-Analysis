E-Commerce Churn Analysis Using Random Forest
📌 Overview
Customer retention is vital for the long-term success of e-commerce businesses. Understanding which customers are likely to churn can help businesses proactively engage with them, reducing potential revenue loss. In this project, a machine learning model was developed to predict customer churn using historical behavioral data.

By leveraging a Random Forest Classifier, this project provides actionable insights into customer retention and allows segmentation of customers by their risk of churning, helping the business make data-driven decisions.

❓ Problem Statement
Customer churn significantly impacts profitability in highly competitive e-commerce markets. The objective is to develop a predictive model that accurately classifies whether a customer is likely to churn, and to segment customers by churn risk to inform retention strategies.

🎯 Objectives
Load and preprocess customer data

Explore and understand data patterns through EDA

Train a Random Forest classifier to predict churn

Evaluate the model with classification metrics and ROC AUC

Analyze feature importance

Segment customers by predicted churn probability

Tune hyperparameters for performance optimization

🧾 Dataset Description
Source: Internal CSV file (customer_data.csv)

Total Records: ~15,819

Target Variable: Churn (binary classification)

Features: Behavioral, transactional, and demographic data

🧪 Methodology
🔧 Data Preprocessing
Missing values were handled using mean imputation for numeric columns.

Categorical variables were encoded using one-hot encoding (pd.get_dummies()).

Target variable: Churn

Feature matrix (X) and target (y) were split into train-test sets.

📊 Exploratory Data Analysis (EDA)
Note: EDA was briefly implied. Visualizations such as class distributions, correlation heatmaps, or univariate plots can be added for a more detailed portfolio.

⚙️ Model Building: Random Forest Classifier
Parameters:

n_estimators = 200

max_depth = 20

min_samples_split = 2

Trained on 70% training data, tested on 30% holdout set.

📈 Evaluation Metrics
Classification Report

Confusion Matrix

ROC AUC Score

The model achieved good performance in identifying churners with a focus on precision and recall for the positive class (Churn = 1).

🔍 Feature Importance
Feature importances were extracted and visualized using Seaborn to determine which factors most influenced the churn prediction.

🧠 Customer Segmentation by Churn Risk
Using churn probabilities, customers were segmented into:

High Risk: Probability > 0.75

Medium Risk: 0.5 < Probability ≤ 0.75

Low Risk: Probability ≤ 0.5

Segmented results were saved to a new file: segmented_customers.csv.

🛠️ Hyperparameter Tuning
Conducted using GridSearchCV

Parameters tuned:

n_estimators: [100, 200, 300]

max_depth: [10, 20, 30]

min_samples_split: [2, 5, 10]

Cross-validation: 5-fold

Optimized model can be further used in production scenarios

📁 File Structure
bash
Copy
Edit
├── customer_data.csv           # Original dataset
├── E-Commerce Churn Analysis with Random Forest.ipynb  # Main notebook
├── segmented_customers.csv     # Final output with risk categories
└── README.md                   # Project summary and documentation
🧰 Tools and Libraries
Python (Pandas, NumPy)

Scikit-learn (RandomForestClassifier, GridSearchCV)

Matplotlib, Seaborn (Visualizations)

📌 Key Insights
Feature importance revealed top predictors of churn, such as behavioral indicators and transaction patterns.

The Random Forest model provided high predictive power and interpretability.

Customer segmentation by churn risk allows for actionable targeting in retention campaigns.

✅ Next Steps / Recommendations
Integrate churn prediction into an automated CRM dashboard

Design targeted marketing interventions for each segment

Further explore time-based behavioral features (e.g., recency, frequency)

📄 Conclusion
This project showcases how machine learning can drive business value by predicting customer churn and guiding strategic customer engagement. Random Forest proved effective in modeling churn behavior with a transparent interpretation of feature importance and customer segmentation.
