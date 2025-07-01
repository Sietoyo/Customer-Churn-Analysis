# E-Commerce Churn Analysis Using Random Forest

##  Overview

Customer retention is vital for the long-term success of e-commerce businesses. Understanding which customers are likely to churn can help businesses proactively engage with them, reducing potential revenue loss. In this project, a machine learning model was developed to predict customer churn using historical behavioral data.

By leveraging a Random Forest Classifier, this project provides actionable insights into customer retention and allows segmentation of customers by their risk of churning, helping the business make data-driven decisions.

---

##  Problem Statement

Customer churn significantly impacts profitability in highly competitive e-commerce markets. The objective is to develop a predictive model that accurately classifies whether a customer is likely to churn, and to segment customers by churn risk to inform retention strategies.

---

##  Objectives

**QuidMetrics** is tasked with developing a predictive machine learning solution to help e-commerce platforms identify customers at risk of churn. By analyzing customer behavior and transaction history, this solution will enable proactive engagement and retention strategies, ultimately improving customer lifetime value and reducing revenue loss.

To achieve this objective, **QuidMetrics** will:

i. **Load and analyze the provided customer dataset**, identifying key behavioral and transactional indicators linked to churn.  
ii. **Clean the dataset**, handling missing values and encoding categorical variables to ensure data quality.  
iii. **Engineer relevant features** that may improve model performance and provide deeper business insights.  
iv. **Develop and train a Random Forest classifier** to predict customer churn based on historical patterns.  
v. **Evaluate model performance** using appropriate metrics such as ROC AUC, accuracy, precision, and recall to ensure robustness.  
vi. **Determine feature importance**, identifying which variables contribute most significantly to churn prediction.  
vii. **Segment customers** into risk categories (*Low*, *Medium*, *High*) based on predicted churn probability to guide targeted retention efforts.  
viii. **Explain the model outputs and insights** in clear, non-technical language for business stakeholders.  
ix. **Optimize model performance** through hyperparameter tuning using cross-validation techniques.  


By implementing this end-to-end solution, **QuidMetrics** aims to deliver a highly accurate and interpretable model that supports data-driven decision-making for customer retention in e-commerce.


---

##  Data Description and Sources

This dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) and contains customer-level data from an e-commerce platform, aimed at understanding the factors that contribute to customer churn.

It includes **5,630 records**, each labeled to indicate whether a customer has **churned (`1`)** or remained **active (`0`)**, making it suitable for a binary classification task. The data spans behavioral, transactional, and demographic features.

---

### 🔹 Target Variable

- `Churn`:  
  - `1` – Customer has churned  
  - `0` – Customer is active  

---

### 🔹 Key Features

| Feature | Description |
|--------|-------------|
| `Tenure` | Months the customer has been active |
| `PreferredLoginDevice` | Device used to access the platform |
| `CityTier` | Customer’s city classification (1, 2, or 3) |
| `PreferredPaymentMode` | Most used payment method |
| `HourSpendOnApp` | Average hours spent on the app daily |
| `OrderCount` | Total number of orders placed |
| `CashbackAmount` | Total cashback received |
| `OrderAmountHikeFromlastYear` | % increase in order value from the previous year |
| `Complain` | Whether the customer has lodged complaints |
| `SatisfactionScore` | Customer satisfaction rating (1–5) |
| `MaritalStatus`, `Gender`, `NumberOfAddress`, etc. | Additional demographic/contextual data

---

###  Files Used

- `customer_data.csv` – Main dataset used for modeling
- `segmented_customers.csv` – Output file with customers grouped by churn risk level
 

---

##  Methodology

This project followed a standard machine learning workflow, from cleaning raw data to building and evaluating a predictive model. Here's a breakdown of how it all came together:


###  Data Preprocessing

Before diving into modeling, I made sure the dataset was clean and ready:

i. Filled missing values in numeric columns using **mean imputation**  
ii. Converted categorical variables into numeric format using **one-hot encoding** via `pd.get_dummies()`  
iii. Defined `Churn` as the target variable  
iv. Split the data into **training (70%)** and **testing (30%)** sets to evaluate model performance fairly  

---

###  Exploratory Data Analysis (EDA)

While EDA wasn’t the main focus here, I took a quick look to understand basic trends and relationships:

i. Checked churn distribution to understand class balance
ii. Reviewed feature ranges and distributions
iii. Spotted a few weak correlations and some dominant variables


---

###  Model Building: Random Forest Classifier

**Random Forest Classifier** is preferred due to its robustness, ability to handle mixed data types, and interpretability. The model was trained using the following parameters:

- `n_estimators = 200`  
- `max_depth = 20`  
- `min_samples_split = 2`

The goal was to strike a balance between performance and overfitting, especially given the moderately sized dataset.

---

###  Model Evaluation

To measure model performance, the standard classification metric was used:

i. **Classification Report** (precision, recall, F1 score)
ii. **Confusion Matrix**
iii. **ROC AUC Score**

---

The model demonstrated **strong overall accuracy (95%)** and an excellent **ROC AUC score of 0.97**, suggesting it’s highly capable of distinguishing between churned and retained customers.

While the model performed almost flawlessly in identifying active customers (Class 0), it still achieved solid results in predicting churners (Class 1), with:
i. **Precision** of **93%**: Most of the churn predictions were correct
ii. **Recall** of **74%**: It caught about three-quarters of actual churners
iii. **F1 Score** of **82%**: A good balance between precision and recall

These results make the model suitable for real-world churn prediction use cases, where prioritizing **recall for churners** can help companies intervene early and retain high-risk customers.


###  Feature Importance

Using the Random Forest’s built-in feature importance, I visualized which variables had the most impact on churn predictions. This helped highlight key drivers such as:

- Tenure, Order count, Cashback received, Order amount change from last year

---

###  Customer Segmentation by Churn Risk

To make the model actionable, I segmented customers based on their predicted churn probability:

i. 🔴 **High Risk**: > 75% likelihood of churn  
ii. 🟠 **Medium Risk**: 50–75%  
iii. 🟢 **Low Risk**: ≤ 50%

This segmentation was saved into a new file, `segmented_customers.csv`, to support future marketing or retention campaigns.

---

###  Hyperparameter Tuning

I used `GridSearchCV` to fine-tune the model and squeeze out better performance. Parameters optimized included:

i. `n_estimators`: [100, 200, 300]
ii. `max_depth`: [10, 20, 30]
iii. `min_samples_split`: [2, 5, 10]

A 5-fold cross-validation was used during tuning to ensure stability and generalization.

---

##  File Structure
##  Tools and Libraries

i. Python (`Pandas`, `NumPy`)  
ii. Scikit-learn (`RandomForestClassifier`, `GridSearchCV`)  
iii. Matplotlib, Seaborn (Visualizations)  

##  Key Insights

i. Feature importance revealed top predictors of churn, such as behavioral indicators and transaction patterns  
ii. The Random Forest model provided high predictive power and interpretability  
iii. Customer segmentation by churn risk allows for actionable targeting in retention campaigns  


##  Conclusion

This project showcases how machine learning can drive business value by predicting customer churn and guiding strategic customer engagement. Random Forest proved effective in modeling churn behavior with a transparent interpretation of feature importance and customer segmentation.


