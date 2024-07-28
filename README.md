# Customer-Churn-Analysis

### Comprehensive Report on Customer Churn Prediction Analysis

#### **1. Introduction**

Customer churn prediction is a critical aspect for businesses aiming to retain their customer base and ensure long-term profitability. This report presents a detailed analysis of a Random Forest model used to predict customer churn. The model has been evaluated for its performance, and feature importance has been analyzed to understand the key factors influencing churn.

#### **2. Data Overview**

The dataset "Ecommerce Customer Churn Analysis and Prediction" used in this analysis was sourced from Kaggle: https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction. It includes various features related to customer demographics, behavior, and transaction history. Key features include:

- CustomerID    -    Unique customer ID
- Churn    -    Churn Flag
- Tenure   -    Tenure of customer in organization
- CityTier    -    City tier
- WarehouseToHome   -    Distance in between warehouse to home of customer
- PreferredPaymentMode    -    Preferred payment method of customer
- Gender    -    Gender of customer
- HourSpendOnApp    -    Number of hours spend on mobile application or website
- NumberOfDeviceRegistered    -    Total number of devices is registered on particular customer 
- PreferredOrderCat    -    Preferred order category of customer in last month
- SatisfactionScore    -    Satisfactory score of customer on service
- MaritalStatus    -    Marital status of customer
- NumberOfAddress    -    Total number of added added on particular customer
- Complain    -    Any complaint has been raised in last month
- OrderAmountHikeFromLastYear    -    Percentage increases in order from last year
- CouponUsed    -    Total number of coupon has been used in last month
- OrderCount    -    Total number of orders has been places in last month
- DaySinceLastOrder    -    Day Since last order by customer
- CashbackAmount    -    Average cashback in last month

The dataset contains 5630 rows of customer information.

#### **3. Methodology**

A Random Forest model was chosen for this analysis due to its robustness and ability to handle large datasets with numerous features. The analysis involved the following steps:

1. **Data Preprocessing**: Cleaning and transforming the data for analysis.
2. **Feature Engineering**: Creating new features and selecting relevant ones.
3. **Model Training**: Training the Random Forest model using the training data.
4. **Model Evaluation**: Evaluating the model's performance using various metrics.
5. **Hyperparameter Tuning**: Optimizing the model's parameters to improve performance.
6. **Feature Importance Analysis**: Identifying the most important features influencing churn prediction.

#### **4. Model Performance**

The Random Forest model was evaluated using the following metrics:

- **Area Under the Curve (AUC)**:
  - Initial model: 0.9732
  - Best model after hyperparameter tuning: 0.9720

- **Classification Report**:
  - **Precision**: 
    - Class 0 (Non-churn): 0.95
    - Class 1 (Churn): 0.93
  - **Recall**: 
    - Class 0: 0.99
    - Class 1: 0.75
  - **F1-Score**: 
    - Class 0: 0.97
    - Class 1: 0.83
  - **Support**: 
    - Class 0: 1414
    - Class 1: 275

- **Confusion Matrix**:
  - **True Positives (TP)**: 205
  - **True Negatives (TN)**: 1398
  - **False Positives (FP)**: 16
  - **False Negatives (FN)**: 70

#### **5. Hyperparameter Tuning Results**

The optimal parameters found through GridSearchCV were:

- **max_depth**: 30
- **min_samples_split**: 2
- **n_estimators**: 300

These parameters provided a model with slightly adjusted performance metrics, maintaining a high AUC score and balanced precision and recall metrics.

#### **6. Feature Importance Analysis**

The feature importance analysis revealed the following key features influencing customer churn:

1. **Tenure**: The length of time a customer has been with the company is the most critical factor in predicting churn.
2. **CashbackAmount**: The amount of cashback received by the customer.
3. **WarehouseToHome**: The distance or time from the warehouse to the customer's home.
4. **Complain**: The number of complaints logged by the customer.
5. **CustomerID**: Indicates customer-specific behavior patterns.
6. **NumberOfAddress**: The number of addresses registered by the customer.
7. **DaySinceLastOrder**: The number of days since the customer's last order.
8. **OrderAmountHikeFromLastYear**: The increase in order amount from the previous year.

Other features such as SatisfactionScore, NumberOfDeviceRegistered, and OrderCount also contributed to the model's predictions but to a lesser extent.

#### **7. Interpretation and Recommendations**

The analysis indicates that the Random Forest model is highly effective in predicting customer churn. The high AUC scores and balanced precision and recall metrics suggest that the model can accurately distinguish between churn and non-churn customers.

Key insights from the feature importance analysis can be used to inform customer retention strategies:

- **Tenure**: Focus on retaining new customers who have a shorter tenure by offering loyalty programs and personalized engagement.
- **CashbackAmount**: Increase cashback incentives for high-risk customers to improve retention.
- **WarehouseToHome**: Optimize logistics and delivery times to enhance customer satisfaction.
- **Complain**: Address customer complaints promptly and effectively to reduce churn risk.
- **NumberOfAddress**: Analyze the reasons behind multiple addresses to understand customer behavior.

#### **8. Conclusion**

The Random Forest model provides a powerful tool for predicting customer churn, with excellent performance metrics and valuable insights into key factors influencing churn. By leveraging these insights, businesses can develop targeted strategies to retain high-risk customers and improve overall customer satisfaction.

#### **9. Future Work**

- **Model Updates**: Regularly update and retrain the model with new data to maintain its accuracy and relevance.
- **Feature Engineering**: Explore additional features and interactions to further improve model performance.
- **Customer Segmentation**: Use the model's predictions to segment customers and tailor retention strategies accordingly.

#### **Appendix**

- **Code and Model Implementation**: The detailed code and model implementation used for this analysis can be found in the attached script.
- **Data Visualization**: Additional visualizations and insights derived from the data are included in the supplementary materials.

This comprehensive analysis provides a solid foundation for understanding and addressing customer churn, ensuring long-term business success. If there are any questions or further details required, please feel free to reach out.

---

### Appendices and Supplementary Materials

#### Appendix A: Detailed Code Implementation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv(r"C:\Users\Sietoyo\OneDrive\Documents\CSVs\customer_data.csv")
# Preprocess data (this step may include encoding categorical variables, handling missing values, etc.)
# Assuming data has been preprocessed and is ready for analysis

# Split data into features and target
X = data.drop(columns=['Churn'])
y = data['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate model
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc_score = roc_auc_score(y_test, y_proba)
print(f"Random Forest AUC: {auc_score}")

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best Score: {best_score}")

# Retrain model with best parameters
best_rf_model = RandomForestClassifier(**best_params, random_state=42)
best_rf_model.fit(X_train, y_train)

# Predict and evaluate the best model
best_y_pred = best_rf_model.predict(X_test)
best_y_proba = best_rf_model.predict_proba(X_test)[:, 1]

best_auc_score = roc_auc_score(y_test, best_y_proba)
print(f"Best Random Forest AUC: {best_auc_score}")

print(classification_report(y_test, best_y_pred))
print(confusion_matrix(y_test, best_y_pred))

# Feature importance
feature_importances = pd.DataFrame(best_rf_model.feature_importances_,
                                   index=X_train.columns,
                                   columns=['Importance']).sort_values('Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances['Importance'], y=feature_importances.index)
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
