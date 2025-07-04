---
title: "HR Attrition Analysis"
date: 2025-07-02
draft: false
summary: "Analysis of employee attrition using Python and logistic regression."
tags: ["Python", "Pandas", "Jupyter", "Logistic Regression"]
---

## Exploratory Data Analysis (EDA)

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/HR-Employee-Attrition.csv')

# Quick preview
print(df.head())
print(df.info())
print(df['Attrition'].value_counts(normalize=True))
```

       Age Attrition     BusinessTravel  DailyRate              Department  \
    0   41       Yes      Travel_Rarely       1102                   Sales
    1   49        No  Travel_Frequently        279  Research & Development
    2   37       Yes      Travel_Rarely       1373  Research & Development
    3   33        No  Travel_Frequently       1392  Research & Development
    4   27        No      Travel_Rarely        591  Research & Development

       DistanceFromHome  Education EducationField  EmployeeCount  EmployeeNumber  \
    0                 1          2  Life Sciences              1               1
    1                 8          1  Life Sciences              1               2
    2                 2          2          Other              1               4
    3                 3          4  Life Sciences              1               5
    4                 2          1        Medical              1               7

       ...  RelationshipSatisfaction StandardHours  StockOptionLevel  \
    0  ...                         1            80                 0
    1  ...                         4            80                 1
    2  ...                         2            80                 0
    3  ...                         3            80                 0
    4  ...                         4            80                 1

       TotalWorkingYears  TrainingTimesLastYear WorkLifeBalance  YearsAtCompany  \
    0                  8                      0               1               6
    1                 10                      3               3              10
    2                  7                      3               3               0
    3                  8                      3               3               8
    4                  6                      3               3               2

      YearsInCurrentRole  YearsSinceLastPromotion  YearsWithCurrManager
    0                  4                        0                     5
    1                  7                        1                     7
    2                  0                        0                     0
    3                  7                        3                     0
    4                  2                        2                     2

    [5 rows x 35 columns]
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1470 entries, 0 to 1469
    Data columns (total 35 columns):
     #   Column                    Non-Null Count  Dtype
    ---  ------                    --------------  -----
     0   Age                       1470 non-null   int64
     1   Attrition                 1470 non-null   object
     2   BusinessTravel            1470 non-null   object
     3   DailyRate                 1470 non-null   int64
     4   Department                1470 non-null   object
     5   DistanceFromHome          1470 non-null   int64
     6   Education                 1470 non-null   int64
     7   EducationField            1470 non-null   object
     8   EmployeeCount             1470 non-null   int64
     9   EmployeeNumber            1470 non-null   int64
     10  EnvironmentSatisfaction   1470 non-null   int64
     11  Gender                    1470 non-null   object
     12  HourlyRate                1470 non-null   int64
     13  JobInvolvement            1470 non-null   int64
     14  JobLevel                  1470 non-null   int64
     15  JobRole                   1470 non-null   object
     16  JobSatisfaction           1470 non-null   int64
     17  MaritalStatus             1470 non-null   object
     18  MonthlyIncome             1470 non-null   int64
     19  MonthlyRate               1470 non-null   int64
     20  NumCompaniesWorked        1470 non-null   int64
     21  Over18                    1470 non-null   object
     22  OverTime                  1470 non-null   object
     23  PercentSalaryHike         1470 non-null   int64
     24  PerformanceRating         1470 non-null   int64
     25  RelationshipSatisfaction  1470 non-null   int64
     26  StandardHours             1470 non-null   int64
     27  StockOptionLevel          1470 non-null   int64
     28  TotalWorkingYears         1470 non-null   int64
     29  TrainingTimesLastYear     1470 non-null   int64
     30  WorkLifeBalance           1470 non-null   int64
     31  YearsAtCompany            1470 non-null   int64
     32  YearsInCurrentRole        1470 non-null   int64
     33  YearsSinceLastPromotion   1470 non-null   int64
     34  YearsWithCurrManager      1470 non-null   int64
    dtypes: int64(26), object(9)
    memory usage: 402.1+ KB
    None
    Attrition
    No     0.838776
    Yes    0.161224
    Name: proportion, dtype: float64

```python
# Visualization
sns.countplot(data=df, x='Attrition')
plt.title('Employee Attrition Count')
plt.show()
```

![png](/images/hr-attrition-analysis/output_1.png)

```python
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Department', hue='Attrition', palette='Set2')
plt.title('Attrition by Department')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
```

![png](/images/hr-attrition-analysis/output_2.png)

```python
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='JobRole', hue='Attrition', palette='Set3')
plt.title('Attrition by Job Role')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

```

![png](/images/hr-attrition-analysis/output_3.png)

```python
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='JobSatisfaction', hue='Attrition', palette='coolwarm')
plt.title('Attrition by Job Satisfaction Level')
plt.xlabel('Job Satisfaction (1=Low, 4=High)')
plt.tight_layout()
plt.show()

```

![png](/images/hr-attrition-analysis/output_4.png)

## Feature Selection & Preprocessing

```python
%pip install scikit-learn
```

    Requirement already satisfied: scikit-learn in /Users/millie/opt/anaconda3/envs/tensorflow/lib/python3.10/site-packages (1.7.0)
    Requirement already satisfied: scipy>=1.8.0 in /Users/millie/opt/anaconda3/envs/tensorflow/lib/python3.10/site-packages (from scikit-learn) (1.10.1)
    Requirement already satisfied: numpy>=1.22.0 in /Users/millie/opt/anaconda3/envs/tensorflow/lib/python3.10/site-packages (from scikit-learn) (1.24.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /Users/millie/opt/anaconda3/envs/tensorflow/lib/python3.10/site-packages (from scikit-learn) (3.6.0)
    Requirement already satisfied: joblib>=1.2.0 in /Users/millie/opt/anaconda3/envs/tensorflow/lib/python3.10/site-packages (from scikit-learn) (1.5.1)
    Note: you may need to restart the kernel to use updated packages.

```python
from sklearn.preprocessing import LabelEncoder

df_model = df[['Attrition', 'Department', 'JobRole', 'JobSatisfaction', 'MonthlyIncome', 'OverTime', 'BusinessTravel', 'Age', 'DistanceFromHome', 'YearsAtCompany', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'PerformanceRating']]
df_model = df_model.dropna()

le = LabelEncoder()
df_model['Attrition'] = le.fit_transform(df_model['Attrition'])  # Yes = 1, No = 0

# One-hot encode categorical variables
df_model = pd.get_dummies(df_model, drop_first=True)

```

## Logistic Regression Modeling

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

X = df_model.drop('Attrition', axis=1)
y = df_model['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model = LogisticRegression(class_weight='balanced')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

```

    [[171  84]
     [ 12  27]]
                  precision    recall  f1-score   support

               0       0.93      0.67      0.78       255
               1       0.24      0.69      0.36        39

        accuracy                           0.67       294
       macro avg       0.59      0.68      0.57       294
    weighted avg       0.84      0.67      0.72       294



    /Users/millie/opt/anaconda3/envs/tensorflow/lib/python3.10/site-packages/sklearn/linear_model/_logistic.py:470: ConvergenceWarning: lbfgs failed to converge after 100 iteration(s) (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT

    Increase the number of iterations to improve the convergence (max_iter=100).
    You might also want to scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(

```python
coeff_df = pd.DataFrame(model.coef_[0], index=X.columns, columns=["Coefficient"])
print(coeff_df.sort_values("Coefficient", ascending=False))

```

                                       Coefficient
    OverTime_Yes                          0.367823
    PerformanceRating                     0.152581
    Department_Sales                      0.146145
    BusinessTravel_Travel_Frequently      0.128062
    JobRole_Sales Representative          0.095459
    RelationshipSatisfaction              0.092712
    YearsAtCompany                        0.091397
    JobRole_Laboratory Technician         0.068907
    JobRole_Sales Executive               0.048670
    WorkLifeBalance                       0.025411
    DistanceFromHome                      0.025401
    Age                                   0.015799
    JobRole_Human Resources               0.013209
    MonthlyIncome                        -0.000043
    JobRole_Manager                      -0.006343
    BusinessTravel_Travel_Rarely         -0.009459
    JobRole_Research Scientist           -0.019459
    JobRole_Research Director            -0.022910
    JobRole_Manufacturing Director       -0.053768
    TrainingTimesLastYear                -0.084642
    Department_Research & Development    -0.100469
    EnvironmentSatisfaction              -0.105870
    TotalWorkingYears                    -0.124786
    JobSatisfaction                      -0.179750

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

```

![png](/images/hr-attrition-analysis/output_5.png)

```python
import pandas as pd
import numpy as np

coefficients = model.coef_[0]
features = X.columns
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
coef_df = coef_df.sort_values(by='Coefficient', key=np.abs, ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm')
plt.title('Logistic Regression Coefficients (Feature Importance)')
plt.tight_layout()
plt.show()

```

    /var/folders/sj/6jz7w0pd3vz6kl3h7kwrx37r0000gn/T/ipykernel_71616/1017806495.py:10: FutureWarning:

    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

      sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm')

![png](/images/hr-attrition-analysis/output_6.png)

## Visualizations & Recommendations

```python
corr = df_model.corr()
plt.figure(figsize=(30, 30))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

```

![png](/images/hr-attrition-analysis/output_7.png)

## Dashboard

```python
# === Dashboard Layout ===
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# 2. Feature Importance
sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm', ax=axes[1])
axes[1].set_title('Feature Importance')
axes[1].set_xlabel('Coefficient')

# 3. Correlation Heatmap (shown as image in axes[2])
sns.heatmap(corr, annot=False, cmap='coolwarm', ax=axes[2], cbar=True)
axes[2].set_title('Feature Correlation Heatmap')

plt.tight_layout()
plt.show()
```

    /var/folders/sj/6jz7w0pd3vz6kl3h7kwrx37r0000gn/T/ipykernel_71616/1014875411.py:12: FutureWarning:

    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.

      sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm', ax=axes[1])

![png](/images/hr-attrition-analysis/output_8.png)

## Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf_model.predict(X_test)

```

```python
# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Confusion Matrix:")
print(cm_rf)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

```

    Confusion Matrix:
    [[252   3]
     [ 38   1]]
    Classification Report:
                  precision    recall  f1-score   support

               0       0.87      0.99      0.92       255
               1       0.25      0.03      0.05        39

        accuracy                           0.86       294
       macro avg       0.56      0.51      0.49       294
    weighted avg       0.79      0.86      0.81       294

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Get feature importances and sort
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=True)

# Plot
plt.figure(figsize=(10, 8))
feat_importances.plot(kind='barh', color='skyblue')
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

```

![png](/images/hr-attrition-analysis/output_9.png)

```python
y_probs_rf = rf_model.predict_proba(X_test)[:, 1]
threshold = 0.2  # try lowering from 0.5
y_pred_adj = (y_probs_rf >= threshold).astype(int)

print(confusion_matrix(y_test, y_pred_adj))
print(classification_report(y_test, y_pred_adj))

```

    [[207  48]
     [ 16  23]]
                  precision    recall  f1-score   support

               0       0.93      0.81      0.87       255
               1       0.32      0.59      0.42        39

        accuracy                           0.78       294
       macro avg       0.63      0.70      0.64       294
    weighted avg       0.85      0.78      0.81       294

## Model Comparision

| Aspect                    | Logistic Regression              | Random Forest (Tuned Threshold) |
| ------------------------- | -------------------------------- | ------------------------------- |
| **Accuracy**              | 67%                              | 78%                             |
| **Recall (attrition)**    | 69% (better at catching leavers) | 59% (slightly lower recall)     |
| **Precision (attrition)** | 24% (more false alarms)          | 32% (fewer false alarms)        |
| **Overall balance**       | Less balanced, lower accuracy    | Better balance, higher accuracy |
| **F1-score (attrition)**  | 0.36                             | 0.42                            |

#### Which is better?

Random Forest with threshold adjustment shows better overall accuracy and better precision for the attrition class, with only a modest decrease in recall.

It has a higher F1-score for attrition (0.42 vs 0.36), meaning better balance between precision and recall.

Random Forest usually captures more complex patterns, which can improve generalization.

## Summary

I conducted an exploratory data analysis (EDA) on an HR dataset to uncover patterns in employee attrition and identify key factors influencing turnover. Using Python, Pandas, and Seaborn, I analyzed the dataset to derive actionable insights and built a logistic regression model to predict attrition likelihood. The project culminated in data-driven recommendations to reduce turnover in high-risk departments.

### Key Findings

#### 1. Attrition Rate

The overall attrition rate in the dataset was 16.12%, with 83.88% of employees retaining their positions.

Visualizations revealed that Sales and Research & Development departments had the highest attrition rates, suggesting a need for targeted retention strategies.

#### 2. Department & Job Role Impact

Sales and R&D showed the highest attrition, while Human Resources had the lowest.

Within job roles, Sales Representatives and Laboratory Technicians were most likely to leave, whereas Managers and Research Directors had higher retention.

#### 3. Satisfaction & Work-Life Balance

Employees with lower Job Satisfaction and Work-Life Balance scores were more prone to attrition.

Overtime was a significant factor—employees working overtime had a higher likelihood of leaving.

#### 4. Predictive Modeling

A logistic regression model was developed to predict attrition based on:

Department

Job Role

Job Satisfaction

Work-Life Balance

Overtime Status

The model helped identify at-risk employees, enabling proactive HR interventions.

### Recommendations

**Enhance Employee Engagement**: Implement mentorship programs and career development opportunities in high-attrition departments.

**Improve Work-Life Balance**: Reduce overtime demands and offer flexible work arrangements.

**Targeted Retention Strategies**: Focus on Sales and R&D teams with incentives, recognition programs, and satisfaction surveys.

**Predictive HR Analytics**: Use the logistic regression model to flag at-risk employees for early intervention.

### Tools & Techniques

Python (Pandas, Seaborn, Matplotlib) – Data cleaning, visualization, and statistical analysis.

Logistic Regression – Predictive modeling for attrition likelihood.

Excel – Supplemental data validation and reporting.
