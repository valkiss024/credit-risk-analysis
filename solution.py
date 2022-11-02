# Import relevant libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

# EDA - EXPLORATORY DATA ANALYSIS

# Load in dataset

credit_data = pd.read_csv('credit_risk_dataset.csv')

# INVESTIGATE DATASET

# Print information about the size of the data set (observations / features)
print(f'The dataset contains {credit_data.shape[0]} observations and {credit_data.shape[1]} features.')
# The dataset contains 32581 observations and 12 features.

# Check the names of the columns
columns = credit_data.columns
# print(columns)

# Inspect the first few rows of the data:
# print(credit_data.head())

# Output descriptive statistics
# print(credit_data.describe())

# Normalize person_income:
# min_max_scaler = MinMaxScaler()
# credit_data['person_income_normalized'] = min_max_scaler.fit_transform(credit_data[['person_income']])
# credit_data.drop(credit_data[credit_data['person_age'] >= 100].index, inplace=True)
# plt.boxplot(credit_data['person_age'])
# plt.boxplot(credit_data['person_income'])
# plt.show()

# Check for missing values and data types in the DataFrame
# print(credit_data.info())
# Types of data in the DataFrame: int, float, string  --> Both Numerical and Categorical data present
# The number of values in each column differs --> Missing values in certain columns

# Unique values that are present in each feature:
# for i in columns:
    # print(i, len(credit_data[i].unique()))

# Check the number / ratio of missing values in each column
"""print(pd.DataFrame({
    'Count': credit_data.isnull().sum(),
    '%': round(credit_data.isnull().sum() / len(credit_data) * 100, 2)
}))"""
# Missing values are present in the 'Employment Length' and 'Loan Interest Rate' features
# Missing values will be handled at the data pre-processing stage

# Inspect class distribution
# The target variable in the dataset : 'loan_status'
# The class label is either 0 (Loan status = False) or 1 (Loan status = True)
# print(credit_data['loan_status'].value_counts())

# Visualize class distribution for target feature
# plt.pie(credit_data['loan_status'].value_counts(), autopct=lambda x: f'{round(x, 2)}%')
# plt.title('Proportion of Loan Status')
# plt.legend(['Not Granted (0)', 'Granted (1)'], bbox_to_anchor=(1.2, 1), loc='upper right')
# plt.show()
# 78.18% of applicants have been denied for loan, while 21.82% of them have been granted the loan

# Visualize class distribution between age groups


# sns.boxplot(credit_data, x='loan_status', y='person_age')
# sns.boxplot(credit_data, x='loan_status', y='person_income_normalized')
# plt.show()

