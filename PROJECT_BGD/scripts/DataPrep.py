
import pandas as pd
import os
import mlflow


def DataPrep():
    
    pass
  

# Start an MLflow run
mlflow.start_run()

train_data=pd.read_csv("../data/external/application_train.csv")
test_data=pd.read_csv("../data/external/application_test.csv")

# Log the number of rows and columns of the train and test data
mlflow.log_metric("train_rows", train_data.shape[0])
mlflow.log_metric("train_columns", train_data.shape[1])
mlflow.log_metric("test_rows", test_data.shape[0])
mlflow.log_metric("test_columns", test_data.shape[1])
"""
Data review 
Review the data to ensure that it is in the correct format and that there are no errors or missing values

"""
 
train_data.head()
test_data.head()

train_data.info()
test_data.info()

# Check columns types
print(train_data.dtypes)
print(test_data.dtypes)

# Count missing values for each column
train_missing = train_data.isnull().sum()
test_missing = test_data.isnull().sum()


mlflow.log_metric("train_missing", train_missing.sum())
mlflow.log_metric("test_missing", test_missing.sum())
# Perform data cleaning and preprocessing tasks

# handling any missing or inconsistent values
train_data.fillna(value=0, inplace=True)
test_data.fillna(value=0, inplace=True)

# removing any duplicate rows
duplicate_rows = train_data[train_data.duplicated()]
train_data.drop_duplicates(inplace=True)

duplicate_rows = test_data[test_data.duplicated()]
test_data.drop_duplicates(inplace=True)

# Log the number of duplicate rows
mlflow.log_metric("train_duplicate_rows", duplicate_rows.shape[0])
mlflow.log_metric("test_duplicate_rows", duplicate_rows.shape[0])

# Save cleaned data to new csv files
train_data.to_csv("../data/interim/cleaned_train.csv", index=False)
test_data.to_csv("../data/interim/cleaned_test.csv", index=False)

# End the MLflow run
mlflow.end_run()

