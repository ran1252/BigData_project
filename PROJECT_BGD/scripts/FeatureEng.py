import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlflow
import os


def FeatureEng():
    pass

# Start an MLflow run
mlflow.start_run()


# Read in cleaned train and test data
train_data = pd.read_csv("../data/interim/cleaned_train.csv")
test_data = pd.read_csv("../data/interim/cleaned_test.csv")
train_data.drop("SK_ID_CURR", axis = 1, inplace = True)
test_data.drop("SK_ID_CURR", axis = 1, inplace = True)


# Log number of rows and columns in train data
mlflow.log_metric("train_rows", train_data.shape[0])
mlflow.log_metric("train_columns", train_data.shape[1])
# Create a label encoder object
le = LabelEncoder()

# fit the label encoder to all non-numeric columns in train dataframe
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        train_data[col] = le.fit_transform(train_data[col])

# Log number of rows and columns in test data
mlflow.log_metric("test_rows", test_data.shape[0])
mlflow.log_metric("test_columns", test_data.shape[1])

# Create a label encoder object
le_t = LabelEncoder()

# fit the label encoder to all non-numeric columns in train dataframe
for col in test_data.columns:
    if test_data[col].dtype == 'object':
        test_data[col] = le_t.fit_transform(test_data[col])
        

# Save the processed train and test dataframes as csv files
train_data.to_csv('../data/Training/Encoded_train.csv', index=False)
test_data.to_csv('../data/Training/Encoded_test.csv', index=False)

# End the MLflow run
mlflow.end_run()

