import pandas as pd
import joblib
import mlflow

def MakePredictions():
    pass

# Load model from file
rf = joblib.load("../models/model_rf.joblib")

# Read in new data for prediction
test_data = pd.read_csv("../data/training/Encoded_test.csv")

# Start an MLFlow run
mlflow.start_run()

# Make predictions on new data
y_pred = rf.predict(test_data)


# Create a dataframe with predictions
predictions_df = pd.DataFrame(y_pred, columns=["TARGET PREDICTED"])

# Concatenate the predictions column to the test data
test_data_with_predictions = pd.concat([predictions_df,test_data], axis=1)

# Save the combined data to csv file
test_data_with_predictions.to_csv("../data/predictions/Test_Predictions.csv", index=False)


mlflow.end_run()
