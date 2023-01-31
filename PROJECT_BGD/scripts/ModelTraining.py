import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
from sklearn.model_selection import train_test_split
import mlflow
import joblib
from joblib import dump, load
import matplotlib.pyplot as plt

def ModelTrain():
    pass

# Start an MLflow run
mlflow.start_run()

# Read in encoded train and test data
train_data = pd.read_csv("../data/Training/Encoded_train.csv")
# Log the number of rows and columns in the data as a metric
mlflow.log_metric("rows", train_data.shape[0])
mlflow.log_metric("columns", train_data.shape[1])

# Split train data into input and output variables (split the data into features and target)
X = train_data.drop('TARGET', axis=1)
y = train_data["TARGET"]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#define the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Log the number of trees in the model as a parameter
mlflow.log_param("n_estimators", rf.n_estimators)

#fit the model
rf.fit(X_train, y_train)


# Make predictions on test set
y_pred = rf.predict(X_test)

# Print accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
acc = accuracy_score(y_test, y_pred)

# Log the accuracy and precision as metrics
mlflow.log_metric("Accuracy", acc)
mlflow.log_metric("precision", precision_score(y_test, y_pred))
# Calculate recall score
recall = recall_score(y_test, y_pred)

# Log recall as a metric
mlflow.log_metric("recall", recall)

# Log the maximum depth of the trees as a parameter
mlflow.log_param("max_depth", rf.max_depth)

# Plot feature importances
importance = rf.feature_importances_
importance = pd.DataFrame(importance, index=X_train.columns, columns=["Importance"])
importance = importance.sort_values("Importance", ascending=False)
plt.bar(importance.index, importance["Importance"])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance Plot")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("feature_importance.png")

# Log the feature importance plot as an artifact
mlflow.log_artifact("feature_importance.png")
# save model using joblib
joblib.dump(rf, "../models/model_rf.joblib")

# End the MLflow run
mlflow.end_run()

