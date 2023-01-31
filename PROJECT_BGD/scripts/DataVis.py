import joblib
import pandas as pd
from shap.plots import *
import shap


def BuildExplainer():
    pass
# load saved model
model = joblib.load("../models/model_rf.joblib")
data_test = pd.read_csv("../data/Training/Encoded_test.csv")
# Build a TreeExplainer and compute Shaplay Values
data_reduce = data_test[:20]
data_reduce.shape
# Build a TreeExplainer
Explainer = shap.TreeExplainer(model)
shap_values = Explainer.shap_values(data_reduce)

def SpecificPoint():
    pass
# Visualize explanations for a specific point of your data set
data_test.loc[[200]]
# Calculate Shap values
choosen_instance = data_test.loc[[200]]
shap_values1 = Explainer.shap_values(choosen_instance)
shap.initjs()
shap.force_plot(Explainer.expected_value[1], shap_values1[1], choosen_instance)

def AllPoints():
    pass
shap.initjs()
shap.force_plot(Explainer.expected_value[1], shap_values[1], data_reduce)

def EachClass():
    pass
# Visualize explanations for all points of your data set at once
shap.summary_plot(shap_values, data_reduce, plot_type='bar')