
import h2o
h2o.init(max_mem_size = 2)            #uses all cores by default
h2o.remove_all()
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
higgs = h2o.import_file('higgs_boston_train.csv')
higgs.head()
higgs.shape
higgs_df = higgs.as_data_frame(use_pandas=True)
higgs_df['Label'].value_counts()
higgs.describe()
train, valid, test = higgs.split_frame([0.6, 0.2], seed = 2019)
higgs_X = higgs.col_names[1: -1]
higgs_y = higgs.col_names[-1]
higgs_model_v1 = H2ODeepLearningEstimator(model_id = 'higgs_v1', epochs = 1, variable_importances = True)
higgs_model_v1.train(higgs_X, higgs_y, training_frame = train, validation_frame = valid)
print(higgs_model_v1)
var_df = pd.DataFrame(higgs_model_v1.varimp(), columns = ['Variable', 'Relative Importance', 'Scaled Importance', 'Percentage'])
print(var_df.shape)
var_df.head(10)
higgs_v1_df = higgs_model_v1.score_history()
higgs_v1_df
plt.plot(higgs_v1_df['training_classification_error'], label="training_classification_error")
plt.plot(higgs_v1_df['validation_classification_error'], label="validation_classification_error")
plt.title("Higgs Deep Learner")
plt.legend();
pred = higgs_model_v1.predict(test[1:-1]).as_data_frame(use_pandas=True)
test_actual = test.as_data_frame(use_pandas=True)['Label']
(test_actual == pred['predict']).mean()
higgs_model_v2 = H2ODeepLearningEstimator(model_id = 'higgs_v2', hidden = [32, 32, 32], epochs = 1000000,
                                           score_validation_samples = 10000, stopping_rounds = 2, stopping_metric = 'misclassification', 
                                           stopping_tolerance = 0.01)
higgs_model_v2.train(higgs_X, higgs_y, training_frame = train, validation_frame = valid)
higgs_v2_df = higgs_model_v2.score_history()
higgs_v2_df
plt.plot(higgs_v2_df['training_classification_error'], label="training_classification_error")
plt.plot(higgs_v2_df['validation_classification_error'], label="validation_classification_error")
plt.title("Higgs Deep Learner (Early Stop)")
plt.legend();
pred = higgs_model_v2.predict(test[1:-1]).as_data_frame(use_pandas=True)
test_actual = test.as_data_frame(use_pandas=True)['Label']
(test_actual == pred['predict']).mean()
higgs_model_v2.varimp_plot();
from h2o.automl import H2OAutoML

aml = H2OAutoML(max_models = 10, max_runtime_secs=100, seed = 1)
aml.train(higgs_X, higgs_y, training_frame = train, validation_frame = valid)
aml.leaderboard
"""
AutoML has built 5 models inlcuding GLM, DRF (Distributed Random Forest) and XRT (Extremely Randomized Trees) and two stacked ensemble models (the 2nd and 3rd) and the best model is XRT.

It turns out, my proud deep learning models are not even on the leaderboard.
"""
