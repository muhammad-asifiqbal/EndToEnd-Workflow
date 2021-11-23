from DataManipulation import  Datasets
from Graph import Plots
from sklearn.model_selection import train_test_split
import mlflow
import pandas as pd
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import cloudpickle
import sklearn
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

class Traintest(Plots,Datasets):
    def train(self,data): 
      train, test = train_test_split(data)
      X_train = train.drop(["quality"], axis=1)
      X_test = test.drop(["quality"], axis=1)
      y_train = train.quality
      y_test = test.quality
      return X_train,X_test,y_train,y_test

# Model Selection...................

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]


object1= Traintest()
Argu1= object1.dataImport()
Argu2= object1.graph(Argu1)
X_train,X_test,y_train,y_test = object1.train(Argu2)

with mlflow.start_run(run_name='asif_random_forest'):
  n_estimators = 10
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)
  # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  print("Accuracy:",auc_score)
  mlflow.log_param('n_estimators', n_estimators)
  #print("modal type:",type(model))
  wrappedModel = SklearnModelWrapper(model) # Calling the class SklearnModelWrapper(mlflow.pyfunc.PythonModel)
  
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  
  conda_env =  _mlflow_conda_env(
      additional_conda_deps=None,
      additional_conda_channels=None)
  
