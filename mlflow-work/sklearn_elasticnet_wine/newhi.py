#!/usr/bin/env python
# coding: utf-8

# In[94]:


import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

df = pd.read_csv ('heart.csv')


# In[95]:


df.head()


# In[96]:


df.target.value_counts()


# In[97]:


y = df.target.values
x_data = df.drop(['target'], axis = 1)


# In[98]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
X_std = StandardScaler().fit_transform(x_data)
dfNorm = pd.DataFrame(X_std, index=df.index, columns=df.columns[0:13])
# # add non-feature target column to dataframe
dfNorm['target'] = df['target']
dfNorm.head(10)

X = dfNorm.iloc[:,0:13].values
y = dfNorm.iloc[:,13].values


# In[99]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[100]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

Log = LogisticRegression()

parameters = {'C': [.01,2, 5, 10, 15, 20,0.1] }

log_regressor = GridSearchCV(Log, parameters, scoring='neg_mean_squared_error' ,cv =5)
log_regressor.fit(x_train, y_train)
log_regressor.best_params_


# In[101]:


model1 = LogisticRegression(C=0.1)
model1.fit(x_train,y_train)
accuracy1 = model1.score(x_test,y_test)
print('Logistic Regression Accuracy -->',((accuracy1)*100))


# In[108]:


import mlflow
import mlflow.sklearn
from urllib.parse import parse_qsl, urljoin, urlparse


# In[111]:


with mlflow.start_run():
    
    model1 = LogisticRegression(C=0.5)
    model1.fit(x_train,y_train)
    y_pred = model1.predict(x_test)
    accuracy1 = model1.score(x_test,y_test)
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("accuracy", accuracy1)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    mlflow.sklearn.log_model(model1, "model")
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(model, "model", registered_model_name="Logistic Regression Model")
    else:
        mlflow.sklearn.log_model(model1, "model")


# In[113]:


import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts

if __name__ == "__main__":
    # Log a parameter (key-value pair)
    log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts("outputs")


# In[ ]:




