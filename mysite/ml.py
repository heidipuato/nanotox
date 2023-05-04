import json # will be needed for saving preprocessing details
import numpy as np # for data manipulation
import pandas as pd # for data manipulation
from sklearn.model_selection import train_test_split # will be used for data split
from sklearn.preprocessing import LabelEncoder # for preprocessing
from imblearn.over_sampling import SMOTE #for imbalance
from xgboost import XGBClassifier #for training
import joblib # for saving algorithm and preprocessing objects

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

#load data
dataset = pd.read_csv(r'C:\Users\Heidi\Documents\nanotox\mysite\IC50.csv')

#normalize data
for column in dataset.columns:
    dataset[column] = (dataset[column] - dataset[column].min()) / (dataset[column].max() - dataset[column].min())
print(dataset)

y = dataset['TEi(cj)_nw']
X = dataset.drop(['TEi(cj)_nw'], axis=1)

# split data into train and test sets
seed = 7
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

#SMOTE
# oversample = SMOTE()
# X_train, y_train = oversample.fit_resample(X_train, y_train)

# le = LabelEncoder()
# y_train = le.fit_transform(y_train)

#model training
model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
y_pred = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test,y_pred)*100
precision = precision_score(y_test,y_pred)*100
f1 = f1_score(y_test,y_pred)*100
recall = recall_score(y_test,y_pred)*100
confusion_mat = confusion_matrix(y_test,y_pred)

# Printing the Results
print("Accuracy for XGBoost is:",accuracy)
print("Precision for XGBoost is:",precision)
print("F1-score for XGBoost is:",f1)
print("Recall for XGBoost is:",recall)
print("Confusion Matrix")
print(confusion_mat)

# save model
# joblib.dump(train_mode, "./train_mode.joblib", compress=True)
# joblib.dump(encoders, "./encoders.joblib", compress=True)
joblib.dump(model, "./model.sav", compress=True)
