import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('sonar.all-data.csv',header=None)

data.head(5)
data.shape
data.isnull().values.any()
data.isnull().sum()
data.isnull().sum().sum()
data.head()
data.info()
data.groupby(60).mean()
data[60].value_counts()
X = data.drop(columns=60,axis = 1)
Y = data[60]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.18,stratify=Y,random_state=2)
X.shape,X_train.shape,X_test.shape
X_train.shape,Y_train.shape,Y_test.shape
model = LogisticRegression()
model.fit(X_train,Y_train)
X_train_predict = model.predict(X_train)
trainingdata_accuracy = accuracy_score(X_train_predict,Y_train)
print("X_train prediction = ",trainingdata_accuracy)
X_test_prediction = model.predict(X_test)
testdata_accuracty = accuracy_score(X_test_prediction, Y_test)
print("testdata predtion =", testdata_accuracty)
a = (0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055)
newdata = np.asarray(a)
print(newdata)
newdata = newdata.reshape(1,-1)
print(newdata)
newdata_prediction = model.predict(newdata)
print('newdata prediction = ',newdata_prediction )