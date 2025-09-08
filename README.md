# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step-1

Load dataset, drop unnecessary columns, and convert categorical values into numeric codes.

Step-2

Split data into features (X) and target (Y), then initialize model parameters (θ).

Step-3

Train the model using gradient descent with the sigmoid function to optimize θ.

Step-4

Predict outcomes on training and new data, evaluate accuracy, and display results.

## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: CHARUKESH S
RegisterNumber:  212224230044
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
print("Name: CHARUKESH S")
print("Reg No: 212224230044")
dataset

dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)


dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
print("Name: CHARUKESH S")
print("Reg No: 212224230044")
dataset.dtypes


dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
print("Name: CHARUKESH S")
print("Reg No: 212224230044")
dataset


X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print("Name: CHARUKESH S")
print("Reg No: 212224230044")
Y

theta=np.random.randn(X.shape[1])
y=Y
```
```python
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))


def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)


accuracy=np.mean(y_pred.flatten()==y)
print("Name: CHARUKESH S")
print("Reg No: 212224230044")
print("Accuracy:",accuracy)
print(y_pred)

print("Name: CHARUKESH S")
print("Reg No: 212224230044")
print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```

## Output:
<img width="1000" height="409" alt="image" src="https://github.com/user-attachments/assets/bad5159b-0d46-48ab-9966-5f5d7c07443a" />

<img width="382" height="297" alt="image" src="https://github.com/user-attachments/assets/1e0c1c41-b3f6-4020-9409-51af04441e61" />

<img width="882" height="422" alt="image" src="https://github.com/user-attachments/assets/d017acfb-76c3-4647-8da9-dfd4307625b2" />

<img width="734" height="233" alt="image" src="https://github.com/user-attachments/assets/ce2872d3-f01b-4e74-b661-232c33d2ce62" />

<img width="619" height="168" alt="image" src="https://github.com/user-attachments/assets/c48375b7-d349-4c52-8fca-c0b4f09f6014" />

<img width="789" height="152" alt="image" src="https://github.com/user-attachments/assets/dbc51f2c-20a9-41a0-918e-212f8f0fd184" />

<img width="769" height="33" alt="image" src="https://github.com/user-attachments/assets/7e62faaf-83c4-4ff5-a4e1-39f36ac3bdc3" />


<img width="821" height="36" alt="image" src="https://github.com/user-attachments/assets/1e98466f-d213-42a6-9a45-e643fd33289c" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

